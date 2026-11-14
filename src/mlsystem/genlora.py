"""Handwritten LoRA and grouped Gaussian RBF adapters, no PEFT/transformers."""
import math
import torch
from torch import nn
from .components import Linear

class BasisGenerator(nn.Module):
    """One shared latent vector; each rank/group has its own scalar function."""
    def __init__(self, width, rank, groups=4, centers=15, zero=False):
        super().__init__()
        if width % groups:
            raise ValueError("width must be divisible by groups")
        self.width, self.groups = width, groups
        self.latent = nn.Parameter(torch.empty(groups, width // groups))
        nn.init.kaiming_uniform_(self.latent, a=math.sqrt(5))
        self.base = nn.Parameter(torch.empty(rank, groups))
        self.rbf = nn.Parameter(torch.empty(rank, groups, centers))
        if zero:
            nn.init.zeros_(self.base); nn.init.zeros_(self.rbf)
        else:
            nn.init.normal_(self.base, std=1/math.sqrt(width//groups))
            nn.init.normal_(self.rbf, std=1/math.sqrt(width//groups))
        self.register_buffer("centers", torch.linspace(-3, 3, centers))
        self.bandwidth = 6/(centers-1)

    def forward(self):
        z = self.latent.float()
        centered = z - z.mean(-1, keepdim=True)
        # Epsilon under sqrt is finite even for constant groups.
        # This stabilizes the paper's instance normalization.
        z = centered / (centered.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        features = torch.exp(-((z[..., None] - self.centers)/self.bandwidth).square())
        nonlinear = torch.einsum("gdk,rgk->rgd", features, self.rbf)
        residual = self.base[..., None] * (z * torch.sigmoid(z))[None, ...]
        return (nonlinear + residual).reshape(self.base.shape[0], self.width)

class AdapterLinear(nn.Module):
    def __init__(self, base, kind, rank, groups=4, centers=15, scale_alpha=16.0):
        super().__init__()
        self.base_layer, self.kind, self.rank = base, kind, rank
        m, n = base.weight.shape
        self.scale = scale_alpha/rank
        if kind == "rbf":
            self.gen_a = BasisGenerator(n, rank, groups, centers=centers)
            # v3 paper initialization: random latent, zero B coefficients.
            # Delta W is exactly zero without zero-variance latent groups.
            self.gen_b = BasisGenerator(m, rank, groups, centers=centers, zero=True)
        elif kind == "lora":
            self.a = nn.Parameter(torch.empty(rank, n))
            self.b = nn.Parameter(torch.zeros(m, rank))
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        else:
            raise ValueError(kind)

    def matrices(self):
        if self.kind == "rbf":
            return self.gen_a(), self.gen_b().T
        return self.a, self.b

    def delta(self):
        a, b = self.matrices()
        return (b @ a) * self.scale

    def forward(self, x):
        a, b = self.matrices()
        return self.base_layer(x) + ((x @ a.T) @ b.T) * self.scale

def attach(model, kind, rank, groups=4, centers=15, scale_alpha=16.0):
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, Linear)]
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, AdapterLinear(module, kind, rank, groups, centers, scale_alpha).to(module.weight.device))
    return [name for name, _ in targets]

def adapter_state(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters()
            if p.requires_grad}

@torch.no_grad()
def merge(model):
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, AdapterLinear)]
    for name, module in targets:
        module.base_layer.weight.add_(module.delta())
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, module.base_layer)
    return model
