"""Supervised OLS initialization of a Gaussian full-vocabulary residual adapter.

Unlike the retired evidence classifier, every output is a next-token logit. OLS
fits the negative token-CE logit gradient on TRAIN examples, not random A values.
Actual subsequent SFT minimizes answer-token cross entropy by steepest descent.
"""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F


def token_error_target(base_logits, correct_tokens):
    if base_logits.ndim != 2 or correct_tokens.shape != base_logits.shape[:1]:
        raise ValueError('Expected logits [N,V] and real next-token labels [N]')
    if torch.any(correct_tokens < 0) or torch.any(correct_tokens >= base_logits.shape[1]):
        raise ValueError('Calibration labels must be real answer tokens, not ignored positions')
    target = -torch.softmax(base_logits.detach().float(), dim=-1)
    target.scatter_add_(1, correct_tokens[:, None], torch.ones_like(correct_tokens[:, None], dtype=target.dtype))
    return target


def gaussian(features, centers, widths):
    distance2 = (features.square().sum(-1, keepdim=True)
                 + centers.square().sum(-1)[None, :] - 2 * features @ centers.T).clamp_min(0)
    return torch.exp(-distance2 / (2 * widths.square().clamp_min(1e-8)[None, :]))


@torch.no_grad()
def orthogonal_select(dictionary, target, count, tolerance=1e-8):
    """Multi-output forward OLS; return original dictionary indices, not new centers.

    Orthogonal directions are used only for candidate scoring. Readout fitting later
    uses the original Gaussian columns. Gain is reduction of the unregularized SSE
    when adding one orthogonal direction across all output columns.
    """
    if dictionary.ndim != 2 or target.ndim != 2 or dictionary.shape[0] != target.shape[0]:
        raise ValueError('Dictionary and target must share calibration rows')
    if not 1 <= count <= dictionary.shape[1]:
        raise ValueError('Invalid number of selected centers')
    if not torch.isfinite(dictionary).all() or not torch.isfinite(target).all():
        raise ValueError('Non-finite OLS calibration data')
    residual_columns = dictionary.float().clone()
    selected, gains, basis = [], [], []
    original_norm = residual_columns.square().sum(0).max().clamp_min(1e-20)
    total_energy = target.float().square().sum()
    for _ in range(count):
        norms = residual_columns.square().sum(0)
        gains_all = (residual_columns.T @ target.float()).square().sum(1) / norms.clamp_min(1e-30)
        gains_all[norms <= tolerance * original_norm] = -torch.inf
        if selected:
            gains_all[selected] = -torch.inf
        index = int(gains_all.argmax())
        gain = gains_all[index]
        if not torch.isfinite(gain) or gain <= tolerance * total_energy.clamp_min(1e-20):
            break
        q = residual_columns[:, index] / norms[index].sqrt()
        selected.append(index)
        gains.append(float(gain))
        basis.append(q)
        # Modified Gram-Schmidt plus a second pass reduces numerical re-selection.
        residual_columns -= q[:, None] * (q @ residual_columns)[None, :]
        Q = torch.stack(basis, dim=1)
        residual_columns -= Q @ (Q.T @ residual_columns)
    if not selected:
        raise RuntimeError('No informative Gaussian candidate under supervised target')
    return selected, gains


class GaussianLogitAdapter(nn.Module):
    def __init__(self, mean, projection, feature_scale, centers, widths, readout):
        super().__init__()
        self.register_buffer('mean', mean.detach().clone())
        self.register_buffer('projection', projection.detach().clone())
        self.register_buffer('feature_scale', feature_scale.detach().clone())
        self.centers = nn.Parameter(centers.detach().clone())
        # Inverse softplus, stable even for a large width.
        widths = widths.detach().clamp_min(1e-3)
        self.raw_widths = nn.Parameter(widths + torch.log(-torch.expm1(-widths)))
        self.readout = nn.Parameter(readout.detach().clone())

    def features(self, hidden):
        return ((hidden.float() - self.mean) @ self.projection) / self.feature_scale

    def forward(self, hidden):
        shape = hidden.shape[:-1]
        features = self.features(hidden.reshape(-1, hidden.shape[-1]))
        widths = F.softplus(self.raw_widths).clamp_min(1e-4)
        response = gaussian(features, self.centers, widths)
        return (response @ self.readout).reshape(*shape, self.readout.shape[1])


@torch.no_grad()
def initialize_supervised(hidden, base_logits, labels, *, dimensions=32,
                          candidates=128, selected_centers=64, ridge=1e-3, seed=7):
    """All rows must come from TRAIN answer positions, with true next-token labels."""
    started = time.perf_counter()
    if hidden.ndim != 2 or hidden.shape[0] != labels.numel() or base_logits.shape[0] != labels.numel():
        raise ValueError('Calibration arrays have incompatible shapes')
    if not 1 <= dimensions <= min(hidden.shape[1], hidden.shape[0]-1):
        raise ValueError('PCA dimensions exceed usable calibration data')
    if not 1 <= selected_centers <= candidates <= hidden.shape[0]:
        raise ValueError('Need selected <= candidates <= TRAIN calibration positions')
    if not 0 < ridge <= 1:
        raise ValueError('ridge must be in (0,1] as a fraction of mean diagonal energy')
    h = hidden.detach().float()
    mean = h.mean(0)
    centered = h - mean
    _, singular_values, vt = torch.linalg.svd(centered, full_matrices=False)
    projection = vt[:dimensions].T.contiguous()
    scale = (singular_values[:dimensions] / math.sqrt(h.shape[0]-1)).clamp_min(1e-3)
    features = (centered @ projection) / scale
    generator = torch.Generator(device=h.device).manual_seed(seed)
    candidate_rows = torch.randperm(h.shape[0], generator=generator, device=h.device)[:candidates]
    centers = features[candidate_rows]
    # A common starting bandwidth gives all candidates the same scoring conditions.
    distances = torch.pdist(centers).float()
    positive = distances[distances > 1e-6]
    width = positive.median() / math.sqrt(2) if positive.numel() else torch.tensor(1., device=h.device)
    widths = width.clamp_min(.1).repeat(candidates)
    dictionary = gaussian(features, centers, widths)
    target = token_error_target(base_logits, labels)
    indices, gains = orthogonal_select(dictionary, target, selected_centers)
    phi = dictionary[:, indices]
    gram = phi.T @ phi
    regularization = ridge * gram.diag().mean().clamp_min(1e-8)
    coefficient = torch.linalg.solve(gram + regularization * torch.eye(len(indices), device=h.device), phi.T @ target)
    fitted = phi @ coefficient
    adapter = GaussianLogitAdapter(mean, projection, scale, centers[indices], widths[indices], coefficient)
    audit = {'target': 'onehot(true next answer token) minus softmax(frozen base logits)',
             'calibration_positions': h.shape[0], 'feature_dimensions': dimensions,
             'candidate_count': candidates, 'requested_centers': selected_centers,
             'selected_centers': len(indices), 'candidate_training_rows': candidate_rows[indices].cpu().tolist(),
             'orthogonal_gains': gains, 'initial_target_sse': float(target.square().sum()),
             'regularized_fit_sse': float((target-fitted).square().sum()),
             'ridge_absolute': float(regularization), 'seconds': time.perf_counter()-started,
             'note': 'OLS is a supervised gradient-surrogate initializer; subsequent SFT optimizes token CE.'}
    return adapter, audit


class RBFAdaptedModel(nn.Module):
    def __init__(self, base, adapter):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.context_length = base.context_length
        self.config = base.config
        for p in base.parameters():
            p.requires_grad_(False)

    def frozen_batch(self, ids):
        with torch.no_grad():
            hidden = self.base.hidden(ids)
            logits = self.base.lm_head(hidden)
        return hidden.detach(), logits.detach()

    def forward(self, ids):
        h, logits = self.frozen_batch(ids)
        return logits.float() + self.adapter(h)


class SteepestDescent:
    """Negative Euclidean gradient + Armijo backtracking, no momentum or Adam states."""
    def __init__(self, parameters, initial_step=8., max_step=512., backtracking=.5,
                 armijo=1e-4, max_evaluations=16):
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters or not 0 < initial_step <= max_step or not 0 < backtracking < 1:
            raise ValueError('Invalid steepest-descent settings')
        if not 0 < armijo < 1 or not 1 <= max_evaluations <= 32:
            raise ValueError('Invalid line search')
        self.next_step, self.max_step = initial_step, max_step
        self.backtracking, self.armijo, self.max_evaluations = backtracking, armijo, max_evaluations

    def step(self, closure):
        for p in self.parameters:
            p.grad = None
        loss = closure()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite initial batch loss')
        loss.backward()
        gradients = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in self.parameters]
        norm2 = sum(g.float().square().sum() for g in gradients)
        if not torch.isfinite(norm2):
            raise RuntimeError('Non-finite steepest direction')
        before = float(loss.detach())
        if norm2 <= 1e-24:
            return {'before': before, 'after': before, 'accepted': True, 'step': 0., 'function_evaluations': 1}
        original = [p.detach().clone() for p in self.parameters]
        step, accepted, after, evaluations = self.next_step, False, before, 1
        try:
            with torch.no_grad():
                for _ in range(self.max_evaluations):
                    for p, start, g in zip(self.parameters, original, gradients):
                        p.copy_(start - step * g)
                    trial = closure()
                    evaluations += 1
                    after = float(trial)
                    if math.isfinite(after) and after <= before - self.armijo * step * float(norm2):
                        accepted = True
                        break
                    step *= self.backtracking
        finally:
            if not accepted:
                with torch.no_grad():
                    for p, start in zip(self.parameters, original):
                        p.copy_(start)
                after = before
        self.next_step = min(step * 1.5, self.max_step) if accepted else max(step, 1e-12)
        return {'before': before, 'after': after, 'accepted': accepted,
                'step': step if accepted else 0., 'gradient_norm': float(norm2.sqrt()),
                'function_evaluations': evaluations}
