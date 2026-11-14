"""Tensor-only decoder Transformer: no prebuilt neural-network layers."""
import math
import torch
from torch import nn


def truncated_normal(shape, std, device=None, dtype=None):
    # Inverse-CDF sampling from N(0,std^2), truncated to +/-3 std.
    x = torch.empty(shape, device=device, dtype=dtype).uniform_(0.001349898, 0.998650102)
    return x.mul_(2).sub_(1).erfinv_().mul_(std * math.sqrt(2))


class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(truncated_normal((d_out, d_in), math.sqrt(2 / (d_in + d_out)), device, dtype))

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(truncated_normal((vocab_size, d_model), 1.0, device, dtype))

    def forward(self, ids):
        return self.weight[ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x, dim=-1):
    y = x - x.max(dim=dim, keepdim=True).values
    e = y.exp()
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    # All-masked rows map to zero instead of NaN. Normal rows use stable softmax.
    invalid = torch.isneginf(scores).all(-1, keepdim=True)
    probs = softmax(scores.float().masked_fill(invalid, 0), -1).masked_fill(invalid, 0)
    return probs.to(v.dtype) @ v


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k, theta, max_seq_len, device=None):
        super().__init__()
        if d_k % 2:
            raise ValueError('RoPE head dimension must be even')
        inv = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = torch.arange(max_seq_len, device=device)[:, None] * inv[None, :]
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        c, s = self.cos[token_positions], self.sin[token_positions]
        # Supports positions [T] and [B,T] with inputs [B,H,T,D].
        while c.ndim < x.ndim:
            c, s = c.unsqueeze(-3), s.unsqueeze(-3)
        even, odd = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=256, theta=10000.0,
                 num_kv_heads=None, use_rope=True, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if d_model % num_heads or num_heads % self.num_kv_heads:
            raise ValueError('d_model / num_heads and num_heads / num_kv_heads must be integral')
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, self.num_kv_heads * self.head_dim, device, dtype)
        self.v_proj = Linear(d_model, self.num_kv_heads * self.head_dim, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = RotaryPositionalEmbedding(self.head_dim, theta, max_seq_len, device) if use_rope else None
        self.register_buffer('causal_mask', torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, token_positions=None):
        t = x.shape[-2]
        q = self.q_proj(x).reshape(*x.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)
        k = self.k_proj(x).reshape(*x.shape[:-1], self.num_kv_heads, self.head_dim).transpose(-3, -2)
        v = self.v_proj(x).reshape(*x.shape[:-1], self.num_kv_heads, self.head_dim).transpose(-3, -2)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(t, device=x.device)
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        groups = self.num_heads // self.num_kv_heads
        if groups != 1:
            k, v = k.repeat_interleave(groups, -3), v.repeat_interleave(groups, -3)
        out = scaled_dot_product_attention(q, k, v, self.causal_mask[:t, :t])
        return self.output_proj(out.transpose(-3, -2).reshape(*x.shape[:-1], -1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta,
                 num_kv_heads=None, device=None, dtype=None, norm_eps=1e-5):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, theta, num_kv_heads, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads,
                 d_ff, rope_theta, num_kv_heads=None, device=None, dtype=None, norm_eps=1e-5):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length,
                                    rope_theta, num_kv_heads, device, dtype, norm_eps=norm_eps) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, ids):
        if ids.shape[-1] > self.context_length:
            raise ValueError('Input exceeds context_length')
        x = self.token_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
