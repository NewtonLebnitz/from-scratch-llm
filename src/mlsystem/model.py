"""New configurable causal GQA model with explicit cross-layer Hopfield score state.

Paper: Masumura & Taki (NeurIPS 2025), Eq.(10) and Fig.1a.
The inner visible-state skip is inside the attention module; the Transformer outer
residual remains present. We adapt this module to Pre-RMSNorm/RoPE/GQA and document
that architecture choice, rather than claiming to reproduce the paper's GPT-2.
"""
import math
import torch
from torch import nn
from .components import Linear, Embedding, RMSNorm, SwiGLU, CausalSelfAttention, softmax
from .config import ModelConfig


class Attention(CausalSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg.d_model, cfg.num_heads, cfg.context_length,
                         cfg.rope_theta, cfg.num_kv_heads)
        self.mode = cfg.attention
        self.alpha = cfg.hopfield_alpha
        self.alpha_prime = cfg.hopfield_alpha_prime

    def forward(self, x, score_state=None):
        batch, length, width = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(length, device=x.device)
        q, k = self.rope(q, positions), self.rope(k, positions)
        groups = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.head_dim)
        mask = self.causal_mask[:length, :length]
        # Keep state finite. Future entries never influence any softmax or later row.
        scores = scores.masked_fill(~mask, 0)
        if self.mode in ['hopfield', 'hopfield_official_v8']:
            if score_state is None:
                score_state = scores if self.mode == 'hopfield_official_v8' else (1 - self.alpha_prime) * scores
            else:
                if score_state.shape != scores.shape:
                    raise ValueError('Cross-layer attention state has incompatible shape')
                score_state = self.alpha_prime * score_state + (1 - self.alpha_prime) * scores
            logits = score_state
        else:
            logits = scores
            score_state = None
        probabilities = softmax(logits.masked_fill(~mask, float('-inf')), dim=-1)
        attended = (probabilities.to(v.dtype) @ v).transpose(1, 2).reshape(batch, length, width)
        if self.mode == 'hopfield_official_v8':
            attended = (1 - self.alpha) * attended + self.alpha * x
        output = self.output_proj(attended)
        if self.mode == 'hopfield':
            # Historical v1 behavior retained exclusively for historical weights.
            output = self.alpha * x + (1 - self.alpha) * output
        return output, score_state


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, state):
        update, state = self.attn(self.ln1(x), state)
        x = x + update
        return x + self.ffn(self.ln2(x)), state


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        self.context_length = config.context_length
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_final = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)

    def hidden(self, ids, collect_layers=False):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.context_length:
            raise ValueError('Expected nonempty [batch, tokens] within configured context')
        x = self.token_embeddings(ids)
        trace = [x] if collect_layers else None
        state = None  # New sequence/request always starts a new cross-layer trajectory.
        for layer in self.layers:
            x, state = layer(x, state)
            if trace is not None:
                trace.append(x)
        h = self.ln_final(x)
        return (h, trace) if collect_layers else h

    def forward(self, ids):
        return self.lm_head(self.hidden(ids))


@torch.no_grad()
def representation_metrics(x):
    """One unpadded sequence [T,D]; report raw and centered ranks explicitly."""
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError('Need at least two unpadded token vectors')
    x = x.float()

    def ranks(matrix):
        s = torch.linalg.svdvals(matrix)
        mass = s.sum()
        if mass <= 1e-12:
            return {'effective_rank': 0.0, 'stable_rank': 0.0}
        p = s / mass
        effective = torch.exp(-(p * p.clamp_min(1e-30).log()).sum())
        stable = s.square().sum() / s[0].square().clamp_min(1e-30)
        return {'effective_rank': float(effective), 'stable_rank': float(stable)}

    normed = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cosine_sum = normed.sum(0).square().sum() - normed.square().sum()
    n = x.shape[0]
    centered = x - x.mean(0, keepdim=True)
    return {'tokens': n, 'raw': ranks(x), 'centered': ranks(centered),
            'mean_offdiagonal_cosine': float(cosine_sum / (n*(n-1))),
            'centered_energy_fraction': float(centered.square().sum() / x.square().sum().clamp_min(1e-30))}
