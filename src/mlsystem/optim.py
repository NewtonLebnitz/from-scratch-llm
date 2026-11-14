"""Handwritten loss, AdamW, batching, scheduling, clipping and checkpoints."""
import math
import numpy as np
import torch


def cross_entropy(inputs, targets, ignore_index=-100):
    z = inputs.float().reshape(-1, inputs.shape[-1])
    labels = targets.reshape(-1)
    valid = labels != ignore_index
    if not valid.any():
        raise ValueError('Batch has no supervised answer tokens')
    z, labels = z[valid], labels[valid]
    if torch.any(labels < 0) or torch.any(labels >= z.shape[1]):
        raise ValueError('Target outside vocabulary')
    z = z - z.max(-1, keepdim=True).values
    selected = z.gather(-1, labels.reshape(-1, 1)).squeeze(-1)
    return (z.exp().sum(-1).log() - selected).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW hyperparameters')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError('Sparse gradients unsupported')
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state['t'] += 1
                t, m, v = state['t'], state['m'], state['v']
                m.mul_(b1).add_(g, alpha=1-b1)
                v.mul_(b2).addcmul_(g, g, value=1-b2)
                # Decay uses the pre-update parameter; epsilon is outside sqrt(v_hat).
                p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                denominator = (v / (1-b2**t)).sqrt().add_(group['eps'])
                p.addcdiv_(m, denominator, value=-group['lr'] / (1-b1**t))
        return loss


def gradient_clipping(parameters, max_l2_norm):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.)
    norm = torch.stack([p.grad.float().square().sum() for p in params]).sum().sqrt()
    scale = (max_l2_norm / (norm + 1e-6)).clamp(max=1)
    for p in params:
        p.grad.mul_(scale)
    return norm


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * it / max(1, warmup_iters)
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it-warmup_iters) / (cosine_cycle_iters-warmup_iters)
    return min_learning_rate + 0.5 * (max_learning_rate-min_learning_rate) * (1+math.cos(math.pi*progress))


def get_batch(dataset, batch_size, context_length, device):
    if len(dataset) <= context_length:
        raise ValueError('Dataset shorter than context length + 1')
    starts = np.random.randint(0, len(dataset)-context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length+1)[None, :]
    data = torch.from_numpy(np.asarray(dataset[indices], dtype=np.int64)).to(device)
    return data[:, :-1], data[:, 1:]


def save_checkpoint(model, optimizer, iteration, out):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration), out)


def load_checkpoint(src, model, optimizer):
    state = torch.load(src, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['iteration']
