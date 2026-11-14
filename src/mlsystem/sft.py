"""Three real full-answer SFT routes from an immutable pretrained checkpoint.

The OLS route is a Gaussian logit residual, deliberately not described as the
same GenLoRA architecture with another optimizer. No retired classifier is used.
"""
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from .bpe import Tokenizer
from .config import ModelConfig
from .data import encode_sft, iter_jsonl
from .genlora import attach, adapter_state
from .model import LanguageModel
from .ols_rbf import GaussianLogitAdapter, RBFAdaptedModel, SteepestDescent, initialize_supervised
from .optim import AdamW, cross_entropy, gradient_clipping
from .training import atomic_json, file_sha


@dataclass(frozen=True)
class SFTSettings:
    route: str = 'genlora'
    steps: int = 500
    microbatch: int = 2
    accumulation: int = 4
    sequence_length: int = 1024
    learning_rate: float = .0002
    weight_decay: float = .01
    grad_clip: float = 1.
    rank: int = 64
    groups: int = 4
    gaussian_centers: int = 15
    scale_alpha: float = 32.
    calibration_positions: int = 1024
    pca_dimensions: int = 32
    candidate_centers: int = 128
    selected_centers: int = 64
    ridge: float = .001
    initial_step: float = 8.
    max_step: float = 512.
    max_evaluations: int = 12
    eval_every: int = 100
    eval_batches: int = 8
    save_every: int = 100
    seed: int = 7

    def validate(self, config):
        if self.route not in ['genlora', 'ols', 'full']:
            raise ValueError('Choose genlora, ols or full')
        bounds = {'steps': (1, 100000), 'microbatch': (1, 16), 'accumulation': (1, 32),
                  'sequence_length': (32, config.context_length), 'rank': (1, 256),
                  'groups': (1, 16), 'gaussian_centers': (3, 63),
                  'calibration_positions': (32, 4096), 'pca_dimensions': (1, 128),
                  'candidate_centers': (2, 256), 'selected_centers': (1, 128),
                  'max_evaluations': (1, 24), 'eval_every': (1, 10000),
                  'eval_batches': (1, 128), 'save_every': (1, 10000), 'seed': (0, 2147483647)}
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{key} must be an integer in [{lo}, {hi}]')
        for key in ['learning_rate', 'weight_decay', 'grad_clip', 'scale_alpha', 'ridge', 'initial_step', 'max_step']:
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f'{key} must be finite')
        if not 0 < self.learning_rate <= .01 or not 0 <= self.weight_decay <= 1 or not 0 < self.grad_clip <= 100:
            raise ValueError('Invalid learning rate, decay or clip')
        if not 0 < self.scale_alpha <= 512 or not 0 < self.ridge <= 1:
            raise ValueError('Invalid adapter scale or ridge')
        if self.route == 'genlora':
            widths = [config.d_model, config.d_ff, config.vocab_size,
                      config.num_kv_heads * (config.d_model // config.num_heads)]
            if any(width % self.groups or width // self.groups < 2 for width in widths):
                raise ValueError('GenLoRA groups must divide every adapted linear dimension, with >=2 coordinates per group')
        if not 0 < self.initial_step <= self.max_step <= 4096:
            raise ValueError('Invalid steepest descent step bounds')
        if not self.selected_centers <= self.candidate_centers <= self.calibration_positions:
            raise ValueError('Need selected <= candidate <= calibration positions')
        if self.pca_dimensions > min(config.d_model, self.calibration_positions - 1):
            raise ValueError('PCA dimension exceeds calibration capacity')
        return self


def within(root, relative, subtree):
    path = (root / relative).resolve()
    if not path.is_relative_to(root / subtree):
        raise ValueError('Path outside ' + subtree)
    return path


def load_base(root, checkpoint):
    path = within(root, checkpoint, 'artifacts/models')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'mlsystem-pretrain-v1' or state.get('step', 0) < 1:
        raise ValueError('Every SFT must start directly from a new pretrained model, never a previous SFT adapter')
    config = ModelConfig(**state['model_config']).validate()
    model = LanguageModel(config)
    model.load_state_dict(state['model'], strict=True)
    return model, state, file_sha(path)


class AnswerData:
    def __init__(self, root, manifest_path, tokenizer_sha256, context_length, seed):
        self.root = root
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if manifest.get('status') != 'ready' or manifest.get('format') != 'mlsystem-sft-v1':
            raise ValueError('Need reviewed and ready full-answer SFT data')
        self.identity = file_sha(manifest_path)
        tp = within(root, 'artifacts/tokenizers/' + manifest['tokenizer'] + '/tokenizer.json', 'artifacts/tokenizers')
        if file_sha(tp) != tokenizer_sha256 or manifest['tokenizer_sha256'] != tokenizer_sha256:
            raise ValueError('SFT dataset, BPE and base checkpoint must use the same tokenizer')
        tokenizer = Tokenizer.load(tp)
        self.pad = tokenizer.special_ids['<|pad|>']
        self.rng = np.random.default_rng(seed)
        self.examples = {}
        families = {}
        for row in manifest['files']:
            split = row['split']
            if split not in ['train', 'dev', 'test'] or split in self.examples:
                raise ValueError('Exactly one file per registered split')
            path = within(root, row['path'], 'datasets/sft')
            if path.stat().st_size > 256 * 1024**2 or file_sha(path) != row['sha256']:
                raise ValueError('SFT file too large or changed')
            examples = []
            for item in iter_jsonl(path):
                if not item.get('source_ids') or not item.get('family_id'):
                    raise ValueError('Every example needs source IDs and a split family')
                family = item['family_id']
                if family in families and families[family] != split:
                    raise ValueError('Source family leaks across splits')
                families[family] = split
                # Keep variable lengths; batch pads only to its longest actual example.
                x, y = encode_sft(item, tokenizer, context_length)
                end = max(i for i, value in enumerate(y) if value != -100) + 1
                examples.append((x[:end], y[:end]))
            if not examples or len(examples) != row['examples']:
                raise ValueError('SFT record count mismatch or empty split')
            self.examples[split] = examples
        if not all(split in self.examples for split in ['train', 'dev']):
            raise ValueError('Separate TRAIN and DEV are required')

    def batch(self, split, count, device, rng=None):
        rng = self.rng if rng is None else rng
        examples = self.examples[split]
        rows = [examples[int(rng.integers(len(examples)))] for _ in range(count)]
        length = max(len(x) for x, _ in rows)
        x = torch.tensor([x + [self.pad]*(length-len(x)) for x, _ in rows], device=device)
        y = torch.tensor([y + [-100]*(length-len(y)) for _, y in rows], device=device)
        return x, y


@torch.no_grad()
def collect_calibration(base, data, settings, device, bf16):
    """Use TRAIN answer positions only, independent of the subsequent training order."""
    rng = np.random.default_rng(settings.seed + 201)
    hidden, logits, labels = [], [], []
    remaining = settings.calibration_positions
    base.eval()
    while remaining:
        x, y = data.batch('train', settings.microbatch, device, rng)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            h = base.hidden(x)
            scores = base.lm_head(h)
        mask = y != -100
        indices = torch.randperm(int(mask.sum()), generator=torch.Generator().manual_seed(int(rng.integers(2**31))))[:remaining].to(device)
        hidden.append(h[mask][indices].float())
        logits.append(scores[mask][indices].float())
        labels.append(y[mask][indices])
        remaining -= len(indices)
    return torch.cat(hidden), torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def dev_loss(model, data, settings, device, bf16):
    rng = np.random.default_rng(settings.seed + 999)
    was_training = model.training
    model.eval()
    total, count = 0., 0
    try:
        for _ in range(settings.eval_batches):
            x, y = data.batch('dev', settings.microbatch, device, rng)
            n = int((y != -100).sum())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                value = cross_entropy(model(x), y)
            if not torch.isfinite(value):
                raise RuntimeError('Non-finite DEV loss')
            total += float(value) * n
            count += n
        return total / count
    finally:
        model.train(was_training)


def model_payload(model, route):
    if route == 'full':
        return {'weights': model.state_dict()}
    if route == 'genlora':
        return {'weights': adapter_state(model)}
    return {'weights': model.adapter.state_dict()}


def restore_payload(model, route, payload):
    if route == 'ols':
        model.adapter.load_state_dict(payload['weights'], strict=True)
    elif route == 'full':
        model.load_state_dict(payload['weights'], strict=True)
    else:
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(payload['weights']) != expected:
            raise ValueError('GenLoRA checkpoint parameter names disagree')
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(payload['weights'][name])


def make_ols_from_state(state):
    if 'widths' in state:
        from .ols_fixed_v2 import FixedGaussianAdapter
        adapter = FixedGaussianAdapter(state['mean'], state['projection'], state['feature_scale'],
                                       state['centers'], state['widths'], state['readout'].shape[1])
        adapter.load_state_dict(state, strict=True)
        return adapter
    return GaussianLogitAdapter(state['mean'], state['projection'], state['feature_scale'],
                               state['centers'], torch.nn.functional.softplus(state['raw_widths']), state['readout'])


def save_sft(path, model, optimizer, step, data, settings, base_config, identity, best_dev, audit):
    state = {'format': 'mlsystem-sft-v1', 'settings': asdict(settings), 'model_config': base_config.dictionary(),
             'identity': identity, 'step': step, 'best_dev': best_dev, 'initialization': audit,
             'payload': model_payload(model, settings.route), 'sampler_state': data.rng.bit_generator.state,
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
             'cuda_rng': torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
             'optimizer': optimizer.state_dict() if settings.route != 'ols' else {'next_step': optimizer.next_step}}
    part = Path(path).with_suffix('.writing')
    with part.open('wb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(part, path)


def run_sft(root, output, base_checkpoint, manifest_path, settings, *, device, bf16=False,
            resume=False, stop_requested=lambda: False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root/'artifacts/models') or output == root/'artifacts/models':
        raise ValueError('SFT output must be a dedicated model directory')
    random.seed(settings.seed); torch.manual_seed(settings.seed)
    base, parent, parent_sha = load_base(root, base_checkpoint)
    settings.validate(base.config)
    config = base.config
    data = AnswerData(root, manifest_path, parent['identity']['tokenizer_sha256'], settings.sequence_length, settings.seed)
    import hashlib
    code = hashlib.sha256()
    for file in ['sft.py', 'genlora.py', 'ols_rbf.py', 'data.py', 'model.py', 'optim.py', 'components.py', 'config.py']:
        code.update((Path(__file__).parent/file).read_bytes())
    identity = {'parent_path': Path(base_checkpoint).as_posix(), 'parent_sha256': parent_sha,
                'tokenizer_sha256': parent['identity']['tokenizer_sha256'],
                'data_manifest_sha256': data.identity, 'training_code_sha256': code.hexdigest()}
    del parent
    saved = None
    if resume:
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved['settings'] != asdict(settings):
            raise ValueError('Resume base, data, settings or code changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'initializing', 'stage': 'sft', 'route': settings.route, 'pid': os.getpid(),
              'identity': identity, 'settings': asdict(settings), 'model_config': config.dictionary(), 'step': 0}
    atomic_json(output/'status.json', status)
    started = time.perf_counter()
    step = 0
    try:
        base = base.to(device)
        if settings.route == 'genlora':
            targets = attach(base, 'rbf', settings.rank, settings.groups, settings.gaussian_centers, settings.scale_alpha)
            model, audit = base, {'method': 'GenLoRA-style Gaussian plus SiLU', 'targets': targets, 'seconds': 0.}
        elif settings.route == 'ols':
            if saved:
                adapter = make_ols_from_state(saved['payload']['weights']).to(device)
                audit = saved['initialization']
            else:
                arrays = collect_calibration(base, data, settings, device, bf16)
                adapter, audit = initialize_supervised(*arrays, dimensions=settings.pca_dimensions,
                    candidates=settings.candidate_centers, selected_centers=settings.selected_centers,
                    ridge=settings.ridge, seed=settings.seed)
                del arrays
            model = RBFAdaptedModel(base, adapter)
        else:
            model, audit = base, {'method': 'all parameters + AdamW', 'seconds': 0.}
        parameters = [p for p in model.parameters() if p.requires_grad]
        if settings.route == 'ols':
            optimizer = SteepestDescent(parameters, settings.initial_step, settings.max_step,
                                       max_evaluations=settings.max_evaluations)
        else:
            optimizer = AdamW(parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay)
        best_dev = math.inf
        if saved:
            restore_payload(model, settings.route, saved['payload'])
            if settings.route == 'ols':
                optimizer.next_step = saved['optimizer']['next_step']
            else:
                optimizer.load_state_dict(saved['optimizer'])
            step, best_dev = saved['step'], saved['best_dev']
            data.rng.bit_generator.state = saved['sampler_state']
            torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
            if device.type == 'cuda': torch.cuda.set_rng_state(saved['cuda_rng'])
            del saved
        initialization_seconds = time.perf_counter() - started
        status.update(status='running', parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in parameters), initialization=audit,
                      initialization_seconds=initialization_seconds, step=step)
        atomic_json(output/'status.json', status)
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for current in range(step+1, settings.steps+1):
            if stop_requested(): break
            tick = time.perf_counter()
            batches = [data.batch('train', settings.microbatch, device) for _ in range(settings.accumulation)]
            answer_tokens = sum(int((y != -100).sum()) for _, y in batches)
            input_tokens = sum(int((x != data.pad).sum()) for x, _ in batches)
            if settings.route == 'ols':
                cached = []
                for x, y in batches:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                        h, logits = model.frozen_batch(x)
                    mask = y != -100
                    # Only answer positions need score corrections or trial evaluations.
                    cached.append((h[mask], logits[mask], y[mask]))
                def closure():
                    value = torch.zeros((), device=device)
                    for h, logits, y in cached:
                        value = value + cross_entropy(logits.float()+model.adapter(h), y)*y.numel()/answer_tokens
                    return value
                update = optimizer.step(closure)
                # Match AdamW routes: the main fitting curve is evaluated before
                # this update. Armijo's accepted post-update CE remains in update.
                ce = update['before']
                del cached
            else:
                optimizer.zero_grad(set_to_none=True)
                ce = 0.
                for x, y in batches:
                    weight = int((y != -100).sum())/answer_tokens
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                        loss = cross_entropy(model(x), y)*weight
                    if not torch.isfinite(loss): raise RuntimeError('Non-finite answer CE')
                    loss.backward(); ce += float(loss.detach())
                norm = gradient_clipping(parameters, settings.grad_clip)
                if not torch.isfinite(norm): raise RuntimeError('Non-finite SFT gradient')
                optimizer.step()
                update = {'gradient_norm': float(norm), 'function_evaluations': len(batches),
                          'step': settings.learning_rate, 'accepted': True}
            if device.type == 'cuda': torch.cuda.synchronize(device)
            step = current
            seconds = time.perf_counter()-tick
            row = {'step': step, 'answer_ce': ce, 'input_tokens': input_tokens, 'answer_tokens': answer_tokens,
                   'update_seconds': seconds, 'input_tokens_per_second': input_tokens/seconds,
                   'answer_tokens_per_second': answer_tokens/seconds, 'update': update,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None}
            if step % settings.eval_every == 0 or step == settings.steps:
                row['dev_ce'] = dev_loss(model, data, settings, device, bf16)
                if row['dev_ce'] < best_dev:
                    best_dev = row['dev_ce']
                    save_sft(output/'best.pt', model, optimizer, step, data, settings, config, identity, best_dev, audit)
            with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            status.update(step=step, last=row, best_dev=best_dev if math.isfinite(best_dev) else None)
            atomic_json(output/'status.json', status)
            if step % settings.save_every == 0 or step == settings.steps:
                save_sft(output/'last.pt', model, optimizer, step, data, settings, config, identity, best_dev, audit)
        save_sft(output/'last.pt', model, optimizer, step, data, settings, config, identity, best_dev, audit)
        status.update(status='completed' if step == settings.steps else 'cancelled',
                      session_seconds=time.perf_counter()-started)
        if step == settings.steps:
            status.update(checkpoint='best.pt', checkpoint_sha256=file_sha(output/'best.pt'))
    except BaseException as exc:
        status.update(status='failed', error=type(exc).__name__+': '+str(exc), step=step)
        raise
    finally:
        atomic_json(output/'status.json', status)
    return status


def load_sft(root, checkpoint, device):
    """Resolve exact parent hash; adapter files never masquerade as full models."""
    root = Path(root).resolve()
    path = within(root, checkpoint, 'artifacts/models')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'mlsystem-sft-v1': raise ValueError('Not a new SFT artifact')
    settings = SFTSettings(**state['settings'])
    base, _, digest = load_base(root, state['identity']['parent_path'])
    if digest != state['identity']['parent_sha256']: raise ValueError('SFT parent weight changed')
    if settings.route == 'genlora':
        attach(base, 'rbf', settings.rank, settings.groups, settings.gaussian_centers, settings.scale_alpha)
        model = base
    elif settings.route == 'ols':
        geometry = state.get('ols_geometry')
        if geometry == 'lexical_question_gaussian_sd_v5':
            from .lexical_question_rbf import make_lexical_adapter
            adapter = make_lexical_adapter(state['payload']['weights'], hidden_width=base.config.d_model,
                                            vocabulary_size=base.config.vocab_size)
        elif geometry == 'question_conditioned_gaussian_sd_v4':
            from .question_rbf import make_question_adapter
            adapter = make_question_adapter(state['payload']['weights'], hidden_width=base.config.d_model,
                                            vocabulary_size=base.config.vocab_size)
        elif geometry in [None, 'fixed_gaussian_readout_v4', 'joint_gaussian_sd_v4']:
            adapter = make_ols_from_state(state['payload']['weights'])
        else:
            raise ValueError('Unsupported OLS geometry; refuse a guessed loader')
        model = RBFAdaptedModel(base, adapter)
    else:
        model = base
    restore_payload(model, settings.route, state['payload'])
    return model.to(device).eval(), state
