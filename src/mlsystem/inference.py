"""Bounded real autoregressive decoding; retrieval never substitutes the answer."""
from dataclasses import dataclass
import math
import time
import torch
from .data import SPECIAL_TOKENS, clean_document, conversation_prefix
from .inference_metrics import generation_metrics


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = .3
    top_p: float = .9
    repetition_penalty: float = 1.05
    seed: int = 7

    def validate(self, context):
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= min(512, context-8):
            raise ValueError('Generation length exceeds context or 512-token maximum')
        if any(not math.isfinite(x) for x in [self.temperature, self.top_p, self.repetition_penalty]):
            raise ValueError('Generation settings must be finite')
        if not 0 <= self.temperature <= 2 or not .05 <= self.top_p <= 1 or not 1 <= self.repetition_penalty <= 1.5:
            raise ValueError('Invalid decoding settings')
        if type(self.seed) is not int or not 0 <= self.seed <= 2147483647: raise ValueError('Invalid seed')
        return self


def pack_sources(question, records, tokenizer, context_length, max_new_tokens):
    """Report exactly what evidence fits. Never silently evict the question."""
    question = clean_document(question)
    budget = context_length-max_new_tokens
    if len(tokenizer.encode(conversation_prefix(question))) > budget:
        raise ValueError('Question and output budget exceed the model context')
    parts, included, omitted = [], [], []
    for record in records:
        text = clean_document(record['text'])
        identifier = clean_document(record['id'])
        title = clean_document(record.get('title', record.get('source_id', identifier)))
        prefix = f'[{identifier}] {title}\n'
        candidate = '\n\n'.join(parts+[prefix+text])
        clipped = False
        if len(tokenizer.encode(conversation_prefix(question, candidate))) > budget:
            # Find a prefix whose encoded length fits; preserve metadata in UI.
            lo, hi = 0, len(text)
            while lo < hi:
                middle = (lo+hi+1)//2
                trial = '\n\n'.join(parts+[prefix+text[:middle]])
                if len(tokenizer.encode(conversation_prefix(question, trial))) <= budget: lo = middle
                else: hi = middle-1
            if lo < 80:
                omitted.append(identifier); continue
            text = text[:lo]; clipped = True
        parts.append(prefix+text)
        included.append(dict(record, text=text, clipped_for_context=clipped))
    return '\n\n'.join(parts), included, omitted


@torch.no_grad()
def generate(model, tokenizer, prompt, settings, *, stop_requested=lambda: False):
    settings.validate(model.context_length)
    ids = tokenizer.encode(prompt)
    if not ids: raise ValueError('Prompt must encode to at least one token')
    if len(ids)+settings.max_new_tokens > model.context_length:
        raise ValueError('Prompt and output budget exceed trained model context configuration')
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(settings.seed)
    stop_ids = {tokenizer.special_ids[m] for m in ['<|endoftext|>', '<|turn_end|>']}
    banned = [tokenizer.special_ids[m] for m in SPECIAL_TOKENS if tokenizer.special_ids[m] not in stop_ids]
    generated, reason = [], 'length_limit'
    token_times = []
    started = time.perf_counter()
    model.eval()
    from .decoding import make_decoder
    decoder = make_decoder(model)
    decoder_setup_seconds = time.perf_counter()-started
    for _ in range(settings.max_new_tokens):
        if stop_requested(): reason = 'cancelled'; break
        x = torch.tensor([ids+generated if decoder is None else (ids if not generated else [generated[-1]])], dtype=torch.long, device=device)
        scores = (model(x)[0,-1] if decoder is None else decoder.last_logits(x)[0]).float()
        if not torch.isfinite(scores).all(): raise RuntimeError('Non-finite model output, no answer substituted')
        scores[banned] = -torch.inf
        if generated and settings.repetition_penalty > 1:
            recent = list(set(generated[-64:]))
            values = scores[recent]
            scores[recent] = torch.where(values >= 0, values/settings.repetition_penalty, values*settings.repetition_penalty)
        if settings.temperature == 0:
            token = int(scores.argmax())
        else:
            probabilities = torch.softmax(scores/settings.temperature, dim=-1)
            sorted_p, order = probabilities.sort(descending=True)
            remove = sorted_p.cumsum(0)-sorted_p > settings.top_p
            sorted_p[remove] = 0
            token = int(order[torch.multinomial(sorted_p, 1, generator=rng)])
        if token in stop_ids: reason = 'model_terminator'; break
        generated.append(token)
        token_times.append(time.perf_counter()-started)
    seconds = time.perf_counter()-started
    return {'text': tokenizer.decode(generated), 'token_ids': generated, 'prompt_tokens': len(ids),
            'generated_tokens': len(generated), 'seconds': seconds,
            **generation_metrics(ids, token_times, seconds, decoder_setup_seconds),
            'tokens_per_second': len(generated)/max(seconds, 1e-9), 'stop_reason': reason,
            'generation': settings.__dict__, 'answer_origin': 'autoregressive language model tokens',
            'decoding': 'request-local KV cache' if decoder is not None else 'full-prefix fallback'}
