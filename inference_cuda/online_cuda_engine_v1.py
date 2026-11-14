"""Isolated graph scheduler; the frozen model math and compiler options are reused."""
from collections import Counter
from statistics import median
from time import perf_counter

import torch

from common_targeted_v3_engine import bucket_for, bucket_sizes
from online_cuda_buckets_v1 import Graph, OnlineBuckets


class ScheduledEngine:
    def __init__(self, engine, strategy, compile_preset=None, compact_capture=False, eager_capture=False, eager_fallback=False):
        self.engine, self.strategy = engine, strategy
        self.capacity = min(512, engine.context)
        self.policy = OnlineBuckets(self.capacity, max_probes=4)
        self.graphs, self.events = {}, []
        self.lengths, self.fallback, self.paths = [], [], Counter()
        self.compile_preset = compile_preset
        self.compact_capture = compact_capture
        self.eager_capture = eager_capture
        self.eager_fallback = eager_fallback
        self.capture_stream = None
        self.engine.last_logits = self.last_logits

    def setup(self):
        e = self.engine
        mode = self.compile_preset or e.spec.get('compile_mode', 'default')
        if self.compact_capture:
            self.capture_stream = torch.cuda.Stream()
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.accumulated_recompile_limit = 1024
        e.static_ids = torch.ones((1, 1), device='cuda', dtype=torch.long)
        e.static_pos = torch.ones(1, device='cuda', dtype=torch.long)
        from torch._inductor import list_mode_options
        options = dict(list_mode_options(mode))
        # This engine's own scheduler controls all graph admission and replay.
        options['triton.cudagraphs'] = False
        compiler = {'options': options}
        if not (self.eager_capture and self.eager_fallback):
            e.step_call = torch.compile(e.step_tensor, backend='inductor', fullgraph=True,
                                        dynamic=False, **compiler)
            if e.spec.get('target_prefill'):
                e.prefill_call = torch.compile(e.prefill_tensor, backend='inductor',
                                               fullgraph=True, dynamic=True, **compiler)
        if e.spec.get('joint_selection') and not e.spec.get('integrated_selection'):
            e.select_call = torch.compile(e.choose_tensor, fullgraph=True,
                                          dynamic=False, **compiler)
    def input_for_bucket(self, dest, positions, ids):
        n, size = ids.shape[1], dest.shape[1]
        if n == size:
            dest.copy_(ids)
            positions.copy_(self.engine.arange[:size])
        else:
            # The frozen forward selects its last *physical* row. Move the last
            # real token there, keeping its real RoPE coordinate. KV index_copy
            # writes to semantic positions; causal masks are based on coordinates.
            dest[:, :n-1].copy_(ids[:, :-1])
            dest[:, n-1:-1].fill_(1)
            dest[:, -1:].copy_(ids[:, -1:])
            positions[n-1:-1].copy_(self.engine.arange[n:size])
            positions[:n-1].copy_(self.engine.arange[:n-1])
            positions[-1:].fill_(n-1)

    def capture(self, key, measure):
        phase, size = key
        e = self.engine
        before = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        start = perf_counter()
        e.position, e.decode_seeded = 0, False
        if phase == 'prefill':
            assert e.spec.get('target_prefill')
            e.decode_bucket = e.context
            ids = torch.ones((1, size), device='cuda', dtype=torch.long)
            pos = e.arange[:size].clone()
            call = e.prefill_tensor if self.eager_capture else e.prefill_call
        else:
            e.decode_bucket = size
            ids, pos = e.static_ids, e.static_pos
            call = e.step_tensor if self.eager_capture else e.step_call
            ids.fill_(1)
            pos.fill_(min(1, size-1))
        assert call is not None
        if measure and phase == 'decode':
            # Build valid synthetic KV for the full range of the test replay.
            # This work belongs to the trial clock.
            k = size - 1
            e.decode_bucket = e.context
            e.forward(torch.ones((1, k), device='cuda', dtype=torch.long), e.arange[:k])
            e.decode_bucket = size
            pos.fill_(k)
        setup_started = perf_counter()
        call(ids, pos)
        torch.cuda.synchronize()
        first_call_seconds = perf_counter() - setup_started
        warm_started = perf_counter()
        stream = self.capture_stream or torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(1 if self.compact_capture else 3):
                call(ids, pos)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        warm_seconds = perf_counter() - warm_started
        record_started = perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call(ids, pos)
            if phase == 'decode' and e.host_pair is not None:
                e.host_pair.copy_(output, non_blocking=True)
        if phase == 'decode':
            pos.fill_(min(1, size-1))
        graph.replay()
        torch.cuda.synchronize()
        record_seconds = perf_counter() - record_started
        samples = []
        if measure:
            for _ in range(5):
                begin = perf_counter()
                if phase == 'prefill':
                    self.input_for_bucket(ids, pos, e.arange[:size-1][None, :])
                else:
                    ids.fill_(1)
                    pos.fill_(size-1)
                graph.replay()
                if e.spec.get('integrated_selection'):
                    if e.host_pair is not None:
                        if phase == 'prefill':
                            e.host_pair.copy_(output, non_blocking=True)
                        e.delivery_event.record()
                        e.delivery_event.synchronize()
                        e.host_pair.tolist()
                    else:
                        output.cpu().tolist()
                else:
                    torch.cuda.synchronize()
                samples.append(perf_counter() - begin)
        elapsed = perf_counter() - start
        memory = max(0, torch.cuda.memory_allocated() - before)
        stat = Graph(median(samples) if samples else 0., elapsed, memory)
        self.graphs[key] = graph, ids, pos, output
        return stat, dict(phase=phase, bucket=size, elapsed_seconds=elapsed,
                          first_call_seconds=first_call_seconds, graph_seconds=record_seconds,
                          capture_warmup_seconds=warm_seconds, resident_bytes=memory,
                          replay_probe_seconds=stat.seconds if samples else None)

    def record(self, key, adaptive=False):
        stat, event = self.capture(key, measure=adaptive)
        if adaptive:
            keep, victim, net = self.policy.admit(key, stat)
            if victim is not None:
                del self.graphs[victim]
            if not keep:
                del self.graphs[key]
            event.update(kept=keep, evicted=victim, forecast_net_seconds=net)
        else:
            event['kept'] = True
        self.events.append(event)

    def before_request(self, length, max_new):
        self.lengths, self.fallback, self.paths = [], [], Counter()
        if self.strategy == 'adaptive':
            if (key := self.policy.proposal()) is not None:
                self.record(key, adaptive=True)
        elif self.strategy in ('legacy', 'fixed'):
            e = self.engine
            if self.strategy == 'legacy':
                prefill = length if e.spec.get('prefill_graph') else None
                decode = bucket_sizes(e.context) if e.spec.get('target_buckets') else [e.context]
            else:
                decode = [64, 128, 256, 512]
                prefill = next(b for b in decode if b >= length)
            needed = {next(b for b in decode if b >= n)
                      for n in range(length + 1, length + max_new + 1)}
            for size in sorted(needed):
                if ('decode', size) not in self.graphs:
                    self.record(('decode', size))
            if prefill is not None and ('prefill', prefill) not in self.graphs:
                self.record(('prefill', prefill))

    def after_request(self):
        if self.strategy == 'adaptive':
            self.policy.observe(self.lengths, self.fallback)

    def choose_bucket(self, phase, length):
        if self.strategy == 'adaptive':
            return self.policy.costs(phase, length)[1]
        if self.strategy == 'legacy':
            if phase == 'prefill':
                return length if (phase, length) in self.graphs else None
            return bucket_for(length-1, self.engine.context) if self.engine.spec.get('target_buckets') else self.engine.context
        if self.strategy == 'fixed':
            return min((b for p, b in self.graphs if p == phase and length <= b), default=None)
        return None

    def last_logits(self, ids):
        e = self.engine
        if e.position == 0:
            phase, length = 'prefill', ids.shape[1]
        else:
            assert ids.shape[1] == 1
            phase, length = 'decode', e.position + 1
        if length > self.capacity:
            raise ValueError('Request exceeds the bounded benchmark capacity')
        bucket = self.choose_bucket(phase, length)
        self.last_path = phase, length, bucket
        self.paths[(phase, bucket)] += 1
        if phase == 'prefill':
            e.decode_bucket, e.decode_seeded = e.context, False
            if bucket is None:
                prefill = e.prefill_tensor if self.eager_fallback or e.prefill_call is None else e.prefill_call
                value = prefill(ids, e.arange[:length])
            else:
                graph, dest, positions, value = self.graphs[(phase, bucket)]
                self.input_for_bucket(dest, positions, ids)
                graph.replay()
            e.position = length
        else:
            if not e.decode_seeded or not e.spec.get('gpu_feedback'):
                e.static_ids.copy_(ids)
            if not e.decode_seeded or not e.spec.get('gpu_position'):
                e.static_pos.fill_(e.position)
            if bucket is None:
                e.decode_bucket = bucket_for(length-1, self.capacity) if self.eager_fallback else self.capacity
                step = e.step_tensor if self.eager_fallback else e.step_call
                value = step(e.static_ids, e.static_pos)
                if e.host_pair is not None:
                    e.host_pair.copy_(value, non_blocking=True)
            else:
                graph, _, _, value = self.graphs[(phase, bucket)]
                graph.replay()
            e.position, e.decode_seeded = length, True
        return value

    def generate(self, tok, case, settings, banned, stops):
        e = self.engine
        start = perf_counter()
        first = None
        generated, steps, terminal = [], 0, None
        q = e.model.adapter.encode_question(tok, case['question']) if e.question is not None else None
        e.reset(q)
        x = torch.tensor([case['prompt_ids']], device='cuda', dtype=torch.long)
        for _ in range(settings['max_new_tokens']):
            step = perf_counter()
            value = self.last_logits(x)
            if e.spec.get('joint_selection'):
                pair = value if e.spec.get('integrated_selection') else e.select_call(value[0].float())
                if e.host_pair is not None:
                    if not steps:
                        e.host_pair.copy_(pair, non_blocking=True)
                    e.delivery_event.record()
                    e.delivery_event.synchronize()
                    token, finite = e.host_pair.tolist()
                else:
                    token, finite = pair.cpu().tolist()
                if not finite:
                    raise RuntimeError('Nonfinite logits')
                x = pair[:1].reshape(1, 1)
            else:
                z = value[0].float()
                if not torch.isfinite(z).all():
                    raise RuntimeError('Nonfinite logits')
                z[banned] = -torch.inf
                device_token = z.argmax()
                token = int(device_token)
                x = device_token.reshape(1, 1)
            steps += 1
            if self.strategy == 'adaptive':
                phase, length, bucket = self.last_path
                self.lengths.append((phase, length))
                if bucket is None:
                    self.fallback.append((phase, length, perf_counter() - step))
            if token in stops:
                terminal = token
                break
            generated.append(token)
            if first is None:
                first = perf_counter() - start
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        return dict(token_ids=generated, stop_reason='model_terminator' if terminal is not None else 'length_limit',
                    termination_token=terminal, seconds=seconds, first_token_seconds=first,
                    total_tokens_per_second=len(generated)/seconds,
                    decode_tokens_per_second=(len(generated)-1)/(seconds-first) if first is not None and len(generated)>1 else None,
                    model_steps=steps, paths=[[phase, bucket, count] for (phase, bucket), count in self.paths.items()])

