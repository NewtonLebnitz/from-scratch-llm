"""Online admission over the existing exact greedy inference engine."""
from collections import Counter
from statistics import median
from time import perf_counter

import torch

from online_cuda_buckets_v1 import Graph
from online_cuda_engine_v1 import ScheduledEngine
from online_cuda_policy_v2 import CapturePolicy


class OnlineEngine(ScheduledEngine):
    def __init__(self, engine, *, probe=False, capture_enabled=True):
        super().__init__(engine, 'adaptive', compile_preset='default',
                         compact_capture=True, eager_capture=True, eager_fallback=True)
        self.policy = CapturePolicy(self.capacity)
        self.probe = probe
        self.capture_enabled = capture_enabled
        self.smoked = set()
        self.captures = 0
        self.request_index = 0

    def before_request(self, length, max_new):
        self.lengths, self.fallback, self.paths = [], [], Counter()
        force = next((phase for phase in ('prefill', 'decode') if phase not in self.smoked), None) if self.probe else None
        proposal = self.policy.proposal(force) if self.capture_enabled and self.captures < 12 else None
        if proposal is not None:
            key, representative = proposal
            self.captures += 1
            self.record_candidate(key, representative, force=bool(force))
        self.request_index += 1

    def choose_bucket(self, phase, length):
        if self.probe:
            return min((size for kind, size in self.graphs
                        if self.policy.covers((kind, size), phase, length)), default=None)
        _, key = self.policy.choose(phase, length)
        return None if key is None else key[1]

    def record_candidate(self, key, representative, *, force):
        phase, size = key
        e = self.engine
        torch.cuda.synchronize()
        started = perf_counter()
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        e.position, e.decode_seeded = 0, False
        if phase == 'prefill':
            e.decode_bucket = e.context
            ids = torch.ones((1, size), device='cuda', dtype=torch.long)
            positions = e.arange[:size].clone()
            sample = torch.ones((1, representative), device='cuda', dtype=torch.long)
            self.input_for_bucket(ids, positions, sample)
            call = e.prefill_tensor
        else:
            e.decode_bucket = size
            ids, positions = e.static_ids, e.static_pos
            call = e.step_tensor

        def inputs():
            if phase == 'decode':
                ids.fill_(1)
                positions.fill_(representative - 1)

        # Prime libraries and Triton on the stream used for capture, once.
        stream = self.capture_stream
        first = perf_counter()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            inputs()
            call(ids, positions)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        warm_seconds = perf_counter() - first
        first = perf_counter()
        inputs()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = call(ids, positions)
            if phase == 'decode' and e.host_pair is not None:
                e.host_pair.copy_(output, non_blocking=True)
        capture_seconds = perf_counter() - first
        samples = []
        for _ in range(3):
            start = perf_counter()
            if phase == 'prefill':
                self.input_for_bucket(ids, positions, sample)
            else:
                inputs()
            graph.replay()
            if e.host_pair is not None:
                if phase == 'prefill':
                    e.host_pair.copy_(output, non_blocking=True)
                e.delivery_event.record()
                e.delivery_event.synchronize()
                e.host_pair.tolist()
            else:
                output.cpu().tolist()
            samples.append(perf_counter() - start)
        elapsed = perf_counter() - started
        memory = max(0, torch.cuda.memory_allocated() - allocated,
                     torch.cuda.memory_reserved() - reserved)
        stats = Graph(median(samples), elapsed, memory)
        keep, victim, net = self.policy.admit(key, representative, stats, force)
        if victim is not None:
            del self.graphs[victim]
        if keep:
            self.graphs[key] = graph, ids, positions, output
        self.smoked.add(phase)
        self.events.append(dict(phase=phase, bucket=size, representative_tokens=representative,
                                before_request=self.request_index, observations=self.policy.seen,
                                elapsed_seconds=elapsed, first_call_seconds=warm_seconds,
                                graph_seconds=capture_seconds, capture_warmup_seconds=warm_seconds,
                                resident_bytes=memory, replay_probe_seconds=stats.seconds,
                                kept=keep, evicted=victim, forecast_net_seconds=net))
