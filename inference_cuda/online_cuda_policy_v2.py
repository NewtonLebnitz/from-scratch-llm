"""Causal, cost-based admission for prefill and decode CUDA Graphs."""
from collections import Counter, defaultdict, deque
from statistics import median

from online_cuda_buckets_v1 import Graph


class CapturePolicy:
    def __init__(self, capacity, slots=8, memory=384 << 20):
        self.capacity, self.slots, self.memory = capacity, slots, memory
        self.recent = deque(maxlen=24)
        self.fallback = defaultdict(lambda: deque(maxlen=24))
        self.graphs, self.attempted = {}, {}
        self.measured_setup = defaultdict(lambda: deque(maxlen=4))
        self.measured_ratio = defaultdict(lambda: deque(maxlen=4))
        self.seen = 0

    def width(self, phase, length):
        if phase == 'prefill':
            return min(self.capacity, (length + 15) // 16 * 16)
        size = 64
        while size < length:
            size *= 2
        return min(self.capacity, size)

    def covers(self, key, phase, length):
        kind, size = key
        if kind != phase or size < length:
            return False
        if phase == 'prefill':
            return size - length <= max(7, length // 4)
        return size == self.width(phase, length)

    def observe(self, lengths, samples):
        self.seen += 1
        self.recent.append(Counter(lengths))
        for phase, length, seconds in samples:
            self.fallback[phase, self.width(phase, length)].append(seconds)

    def native_cost(self, phase, length):
        samples = self.fallback.get((phase, self.width(phase, length)))
        return median(samples) if samples else None

    def choose(self, phase, length, graphs=None):
        active = self.graphs if graphs is None else graphs
        base = self.native_cost(phase, length)
        choices = [(g.seconds, key) for key, g in active.items()
                   if self.covers(key, phase, length)]
        if base is not None:
            choices.append((base, None))
        return min(choices, key=lambda item: item[0]) if choices else (None, None)

    def forecast(self):
        demand = Counter()
        weights = 0.
        for age, counts in enumerate(reversed(self.recent)):
            weight = .90 ** age
            weights += weight
            for key, count in counts.items():
                demand[key] += count * weight
        # A short, fixed horizon bounds how much future use we credit.
        return {key: 24 * count / weights for key, count in demand.items()} if weights else {}

    def proposal(self, force_phase=None):
        if len(self.attempted) >= 12 and not force_phase:
            return None
        demand = self.forecast()
        keys = {(p, self.width(p, n)) for p, n in demand}
        candidates = []
        for key in keys - self.graphs.keys():
            phase, size = key
            if force_phase and phase != force_phase:
                continue
            if self.seen - self.attempted.get(key, -100) < 16:
                continue
            samples = self.fallback.get(key)
            if not samples or (len(samples) < 3 and not force_phase):
                continue
            ratio_history = self.measured_ratio[phase]
            ratio = median(ratio_history) if ratio_history else (.65 if phase == 'decode' else .80)
            expected = median(samples) * ratio
            times = self.measured_setup[phase]
            setup = max(.012, median(times)) if times else .025
            covered = [(count, n) for (p, n), count in demand.items() if self.covers(key, p, n)]
            if not covered:
                continue
            saved = sum(count * max(0., (self.choose(phase, n)[0] or 0.) - expected)
                        for count, n in covered)
            if force_phase or saved >= 1.25 * setup:
                candidates.append((saved - setup, key, max(covered)[1]))
        return max(candidates)[1:] if candidates else None

    def admit(self, key, representative, new, force=False):
        phase, _ = key
        self.attempted[key] = self.seen
        self.measured_setup[phase].append(new.capture)
        base = self.native_cost(phase, representative)
        if base:
            self.measured_ratio[phase].append(new.seconds / base)
        variants = []
        for victim in (None, *self.graphs):
            proposed = dict(self.graphs)
            if victim is not None:
                del proposed[victim]
            proposed[key] = new
            if len(proposed) > self.slots or sum(g.memory for g in proposed.values()) > self.memory:
                continue
            gain = -new.capture
            for (p, n), count in self.forecast().items():
                before, _ = self.choose(p, n)
                after, _ = self.choose(p, n, proposed)
                if before is not None and after is not None:
                    gain += count * (before - after)
            variants.append((gain, victim, proposed))
        if not variants:
            return False, None, None
        gain, victim, proposed = max(variants, key=lambda item: item[0])
        if gain <= 0 and not force:
            return False, None, gain
        self.graphs = proposed
        return True, victim, gain
