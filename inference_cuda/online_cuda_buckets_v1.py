"""Workload-only online graph policy. All observations follow request arrival."""
from collections import Counter, deque
from dataclasses import dataclass
from statistics import median


@dataclass
class Graph:
    seconds: float
    capture: float
    memory: int


class OnlineBuckets:
    def __init__(self, context, history=32, horizon=64, minimum=8, slots=6,
                 memory=192 * 1024**2, max_probes=8):
        self.context = context
        self.window = deque(maxlen=history)
        self.horizon, self.minimum, self.slots = horizon, minimum, slots
        self.memory, self.max_probes = memory, max_probes
        self.graphs, self.fallback, self.tried = {}, {}, set()
        self.samples = {}
        self.capture_costs = {}
        self.seen = 0

    def width(self, phase, length):
        tile = 8 if phase == 'prefill' else 16
        return min(self.context, ((length + tile - 1) // tile) * tile)

    def observe(self, lengths, fallback):
        self.window.append(Counter(lengths))
        self.seen += 1
        changed = set()
        for phase, length, elapsed in fallback:
            key = phase, self.width(phase, length)
            self.samples.setdefault(key, deque(maxlen=31)).append(elapsed)
            changed.add(key)
        for key in changed:
            if len(self.samples[key]) >= 3:
                self.fallback[key] = median(self.samples[key])

    def costs(self, phase, length, graphs=None):
        graphs = self.graphs if graphs is None else graphs
        base = self.fallback.get((phase, self.width(phase, length)), float('inf'))
        if not graphs:
            return base, None
        return min([(base, None)] + [
            (g.seconds, size) for (kind, size), g in graphs.items()
            if kind == phase and length <= size and g.seconds < base
        ], key=lambda pair: pair[0])

    def demand(self):
        weights = Counter()
        total = 0.0
        for age, row in enumerate(reversed(self.window)):
            weight = .92 ** age
            total += weight
            for key, count in row.items():
                weights[key] += weight * count
        return {key: count * self.horizon / total for key, count in weights.items()} if total else {}

    def proposal(self):
        if self.seen < self.minimum or self.seen % 8 or len(self.tried) >= self.max_probes:
            return None
        demand = self.demand()
        choices = []
        for key in {(p, self.width(p, n)) for p, n in demand} - self.tried - self.graphs.keys():
            phase, size = key
            # Optimistic screening bounds exploration. Measured replacement score
            # below decides admission and charges the full measured setup cost.
            upper = sum(count * self.costs(p, n)[0]
                        for (p, n), count in demand.items()
                        if p == phase and self.width(p, n) == size
                        and (p, size) in self.fallback)
            if upper >= max(.010, self.capture_costs.get(phase, 0.)):
                choices.append((upper, key))
        return max(choices)[1] if choices else None

    def admit(self, key, new):
        self.tried.add(key)
        phase = key[0]
        self.capture_costs[phase] = min(self.capture_costs.get(phase, new.capture), new.capture)
        demand = self.demand()
        variants = []
        if new.memory > self.memory:
            return False, None, None
        for victim in [None, *self.graphs]:
            proposed = dict(self.graphs)
            if victim is not None:
                del proposed[victim]
            proposed[key] = new
            if len(proposed) > self.slots or sum(g.memory for g in proposed.values()) > self.memory:
                continue
            net = -new.capture
            for (phase, length), count in demand.items():
                current = self.costs(phase, length)[0]
                candidate = self.costs(phase, length, proposed)[0]
                if current != float('inf') and candidate != float('inf'):
                    net += count * (current - candidate)
            variants.append((net, victim, proposed))
        if not variants:
            return False, None, None
        net, victim, proposed = max(variants, key=lambda item: item[0])
        if net <= 0:
            return False, None, net
        self.graphs = proposed
        return True, victim, net



