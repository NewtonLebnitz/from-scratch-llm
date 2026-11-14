"""Byte BPE with incremental pair counts and deterministic tie breaking."""
import base64
import heapq
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import regex

PAT = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def special_pattern(special_tokens):
    tokens = sorted(set(special_tokens or []), key=lambda s: (-len(s), s))
    if any(not s for s in tokens):
        raise ValueError('Special tokens must be non-empty')
    return regex.compile('|'.join(regex.escape(s) for s in tokens)) if tokens else None


class ReversePair:
    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    specials = list(dict.fromkeys(special_tokens))
    if vocab_size < 256 + len(specials):
        raise ValueError('Vocabulary must include 256 bytes and all special tokens')
    text = Path(input_path).read_text(encoding='utf-8')
    sp = special_pattern(specials)
    parts = sp.split(text) if sp else [text]
    counts = Counter()
    for part in parts:
        counts.update(m.group().encode('utf-8') for m in PAT.finditer(part))
    del parts, text
    return _learn_bpe(counts, vocab_size, specials)


def train_bpe_documents(documents, vocab_size, special_tokens, *, max_bytes=16*1024**2,
                        max_unique_pieces=1000000):
    """Fit an explicitly bounded TRAIN document sample, without reading the corpus whole."""
    specials = list(dict.fromkeys(special_tokens))
    if vocab_size < 256 + len(specials) or max_bytes <= 0:
        raise ValueError('Invalid vocabulary/sample budget')
    sp = special_pattern(specials)
    counts, used_bytes, used_documents = Counter(), 0, 0
    for text in documents:
        data = text.encode('utf-8')
        if used_bytes + len(data) > max_bytes:
            continue  # Keep complete documents; never cut a Unicode sequence/pre-token.
        for part in sp.split(text) if sp else [text]:
            counts.update(m.group().encode('utf-8') for m in PAT.finditer(part))
        used_bytes += len(data)
        used_documents += 1
        if len(counts) > max_unique_pieces:
            raise ValueError('BPE sample has too many unique pre-tokens; reduce the sample budget')
        if used_bytes == max_bytes:
            break
    if not counts:
        raise ValueError('No usable BPE training documents')
    vocab, merges = _learn_bpe(counts, vocab_size, specials)
    return vocab, merges, {'sample_bytes':used_bytes, 'sample_documents':used_documents,
                            'unique_pre_tokens':len(counts), 'actual_vocab_size':len(vocab)}


def _learn_bpe(counts, vocab_size, specials):
    vocab = {i: bytes([i]) for i in range(256)}
    for s in specials:
        vocab[len(vocab)] = s.encode('utf-8')
    words = [tuple(bytes([b]) for b in w) for w in counts]
    frequencies = list(counts.values())
    pair_counts = Counter()
    locations = defaultdict(set)
    for wi, word in enumerate(words):
        for pair, n in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += frequencies[wi] * n
            locations[pair].add(wi)
    heap = [(-n, ReversePair(p), p) for p, n in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(vocab) < vocab_size and heap:
        neg_count, _, pair = heapq.heappop(heap)
        if pair_counts[pair] != -neg_count or pair_counts[pair] == 0:
            continue
        merged = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged
        changed = set()
        for wi in list(locations[pair]):
            old = words[wi]
            before = Counter(zip(old, old[1:]))
            new, i = [], 0
            while i < len(old):
                if i+1 < len(old) and (old[i], old[i+1]) == pair:
                    new.append(merged)
                    i += 2
                else:
                    new.append(old[i])
                    i += 1
            words[wi] = tuple(new)
            after = Counter(zip(new, new[1:]))
            for p in before.keys() | after.keys():
                delta = (after[p] - before[p]) * frequencies[wi]
                if delta:
                    pair_counts[p] += delta
                    changed.add(p)
                if after[p]:
                    locations[p].add(wi)
                else:
                    locations[p].discard(wi)
        for p in changed:
            if pair_counts[p] > 0:
                heapq.heappush(heap, (-pair_counts[p], ReversePair(p), p))
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.special_re = special_pattern(self.special_tokens)
        self.byte_to_id = {b: i for i, b in self.vocab.items()}
        for s in self.special_tokens:
            b = s.encode('utf-8')
            if b not in self.byte_to_id:
                idx = max(self.vocab, default=-1) + 1
                self.vocab[idx] = b
                self.byte_to_id[b] = idx
        self.special_ids = {s: self.byte_to_id[s.encode('utf-8')] for s in self.special_tokens}
        self.ranks = {p: i for i, p in enumerate(merges)}
        # Bounded cache: useful for repetitive prose, safe for streaming large files.
        self._encode_piece = lru_cache(maxsize=256)(self._merge_piece)

    def _merge_piece(self, piece):
        if len(piece) <= 32:
            return self._merge_piece_slow(piece)
        # Long Chinese pre-tokens otherwise repeatedly rescan the entire byte list.
        # A versioned linked list + heap keeps the same BPE rank/leftmost ordering.
        tokens = [bytes([b]) for b in piece]
        n = len(tokens)
        previous = [i-1 for i in range(n)]
        following = [i+1 for i in range(n)]
        following[-1] = -1
        versions = [0]*n
        alive = [True]*n
        heap = []
        def push(left):
            if left < 0:
                return
            right = following[left]
            if right < 0:
                return
            rank = self.ranks.get((tokens[left], tokens[right]))
            if rank is not None:
                heapq.heappush(heap,(rank,left,right,versions[left],versions[right]))
        for left in range(n-1):
            push(left)
        while heap:
            _,left,right,vl,vr = heapq.heappop(heap)
            if not alive[left] or not alive[right] or following[left] != right or versions[left] != vl or versions[right] != vr:
                continue
            tokens[left] += tokens[right]
            alive[right] = False
            versions[left] += 1
            versions[right] += 1
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            push(previous[left])
            push(left)
        result=[]
        index=0
        while index >= 0:
            result.append(self.byte_to_id[tokens[index]])
            index=following[index]
        return tuple(result)

    def _merge_piece_slow(self, piece):
        tokens = [bytes([b]) for b in piece]
        while len(tokens) > 1:
            pairs = zip(tokens, tokens[1:])
            best = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            if best not in self.ranks:
                break
            new, i = [], 0
            while i < len(tokens):
                if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == best:
                    new.append(tokens[i]+tokens[i+1])
                    i += 2
                else:
                    new.append(tokens[i])
                    i += 1
            tokens = new
        return tuple(self.byte_to_id[t] for t in tokens)

    def _ordinary(self, text):
        for m in PAT.finditer(text):
            yield from self._encode_piece(m.group().encode('utf-8'))

    def encode(self, text):
        return list(self._encode_text(text))

    def _encode_text(self, text):
        pos = 0
        if self.special_re:
            for m in self.special_re.finditer(text):
                yield from self._ordinary(text[pos:m.start()])
                yield self.special_ids[m.group()]
                pos = m.end()
        yield from self._ordinary(text[pos:])

    def decode(self, ids):
        return b''.join(self.vocab[int(i)] for i in ids).decode('utf-8', errors='replace')

    def encode_iterable(self, iterable):
        # Keep the unstable final pre-tokens, including split contractions/specials.
        buffer = ''
        for chunk in iterable:
            buffer += chunk
            stable_end = len(buffer)
            # Exclude an unfinished special from regex pre-tokenization. Otherwise
            # it can incorrectly turn preceding trailing whitespace into separate
            # tokens before the special is completed in the next chunk.
            for s in self.special_tokens:
                for n in range(1, min(len(s), len(buffer)+1)):
                    if buffer.endswith(s[:n]):
                        stable_end = min(stable_end, len(buffer)-n)
            boundary = 0
            scan_start = 0
            if self.special_re:
                matches = list(self.special_re.finditer(buffer, endpos=stable_end))
                if matches:
                    boundary = scan_start = matches[-1].end()
            tail = buffer[scan_start:stable_end]
            matches = list(PAT.finditer(tail))
            if len(matches) > 2:
                boundary = max(boundary, scan_start+matches[-2].start())
            if boundary:
                yield from self._encode_text(buffer[:boundary])
                buffer = buffer[boundary:]
        yield from self._encode_text(buffer)

    def save(self, path):
        enc = lambda b: base64.b64encode(b).decode('ascii')
        obj = dict(vocab={str(i): enc(b) for i,b in self.vocab.items()},
                   merges=[[enc(a),enc(b)] for a,b in self.merges], special_tokens=self.special_tokens)
        Path(path).write_text(json.dumps(obj), encoding='utf-8')

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Our JSON format uses base64 so every possible byte is losslessly represented.
        vocab = json.loads(Path(vocab_filepath).read_text(encoding='utf-8'))
        merges = json.loads(Path(merges_filepath).read_text(encoding='utf-8'))
        return cls({int(i): base64.b64decode(b) for i,b in vocab.items()},
                   [(base64.b64decode(a),base64.b64decode(b)) for a,b in merges], special_tokens)

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls({int(i): base64.b64decode(b) for i,b in obj['vocab'].items()},
                   [(base64.b64decode(a),base64.b64decode(b)) for a,b in obj['merges']], obj['special_tokens'])
