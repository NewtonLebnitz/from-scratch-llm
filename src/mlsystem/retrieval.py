"""Versioned multilingual embeddings and a trainable batch-SOM retrieval index.

This component returns source chunks. It never writes an answer or changes LM
weights; the selected autoregressive SFT model must generate the answer itself.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import time
import sys
import numpy as np
import torch
from .training import atomic_json, file_sha


def encoder_safe_chunks(documents, token_length, *, limit=512, overlap=40):
    """Split only overlength source chunks; preserve parents and character spans.

    token_length must count the encoder prefix and special tokens. Original text
    is covered without silent cropping. The encoder still checks its own limit.
    """
    if not 32 <= limit <= 512 or not 0 <= overlap <= 100: raise ValueError('Invalid encoder chunk limits')
    result = []; parents = 0
    for row in documents:
        text = row['text']
        if token_length(text) <= limit:
            result.append(row); continue
        parents += 1; start = 0
        while start < len(text):
            end = len(text)
            while token_length(text[start:end]) > limit:
                end = start+(end-start)*3//4
                if end <= start: raise ValueError('A source character exceeds the encoder budget')
            # Prefer readable boundaries inside the safe region when possible.
            lower = start+(end-start)//2
            boundary = max(text.rfind('\n', lower, end), text.rfind('。', lower, end), text.rfind('. ', lower, end))
            if boundary > start and end < len(text) and token_length(text[start:boundary+1]) <= limit:
                end = boundary+1
            piece = text[start:end]
            digest = hashlib.sha256(piece.encode('utf-8')).hexdigest()
            result.append(dict(row, id=row['id']+'-enc-'+str(start)+'-'+str(end), text=piece,
                sha256=digest, parent_chunk_id=row['id'], parent_character_span=[start,end],
                encoder_split='512-token bounded; prefix and special tokens included'))
            if len(result) > 1_000_000: raise ValueError('Encoder splitting exceeds one million chunks')
            if end == len(text): break
            start = max(start+1, end-min(overlap, (end-start)//4))
    return result, dict(original_chunks=len(documents), indexed_chunks=len(result),
                        split_parent_chunks=parents, token_limit=limit, silent_truncations=0)


class MultilingualEncoder:
    def __init__(self, root, device='cpu'):
        root = Path(root).resolve()
        path = root/'artifacts/encoders/multilingual-e5-small'
        self.manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('status') != 'ready' or self.manifest['model'] != 'intfloat/multilingual-e5-small':
            raise ValueError('Need the registered and verified multilingual encoder')
        for row in self.manifest['files']:
            candidate = (root/row['path']).resolve()
            if not candidate.is_relative_to(path) or file_sha(candidate) != row['sha256']:
                raise ValueError('Encoder file path/hash mismatch')
        self.identity = file_sha(path/'manifest.json')
        # Transformers is used ONLY for this separately identified pretrained
        # retrieval encoder. Our decoder LM, adapters and training are handwritten.
        dependency_path = root/'vendor'/f'retrieval-{sys.platform}-py{sys.version_info.major}{sys.version_info.minor}'
        if dependency_path.is_dir():
            if not dependency_path.resolve().is_relative_to(root/'vendor'): raise ValueError('Retrieval dependency path escaped workspace')
            sys.path.insert(0, str(dependency_path))
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True).to(device).eval()
        self.device = torch.device(device)
        self.token_audit = {'encoded_texts': 0, 'maximum_tokens': 0, 'truncated_texts': 0,
                            'limit': 512, 'overflow_policy': 'reject; rebuild smaller source chunks'}
        if self.model.config.hidden_size != 384:
            raise ValueError('Unexpected multilingual E5 embedding dimension')

    def passage_tokens(self, text):
        return len(self.tokenizer.encode('passage: '+text, truncation=False))

    @torch.no_grad()
    def encode(self, texts, *, query=False, batch_size=16):
        if not texts or not 1 <= batch_size <= 64: raise ValueError('Empty text list or invalid embedding batch')
        prefix = 'query: ' if query else 'passage: '
        results = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset+batch_size]
            if any(not isinstance(t, str) or not t.strip() or len(t) > 20000 for t in batch):
                raise ValueError('Embedding text must be nonempty and at most 20,000 characters')
            tokens = self.tokenizer([prefix+t for t in batch], padding=True,
                                    truncation=False, return_tensors='pt')
            lengths = tokens['attention_mask'].sum(dim=1)
            maximum = int(lengths.max().item())
            if maximum > 512:
                raise ValueError(f'Embedding input needs {maximum} tokens including prefix; limit is 512. '
                                 'Use smaller document chunks or a shorter retrieval query. Text was not truncated.')
            self.token_audit['encoded_texts'] += len(batch)
            self.token_audit['maximum_tokens'] = max(self.token_audit['maximum_tokens'], maximum)
            tokens = {key:value.to(self.device) for key,value in tokens.items()}
            hidden = self.model(**tokens).last_hidden_state.float()
            mask = tokens['attention_mask'].bool()
            mean = hidden.masked_fill(~mask[..., None], 0.).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
            results.append(torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy())
        return np.concatenate(results).astype(np.float32)


@dataclass(frozen=True)
class SOMSettings:
    width: int = 8
    height: int = 8
    dimensions: int = 32
    epochs: int = 10
    sample_fraction: float = .2
    max_samples: int = 100000
    learning_rate: float = .5
    initial_radius: float = 4.
    final_radius: float = 1.
    batch_size: int = 256
    seed: int = 7

    def validate(self, count, input_dimensions):
        bounds = {'width': (2, 64), 'height': (2, 64), 'dimensions': (2, min(128, input_dimensions)),
                  'epochs': (1, 50), 'max_samples': (16, 100000), 'batch_size': (16, 1024), 'seed': (0, 2147483647)}
        for name, (lo, hi) in bounds.items():
            if type(getattr(self, name)) is not int or not lo <= getattr(self, name) <= hi:
                raise ValueError(f'{name} must be integer in [{lo}, {hi}]')
        if count > 1000000: raise ValueError('Split index into collections above one million chunks')
        if not .1 <= self.sample_fraction <= 1 or not 0 < self.learning_rate <= 1:
            raise ValueError('Invalid sample fraction or SOM learning rate')
        if not 0 < self.final_radius <= self.initial_radius <= max(self.width, self.height):
            raise ValueError('Need 0 < final radius <= initial radius <= maximum grid width')
        sample_count = min(self.max_samples, int(count*self.sample_fraction))
        if sample_count <= self.dimensions or sample_count < self.width*self.height:
            raise ValueError('Training sample count must exceed PCA dimensions and cover at least one sample per grid neuron; reduce grid/dimensions or increase sampling')
        return sample_count


def distances(x, centers):
    return (x.square().sum(-1, keepdim=True)+centers.square().sum(-1)[None, :]-2*x@centers.T).clamp_min(0)


@torch.no_grad()
def fit_som(vectors, settings, *, device='cpu', stop_requested=lambda: False):
    vectors = torch.as_tensor(vectors, dtype=torch.float32, device=device)
    if vectors.ndim != 2 or not torch.isfinite(vectors).all(): raise ValueError('Need finite embedding matrix [chunks, dimensions]')
    norms = vectors.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=.002, rtol=0):
        raise ValueError('Input sentence embeddings must be L2 normalized')
    n = settings.validate(*vectors.shape)
    rng = torch.Generator(device=vectors.device).manual_seed(settings.seed)
    rows = torch.randperm(len(vectors), generator=rng, device=vectors.device)[:n]
    training = vectors[rows]
    mean = training.mean(0)
    covariance = (training-mean).T@(training-mean)/(n-1)
    values, axes = torch.linalg.eigh(covariance)
    projection = axes[:, -settings.dimensions:].flip(1).contiguous()
    reduced = (vectors-mean)@projection
    training = reduced[rows]
    prototypes = training[torch.randperm(n, generator=rng, device=vectors.device)[:settings.width*settings.height]].clone()
    grid = torch.cartesian_prod(torch.arange(settings.height, device=vectors.device),
                                torch.arange(settings.width, device=vectors.device)).float()
    grid_dist2 = distances(grid, grid)
    history = []
    for epoch in range(settings.epochs):
        if stop_requested(): raise InterruptedError('SOM stopped before completing an epoch; no ready index published')
        fraction = epoch/max(settings.epochs-1, 1)
        radius = settings.initial_radius+(settings.final_radius-settings.initial_radius)*fraction
        # Gaussian neighborhood centered on each sample's BMU. Batch SOM uses
        # accumulated weighted means followed by a learning-rate interpolation.
        sums = torch.zeros_like(prototypes)
        weights = torch.zeros(len(prototypes), device=vectors.device)
        for start in range(0, n, settings.batch_size):
            batch = training[start:start+settings.batch_size]
            bmu = distances(batch, prototypes).argmin(1)
            neighborhood = torch.exp(-grid_dist2[bmu]/(2*radius**2))
            sums += neighborhood.T@batch
            weights += neighborhood.sum(0)
        active = weights > 1e-12
        rate = settings.learning_rate*(1-.9*fraction)
        target = sums[active]/weights[active, None]
        prototypes[active] = (1-rate)*prototypes[active]+rate*target
        error = 0.
        for start in range(0, n, settings.batch_size):
            error += float(distances(training[start:start+settings.batch_size], prototypes).min(1).values.sqrt().sum())
        history.append({'epoch': epoch+1, 'radius': radius, 'learning_rate': rate,
                        'training_quantization_error': error/n})
    bmus, errors = [], []
    for start in range(0, len(reduced), settings.batch_size):
        ds, indices = distances(reduced[start:start+settings.batch_size], prototypes).min(1)
        bmus.append(indices.cpu().numpy()); errors.append(ds.sqrt().cpu().numpy())
    result = {'mean': mean.cpu().numpy(), 'projection': projection.cpu().numpy(),
              'prototypes': prototypes.cpu().numpy(), 'bmu': np.concatenate(bmus),
              'quantization_errors': np.concatenate(errors), 'sample_rows': rows.cpu().numpy()}
    report = {'settings': asdict(settings), 'training_chunks': n, 'indexed_chunks': len(vectors),
              'retained_variance_fraction': float(values[-settings.dimensions:].sum()/values.sum().clamp_min(1e-20)),
              'mean_quantization_error': float(result['quantization_errors'].mean()), 'history': history,
              'error_definition': 'Mean Euclidean ||P^T(e_i - sample_mean) - w_BMU(i)||_2; PCA coordinates without whitening; inputs are L2-normalized embeddings.',
              'coordinates': 'flat BMU = y * grid_width + x; rectangular non-toroidal grid',
              'algorithm': 'Gaussian batch SOM with linearly decaying radius and learning-rate interpolation'}
    return result, report


def build_index(root, output, documents, embeddings, encoder_identity, settings, *, device='cpu', stop_requested=lambda: False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root/'artifacts/som') or output == root/'artifacts/som': raise ValueError('Invalid index output')
    if len(documents) != len(embeddings) or not documents: raise ValueError('Document/vector count mismatch')
    ids = set()
    for document in documents:
        if not document.get('id') or not document.get('text') or not document.get('source_id'):
            raise ValueError('Every chunk needs an ID, text and source ID')
        if document['id'] in ids: raise ValueError('Duplicate chunk IDs')
        ids.add(document['id'])
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'building', 'format': 'mlsystem-som-v1', 'encoder_identity': encoder_identity,
             'settings': asdict(settings), 'document_count': len({d['source_id'] for d in documents}),
             'chunk_count': len(documents)}
    atomic_json(output/'manifest.json', state)
    started = time.perf_counter()
    try:
        arrays, report = fit_som(embeddings, settings, device=device, stop_requested=stop_requested)
        np.savez_compressed(output/'som.npz', **arrays)
        np.save(output/'embeddings.npy', np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
        with (output/'documents.jsonl').open('w', encoding='utf-8') as stream:
            for row in documents: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        state.update(status='ready', report=report, seconds=time.perf_counter()-started,
                     files={name: file_sha(output/name) for name in ['som.npz', 'embeddings.npy', 'documents.jsonl']})
    except BaseException as exc:
        state.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally: atomic_json(output/'manifest.json', state)
    return state


class SOMIndex:
    def __init__(self, root, path, encoder_identity):
        root, path = Path(root).resolve(), Path(path).resolve()
        if not path.is_relative_to(root/'artifacts/som'): raise ValueError('Index outside registry')
        state = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if state.get('status') != 'ready' or state.get('format') != 'mlsystem-som-v1' or state['encoder_identity'] != encoder_identity:
            raise ValueError('Index is not ready or encoder differs')
        for name, digest in state['files'].items():
            if name not in ['som.npz', 'embeddings.npy', 'documents.jsonl'] or file_sha(path/name) != digest:
                raise ValueError('Index file changed')
        with np.load(path/'som.npz', allow_pickle=False) as data:
            self.arrays = {name: data[name] for name in data.files}
        self.embeddings = np.load(path/'embeddings.npy', mmap_mode='r', allow_pickle=False)
        self.documents = [json.loads(line) for line in (path/'documents.jsonl').read_text(encoding='utf-8').splitlines()]
        self.manifest = state

    def close(self):
        mapping = getattr(self.embeddings, '_mmap', None)
        if mapping is not None and not mapping.closed: mapping.close()

    def __enter__(self): return self

    def __exit__(self, *args): self.close()

    def search(self, query_vector, *, top_k=5, nearest_nodes=8, max_candidates=10000):
        if getattr(self.embeddings, '_mmap', None) is not None and self.embeddings._mmap.closed:
            raise ValueError('Index is closed; reopen it before searching')
        if not 1 <= top_k <= 20 or not 1 <= nearest_nodes <= 128 or not 20 <= max_candidates <= 50000:
            raise ValueError('Invalid retrieval bounds')
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != self.embeddings.shape[1:] or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1) > .002:
            raise ValueError('Query needs one normalized vector from the index encoder')
        a = self.arrays
        projected = (vector-a['mean'])@a['projection']
        node_distances = np.square(a['prototypes']-projected).sum(1)
        nodes = np.argsort(node_distances)[:nearest_nodes]
        candidates = np.flatnonzero(np.isin(a['bmu'], nodes))
        # Empty selected cells are expanded by prototype proximity. This is
        # explicitly reported, not a silent switch to fabricated evidence.
        expanded = False
        if len(candidates) < top_k:
            expanded = True
            for node in np.argsort(node_distances)[nearest_nodes:]:
                candidates = np.flatnonzero(np.isin(a['bmu'], np.append(nodes, node)))
                nodes = np.append(nodes, node)
                if len(candidates) >= top_k: break
        capped = len(candidates) > max_candidates
        if capped:
            # Bound reranking using PCA distance; no first-N document-order bias.
            candidate_reduced = (self.embeddings[candidates]-a['mean'])@a['projection']
            distances_to_query = np.square(candidate_reduced-projected).sum(1)
            candidates = candidates[np.argsort(distances_to_query)[:max_candidates]]
        similarities = np.asarray(self.embeddings[candidates]@vector)
        order = np.argsort(-similarities, kind='stable')[:top_k]
        width = self.manifest['settings']['width']
        results = []
        for i in order:
            index = int(candidates[i]); node = int(a['bmu'][index])
            results.append(dict(self.documents[index], score=float(similarities[i]), bmu=node,
                                x=node % width, y=node // width, quantization_error=float(a['quantization_errors'][index])))
        return {'results': results, 'candidate_count': len(candidates), 'selected_nodes': nodes.tolist(),
                'expanded_empty_cells': expanded, 'candidate_cap_applied': capped,
                'note': 'Cosine similarity is a ranking score, not evidence sufficiency or answer correctness.'}
