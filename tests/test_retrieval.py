import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mlsystem.retrieval import SOMSettings, fit_som, build_index, SOMIndex, MultilingualEncoder, encoder_safe_chunks


class RetrievalTests(unittest.TestCase):
    def test_encoder_chunking_preserves_every_character_and_parent(self):
        text = ''.join(chr(0x4e00+i%500) for i in range(1300))
        source = dict(id='paper-1', source_id='paper', text=text)
        chunks, report = encoder_safe_chunks([source], lambda s:len(s)+5, limit=128, overlap=20)
        self.assertEqual(report['split_parent_chunks'], 1)
        covered = [False]*len(text)
        for row in chunks:
            start,end = row['parent_character_span']
            self.assertEqual(row['text'], text[start:end])
            self.assertLessEqual(len(row['text'])+5, 128)
            self.assertEqual(row['parent_chunk_id'], 'paper-1')
            covered[start:end] = [True]*(end-start)
        self.assertTrue(all(covered))
        unchanged, report = encoder_safe_chunks([dict(source,text='short')], lambda s:len(s)+5)
        self.assertEqual(unchanged[0]['id'], 'paper-1')
        self.assertEqual(report['split_parent_chunks'], 0)

    def test_encoder_rejects_overlength_before_model_forward(self):
        encoder = MultilingualEncoder.__new__(MultilingualEncoder)
        encoder.device = torch.device('cpu')
        encoder.token_audit = dict(encoded_texts=0, maximum_tokens=0)
        def tokenizer(texts, **kwargs):
            self.assertFalse(kwargs['truncation'])
            return {'input_ids': torch.ones((len(texts), 513), dtype=torch.long),
                    'attention_mask': torch.ones((len(texts), 513), dtype=torch.long)}
        encoder.tokenizer = tokenizer
        # No model is present: the limit must be checked before any forward.
        with self.assertRaisesRegex(ValueError, 'not truncated'):
            encoder.encode(['A long source paragraph'])
        self.assertEqual(encoder.token_audit['encoded_texts'], 0)

    def test_actual_som_fit_save_select_and_retrieve(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(40, 16)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        settings = SOMSettings(width=2, height=2, dimensions=4, epochs=3, sample_fraction=1., initial_radius=2.)
        arrays, report = fit_som(embeddings, settings)
        self.assertEqual(arrays['bmu'].shape, (40,))
        self.assertEqual(report['training_chunks'], 40)
        self.assertEqual(len(report['history']), 3)
        reduced = (embeddings-arrays['mean'])@arrays['projection']
        explicit = np.linalg.norm(reduced-arrays['prototypes'][arrays['bmu']], axis=1)
        np.testing.assert_allclose(arrays['quantization_errors'], explicit, rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(report['mean_quantization_error'], float(explicit.mean()), places=5)
        tmp = ROOT/'tmp/core-tests'; tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as temporary:
            root = Path(temporary); output = root/'artifacts/som/test'
            documents = [{'id': f'chunk-{i}', 'source_id': f'paper-{i//2}', 'text': f'Source paragraph {i}'} for i in range(40)]
            result = build_index(root, output, documents, embeddings, 'fixture-encoder-hash', settings)
            self.assertEqual(result['document_count'], 20)
            self.assertEqual(result['chunk_count'], 40)
            index = SOMIndex(root, output, 'fixture-encoder-hash')
            matches = index.search(embeddings[7], nearest_nodes=4, top_k=3)
            self.assertEqual(matches['results'][0]['id'], 'chunk-7')
            self.assertAlmostEqual(matches['results'][0]['score'], 1., places=5)
            with self.assertRaises(ValueError): SOMIndex(root, output, 'other-encoder')
            (output/'documents.jsonl').write_text('{}\n', encoding='utf-8')
            with self.assertRaises(ValueError): SOMIndex(root, output, 'fixture-encoder-hash')
            index.close()

    def test_invalid_configuration_and_cancel_never_publishes_ready(self):
        with self.assertRaises(ValueError): SOMSettings().validate(20, 384)
        with self.assertRaises(ValueError): SOMSettings(initial_radius=float('nan')).validate(5000, 384)
        x = np.eye(16, dtype=np.float32)
        settings = SOMSettings(width=2, height=2, dimensions=4, epochs=2, sample_fraction=1., initial_radius=2.)
        tmp = ROOT/'tmp/core-tests'; tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as temporary:
            root = Path(temporary); output = root/'artifacts/som/stopped'
            docs = [{'id': str(i), 'source_id': 'fixture', 'text': str(i)} for i in range(16)]
            with self.assertRaises(InterruptedError):
                build_index(root, output, docs, x, 'test', settings, stop_requested=lambda: True)
            state = json.loads((output/'manifest.json').read_text())
            self.assertEqual(state['status'], 'cancelled')
            self.assertNotIn('files', state)


if __name__ == '__main__': unittest.main()
