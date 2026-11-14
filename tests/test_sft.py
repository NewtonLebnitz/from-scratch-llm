import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mlsystem.bpe import Tokenizer
from mlsystem.config import ModelConfig
from mlsystem.data import SPECIAL_TOKENS
from mlsystem.model import LanguageModel
from mlsystem.sft import SFTSettings, run_sft, load_sft
from mlsystem.training import file_sha


class SFTTests(unittest.TestCase):
    def fixture(self, root):
        tp = root/'artifacts/tokenizers/fixture'; tp.mkdir(parents=True)
        vocab = {i: bytes([i]) for i in range(256)}
        for marker in SPECIAL_TOKENS: vocab[len(vocab)] = marker.encode()
        while len(vocab) < 512: vocab[len(vocab)] = ('unused-test-token-'+str(len(vocab))).encode()
        tokenizer = Tokenizer(vocab, [], SPECIAL_TOKENS)
        tokenizer.save(tp/'tokenizer.json')
        tokenizer_sha = file_sha(tp/'tokenizer.json')
        config = ModelConfig(vocab_size=512, context_length=128, d_model=64, num_layers=1,
                             num_heads=4, num_kv_heads=2, d_ff=192)
        torch.manual_seed(7)
        model = LanguageModel(config)
        base = root/'artifacts/models/base'; base.mkdir(parents=True)
        torch.save({'format': 'mlsystem-pretrain-v1', 'step': 1, 'model_config': config.dictionary(),
                    'model': model.state_dict(), 'identity': {'tokenizer_sha256': tokenizer_sha}}, base/'best.pt')
        data = root/'datasets/sft/fixture'; data.mkdir(parents=True)
        manifest = {'status': 'ready', 'format': 'mlsystem-sft-v1', 'tokenizer': 'fixture',
                    'tokenizer_sha256': tokenizer_sha, 'files': []}
        for split in ['train', 'dev']:
            path = data/(split+'.jsonl')
            path.write_text(json.dumps({'question': 'VLA?', 'context': 'VLA: vision language action.',
                'answer': 'Vision language action.', 'system': 'Be clear.', 'source_ids': ['test-fixture'],
                'family_id': 'test-fixture-'+split})+'\n', encoding='utf-8')
            manifest['files'].append({'path': path.relative_to(root).as_posix(), 'split': split,
                                      'examples': 1, 'sha256': file_sha(path)})
        mp = data/'manifest.json'; mp.write_text(json.dumps(manifest), encoding='utf-8')
        return mp

    def test_three_generative_routes_real_update_resume_and_reload(self):
        torch.set_num_threads(2)
        tmp = ROOT/'tmp/core-tests'; tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as temporary:
            root = Path(temporary)
            mp = self.fixture(root)
            bp = root/'artifacts/models/base/best.pt'
            original = file_sha(bp)
            for route in ['genlora', 'ols', 'full']:
                with self.subTest(route=route):
                    settings = SFTSettings(route=route, steps=2, microbatch=1, accumulation=1, sequence_length=128,
                        learning_rate=.003, rank=4, groups=2, gaussian_centers=5, calibration_positions=32,
                        pca_dimensions=4, candidate_centers=8, selected_centers=4, eval_every=1, eval_batches=1,
                        save_every=1, initial_step=8.)
                    full = root/'artifacts/models'/route
                    status = run_sft(root, full, bp.relative_to(root), mp, settings, device=torch.device('cpu'))
                    self.assertEqual(status['status'], 'completed')
                    self.assertEqual(status['step'], 2)
                    self.assertEqual(file_sha(bp), original)
                    self.assertEqual(status['parameters'] == status['trainable_parameters'], route == 'full')
                    metrics = [json.loads(x) for x in (full/'metrics.jsonl').read_text().splitlines()]
                    self.assertLess(metrics[-1]['answer_ce'], metrics[0]['answer_ce'])
                    if route == 'ols':
                        self.assertLessEqual(metrics[0]['update']['after'], metrics[0]['update']['before'])
                        self.assertIn('true next answer token', status['initialization']['target'])
                    resumed_dir = root/'artifacts/models'/(route+'-resume')
                    calls = 0
                    def stop():
                        nonlocal calls
                        calls += 1
                        return calls > 1
                    first = run_sft(root, resumed_dir, bp.relative_to(root), mp, settings,
                                    device=torch.device('cpu'), stop_requested=stop)
                    self.assertEqual(first['step'], 1)
                    self.assertEqual(first['status'], 'cancelled')
                    second = run_sft(root, resumed_dir, bp.relative_to(root), mp, settings,
                                     device=torch.device('cpu'), resume=True)
                    self.assertEqual(second['status'], 'completed')
                    a = torch.load(full/'last.pt', weights_only=True)
                    b = torch.load(resumed_dir/'last.pt', weights_only=True)
                    for key in a['payload']['weights']:
                        torch.testing.assert_close(a['payload']['weights'][key], b['payload']['weights'][key], rtol=0, atol=0)
                    loaded, state = load_sft(root, (full/'last.pt').relative_to(root), torch.device('cpu'))
                    scores = loaded(torch.tensor([[256, 85, 76, 65]]))
                    self.assertEqual(scores.shape, (1, 4, 512))
                    self.assertTrue(torch.isfinite(scores).all())
                    self.assertEqual(state['settings']['route'], route)

    def test_invalid_genlora_groups_and_ols_dimensions_rejected(self):
        config = ModelConfig(vocab_size=512, context_length=128, d_model=64, num_layers=1,
                             num_heads=4, num_kv_heads=2, d_ff=192)
        with self.assertRaises(ValueError): SFTSettings(groups=3).validate(config)
        with self.assertRaises(ValueError): SFTSettings(route='ols', calibration_positions=32, pca_dimensions=64).validate(config)
        with self.assertRaises(ValueError): SFTSettings(learning_rate=float('nan')).validate(config)


if __name__ == '__main__': unittest.main()
