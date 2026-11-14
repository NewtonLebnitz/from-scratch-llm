"""Core invariants: causality, state reset, alpha-zero equivalence and metric meaning."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from mlsystem.config import ModelConfig
from mlsystem.model import LanguageModel, representation_metrics


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.cfg = ModelConfig(vocab_size=512, context_length=32, d_model=64,
                               num_layers=3, num_heads=4, num_kv_heads=2, d_ff=192)

    def test_parameter_count(self):
        model = LanguageModel(self.cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), self.cfg.parameter_estimate())

    def test_alpha_zero_recovers_standard(self):
        standard = LanguageModel(self.cfg)
        hopfield = LanguageModel(replace(self.cfg, attention='hopfield', hopfield_alpha=0, hopfield_alpha_prime=0))
        hopfield.load_state_dict(standard.state_dict(), strict=True)
        ids = torch.randint(512, (2, 16))
        torch.testing.assert_close(standard(ids), hopfield(ids))

    def test_causality_and_request_reset(self):
        for mode in ['standard', 'hopfield']:
            model = LanguageModel(replace(self.cfg, attention=mode))
            ids = torch.randint(512, (1, 16))
            altered = ids.clone()
            altered[:, 8:] = (altered[:, 8:] + 31) % 512
            first, second = model(ids), model(altered)
            torch.testing.assert_close(first[:, :8], second[:, :8])
            torch.testing.assert_close(first, model(ids))

    def test_hopfield_backprop_is_finite(self):
        model = LanguageModel(replace(self.cfg, attention='hopfield'))
        logits = model(torch.randint(512, (2, 16)))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), torch.randint(512, (32,)))
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_metrics_constant_tokens(self):
        metrics = representation_metrics(torch.ones(8, 16))
        self.assertAlmostEqual(metrics['mean_offdiagonal_cosine'], 1., places=5)
        self.assertAlmostEqual(metrics['raw']['stable_rank'], 1., places=5)
        self.assertEqual(metrics['centered']['effective_rank'], 0.)

    def test_invalid_settings(self):
        for cfg in [replace(self.cfg, d_model=66), replace(self.cfg, num_kv_heads=3),
                    replace(self.cfg, hopfield_alpha=1), replace(self.cfg, d_ff=65)]:
            with self.assertRaises(ValueError):
                cfg.validate()


if __name__ == '__main__':
    unittest.main()
