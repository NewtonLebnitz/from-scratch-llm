import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from mlsystem.config import ModelConfig
from mlsystem.model import LanguageModel
from mlsystem.genlora import attach, AdapterLinear


class GenLoRATests(unittest.TestCase):
    def test_zero_delta_preserves_full_vocab_model_and_freezes_base(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cfg = ModelConfig(vocab_size=512, context_length=32, d_model=64,
                          num_layers=2, num_heads=4, num_kv_heads=2, d_ff=192)
        model = LanguageModel(cfg)
        ids = torch.randint(512, (2, 16))
        with torch.no_grad():
            original = model(ids)
        paths = attach(model, 'rbf', rank=4, groups=4, centers=7)
        self.assertTrue(paths)
        torch.testing.assert_close(original, model(ids))
        self.assertEqual(model(ids).shape, (2,16,512))
        for m in model.modules():
            if isinstance(m, AdapterLinear):
                self.assertFalse(m.base_layer.weight.requires_grad)
        loss = torch.nn.functional.cross_entropy(model(ids).reshape(-1,512), torch.randint(512,(32,)))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))


if __name__ == '__main__':
    unittest.main()
