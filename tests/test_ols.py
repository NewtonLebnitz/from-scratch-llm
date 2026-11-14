import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from mlsystem.ols_rbf import token_error_target, orthogonal_select, initialize_supervised, SteepestDescent


class OLSTests(unittest.TestCase):
    def test_target_is_real_token_ce_error(self):
        logits = torch.tensor([[1., 2., 3.], [0., 1., 0.]], requires_grad=True)
        labels = torch.tensor([2, 0])
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
        loss.backward()
        torch.testing.assert_close(token_error_target(logits, labels), -logits.grad)

    def test_duplicate_has_no_new_direction(self):
        dictionary = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        target = torch.stack([dictionary[:, 0]*2 + dictionary[:, 1], dictionary[:, 1]*3], dim=1)
        selected, gains = orthogonal_select(dictionary, target, 3)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(gains), 2)
        self.assertFalse(0 in selected and 2 in selected)

    def test_initialization_is_full_vocabulary(self):
        torch.manual_seed(7)
        hidden = torch.randn(32, 8)
        logits = torch.randn(32, 16)
        labels = torch.randint(16, (32,))
        adapter, audit = initialize_supervised(hidden, logits, labels, dimensions=4, candidates=12, selected_centers=6)
        self.assertEqual(adapter(hidden).shape, logits.shape)
        self.assertLess(audit['regularized_fit_sse'], audit['initial_target_sse'])
        loss = torch.nn.functional.cross_entropy(logits + adapter(hidden), labels)
        loss.backward()
        for p in adapter.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_actual_negative_gradient_line_search(self):
        weight = torch.nn.Parameter(torch.tensor(4.))
        optimizer = SteepestDescent([weight], initial_step=4.)
        outcome = optimizer.step(lambda: (weight-1).square())
        self.assertTrue(outcome['accepted'])
        self.assertLess(outcome['after'], outcome['before'])
        self.assertGreater(outcome['function_evaluations'], 2)
        self.assertAlmostEqual(float(weight.detach()), 1., places=6)


if __name__ == '__main__':
    unittest.main()
