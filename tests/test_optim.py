import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from mlsystem.optim import AdamW, cross_entropy


class OptimTests(unittest.TestCase):
    def test_handwritten_adamw_matches_reference(self):
        a = torch.nn.Parameter(torch.tensor([1., -2., 0.01], dtype=torch.float64))
        b = torch.nn.Parameter(a.detach().clone())
        ours = AdamW([a], lr=.02, betas=(.9,.95), eps=1e-6, weight_decay=.1)
        reference = torch.optim.AdamW([b], lr=.02, betas=(.9,.95), eps=1e-6, weight_decay=.1)
        for index in range(7):
            gradient = torch.tensor([.03, -.2, 1e-8], dtype=torch.float64) * (index+1)
            a.grad, b.grad = gradient.clone(), gradient.clone()
            ours.step()
            reference.step()
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_masked_answer_loss_and_gradient(self):
        torch.manual_seed(3)
        a = torch.randn(2, 4, 7, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        labels = torch.tensor([[-100,-100,3,2], [-100,1,5,-100]])
        ours = cross_entropy(a, labels)
        reference = torch.nn.functional.cross_entropy(b.reshape(-1,7), labels.reshape(-1), ignore_index=-100)
        ours.backward()
        reference.backward()
        torch.testing.assert_close(ours, reference)
        torch.testing.assert_close(a.grad,b.grad)


if __name__ == '__main__':
    unittest.main()
