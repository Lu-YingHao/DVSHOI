import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from ops import binary_focal_loss_with_logits
from upt import UPT
from utils import CustomisedDLE


class TestHOILoss(unittest.TestCase):
    def setUp(self):
        self.model = UPT(nn.Identity(), nn.Identity(), nn.Identity(), 0, 2)
        self.target = dict(
            boxes_h=torch.tensor([[.5, .5, .2, .2]]),
            boxes_o=torch.tensor([[.7, .7, .2, .2]]),
            size=torch.tensor([100, 100]), labels=torch.tensor([1]),
        )

    def loss(self, boxes, logits, prior=None):
        count = len(logits)
        humans = torch.zeros(count, dtype=torch.long)
        objects = torch.ones(count, dtype=torch.long)
        if prior is None:
            prior = torch.full((2, count, 2), .8)
        return self.model.compute_interaction_loss(
            [boxes], [humans], [objects], logits, [prior], [self.target])

    def test_no_matched_hoi_keeps_negative_loss_and_finite_gradients(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [15., 15., 25., 25.]])
        logits = torch.zeros(1, 2, requires_grad=True)
        loss = self.loss(boxes, logits)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0)
        self.assertEqual(self.model._last_hoi_loss_stats['matched_positives'], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_no_candidate_pairs_has_differentiable_zero_loss(self):
        logits = torch.zeros(0, 2, requires_grad=True)
        loss = self.loss(torch.empty(0, 4), logits)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_no_allowed_actions_has_differentiable_zero_loss(self):
        logits = torch.zeros(1, 2, requires_grad=True)
        boxes = torch.tensor([[40., 40., 60., 60.], [60., 60., 80., 80.]])
        loss = self.loss(boxes, logits, torch.zeros(2, 1, 2))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_positive_batch_preserves_original_loss(self):
        boxes = torch.tensor([[40., 40., 60., 60.], [60., 60., 80., 80.]])
        logits = torch.tensor([[.3, -.2], [.2, -.1]], requires_grad=True)
        loss = self.loss(boxes, logits)
        combined_prior = torch.full_like(logits, .8).pow(2)
        labels = torch.tensor([[0., 1.], [0., 1.]])
        expected = binary_focal_loss_with_logits(
            torch.log(combined_prior / (1 + torch.exp(-logits) - combined_prior) + 1e-8),
            labels, reduction='sum', alpha=self.model.alpha, gamma=self.model.gamma,
        ) / 2
        torch.testing.assert_allclose(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_distributed_normalizer_uses_global_count(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [15., 15., 25., 25.]])
        logits = torch.zeros(1, 2, requires_grad=True)
        baseline = self.loss(boxes, logits)
        with patch('upt.dist.is_initialized', return_value=True), \
                patch('upt.dist.get_world_size', return_value=2), \
                patch('upt.dist.all_reduce', side_effect=lambda count: count.fill_(4)):
            loss = self.loss(boxes, logits)
        torch.testing.assert_allclose(loss, baseline / 2)

    def test_distributed_all_zero_positives_is_finite(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [15., 15., 25., 25.]])
        logits = torch.zeros(1, 2, requires_grad=True)
        with patch('upt.dist.is_initialized', return_value=True), \
                patch('upt.dist.get_world_size', return_value=2), \
                patch('upt.dist.all_reduce'):
            loss = self.loss(boxes, logits)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_nonfinite_loss_is_rejected_before_optimizer_update(self):
        for value in (float('inf'), float('nan')):
            engine = object.__new__(CustomisedDLE)
            engine._rank = 0
            optimizer = Mock()
            engine._state = SimpleNamespace(
                net=Mock(return_value={'interaction_loss': torch.tensor(value)}),
                inputs=([],), targets=[{'file_name': 'example.jpg'}],
                epoch=1, iteration=201, optimizer=optimizer,
            )
            with self.assertRaisesRegex(ValueError, 'non-finite.*example.jpg'):
                engine._on_each_iteration()
            optimizer.step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
