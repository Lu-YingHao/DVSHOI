import copy
import unittest

import torch
from torch import nn

from dvs_data.query import DVSPairActionQuery
from dvs_data.checkpoint import load_hoi_weights


class TestDVSQuery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DVSPairActionQuery(16, 8, 3, query_dim=16, num_heads=4,
                                       pair_chunk_size=2, spatial_grid=(2, 3)).eval()
        self.pairs = torch.randn(5, 16, requires_grad=True)
        self.sequence = torch.randn(4, 8, 3, 4, requires_grad=True)

    def test_all_time_bins_are_retained_and_features_are_pair_action_specific(self):
        self.assertEqual(self.model.prepare_memory(self.sequence).shape, (4 * 2 * 3, 1, 16))
        features = self.model.extract_features(self.pairs, self.sequence)
        self.assertEqual(features.shape, (5, 3, 16))
        self.assertFalse(torch.allclose(features[0, 0], features[0, 1]))
        self.assertFalse(torch.allclose(features[0, 0], features[1, 0]))

    def test_reversing_event_features_changes_retrieved_evidence(self):
        first = self.model.extract_features(self.pairs, self.sequence)
        reversed_time = self.model.extract_features(self.pairs, self.sequence.flip(0))
        self.assertFalse(torch.allclose(first, reversed_time, atol=1e-6))

    def test_temporal_coordinates_break_permutation_invariance(self):
        # Disable temporal position: global cross-attention is invariant to
        # swapping complete time blocks. Restore it: the same swap changes features.
        original = self.model.position_proj.weight[:, 0].detach().clone()
        with torch.no_grad():
            self.model.position_proj.weight[:, 0].zero_()
        first = self.model.extract_features(self.pairs, self.sequence)
        swapped = self.model.extract_features(self.pairs, self.sequence.flip(0))
        torch.testing.assert_allclose(first, swapped, atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            self.model.position_proj.weight[:, 0].copy_(original)
        first = self.model.extract_features(self.pairs, self.sequence)
        swapped = self.model.extract_features(self.pairs, self.sequence.flip(0))
        self.assertFalse(torch.allclose(first, swapped, atol=1e-6))

    def test_chunking_preserves_results_and_gradients(self):
        second = copy.deepcopy(self.model)
        second.pair_chunk_size = 100
        with torch.no_grad():
            self.model.score.weight.normal_(0, .1)
            second.score.weight.copy_(self.model.score.weight)
        first_pairs = self.pairs.detach().clone().requires_grad_()
        second_pairs = self.pairs.detach().clone().requires_grad_()
        first_sequence = self.sequence.detach().clone().requires_grad_()
        second_sequence = self.sequence.detach().clone().requires_grad_()
        first = self.model(first_pairs, first_sequence)
        other = second(second_pairs, second_sequence)
        torch.testing.assert_allclose(first, other, atol=1e-6, rtol=1e-5)
        first.square().sum().backward()
        other.square().sum().backward()
        torch.testing.assert_allclose(first_pairs.grad, second_pairs.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_allclose(first_sequence.grad, second_sequence.grad, atol=1e-6, rtol=1e-5)
        self.assertGreater(first_sequence.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.action_embeddings.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.position_proj.weight.grad[:, 0].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(first_sequence.grad).all().item())

    def test_zero_score_initialization_and_empty_pairs(self):
        logits = self.model(self.pairs, self.sequence)
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        logits.sum().backward()
        self.assertGreater(self.model.score.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.model(self.pairs[:0], self.sequence).shape, (0, 3))


class TestLegacyMigration(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        head = nn.Module()
        head.dvs_encoder = nn.Linear(8, 8)
        head.box_pair_predictor = nn.Linear(16, 3)
        head.dvs_query = DVSPairActionQuery(16, 8, 3, query_dim=16)
        self.model.interaction_head = head
        self.legacy = {key: torch.ones_like(value) for key, value in self.model.state_dict().items()
                       if not key.startswith('interaction_head.dvs_query.')}
        for name, tensor in dict(weight=torch.ones(8), bias=torch.zeros(8)).items():
            self.legacy['interaction_head.dvs_norm.' + name] = tensor
        self.legacy['interaction_head.dvs_adapter.weight'] = torch.ones(16, 8)
        self.legacy['interaction_head.dvs_adapter.bias'] = torch.zeros(16)

    def test_legacy_weights_require_explicit_new_run_initialization(self):
        with self.assertRaisesRegex(ValueError, 'init-legacy-dvs'):
            load_hoi_weights(self.model, self.legacy)
        removed = load_hoi_weights(self.model, self.legacy, init_legacy_dvs=True)
        self.assertEqual(len(removed), 4)
        self.assertTrue(torch.equal(self.model.interaction_head.dvs_encoder.weight, torch.ones(8, 8)))
        self.assertEqual(self.model.interaction_head.dvs_query.score.weight.abs().sum().item(), 0)

    def test_missing_shared_weights_are_not_silently_ignored(self):
        self.legacy.pop('interaction_head.dvs_encoder.weight')
        with self.assertRaisesRegex(ValueError, 'incompatible shared layers'):
            load_hoi_weights(self.model, self.legacy, init_legacy_dvs=True)

    def test_new_query_weights_load_strictly(self):
        state = copy.deepcopy(self.model.state_dict())
        self.assertEqual(load_hoi_weights(self.model, state), [])
        state.pop('interaction_head.dvs_query.score.weight')
        with self.assertRaises(RuntimeError):
            load_hoi_weights(self.model, state)


if __name__ == '__main__':
    unittest.main()
