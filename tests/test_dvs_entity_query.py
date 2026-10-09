import copy
import unittest

import torch
from torch import nn

from dvs_data.entity_query import DVSTemporalEntityRelation
from dvs_data.query import DVSPairActionQuery, DVSPairRelationQuery
from dvs_data.checkpoint import load_hoi_weights, validate_query_layout


class TestEntityReadout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.model = DVSTemporalEntityRelation(16, 32, 16, 4, 2).eval()
        self.entities = torch.randn(4, 16)
        self.roles = torch.tensor([True, False, False, True])
        self.memory = torch.randn(5, 6, 16, requires_grad=True)
        self.pairs = torch.randn(5, 32)
        self.human = torch.tensor([0, 0, 0, 3, 3])
        self.obj = torch.tensor([1, 2, 3, 1, 2])

    def test_entity_and_pair_permutations_preserve_identity(self):
        original = self.model.extract_entity_sequence(self.entities, self.roles, self.memory)
        perm = torch.tensor([2, 0, 3, 1])
        other = self.model.extract_entity_sequence(self.entities[perm], self.roles[perm], self.memory)
        torch.testing.assert_allclose(other, original[perm], atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            self.model.residual.weight.normal_(0, .1)
        result = self.model.read_pairs(self.pairs, original, self.human, self.obj)
        order = torch.tensor([4, 2, 0, 3, 1])
        reordered = self.model.read_pairs(self.pairs[order], original, self.human[order], self.obj[order])
        torch.testing.assert_allclose(reordered, result[order], atol=1e-6, rtol=1e-5)
        # No pair-level exclusion: all incident pairs share entity zero's
        # exact same event sequence, including a human-human candidate.
        self.assertEqual(original.shape, (4, 5, 16))
        self.assertFalse(torch.allclose(original[0], original[1]))

    def test_content_follows_moving_tokens_without_rgb_coordinates(self):
        model = DVSTemporalEntityRelation(8, 16, 8, 1, 2).eval()
        with torch.no_grad():
            for layer in [model.entity_proj, model.key, model.value, model.slot_query]:
                layer.weight.copy_(torch.eye(8))
                if layer.bias is not None:
                    layer.bias.zero_()
            model.role_embeddings.weight.zero_()
            model.background.zero_()
            for parameter in model.update.parameters():
                parameter.zero_()
        a = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])
        b = torch.tensor([1., 1., -1., -1., 1., 1., -1., -1.])
        entities = torch.stack([a, b])
        stationary = torch.stack([a, b, -a, -b]).unsqueeze(0).repeat(4, 1, 1)
        permutations = [torch.tensor([0, 1, 2, 3]), torch.tensor([2, 0, 3, 1]),
                        torch.tensor([3, 2, 1, 0]), torch.tensor([1, 3, 0, 2])]
        moving = torch.stack([stationary[t, p] for t, p in enumerate(permutations)])
        sequence, maps = model.extract_entity_sequence(entities, torch.tensor([True, False]),
                                                       stationary, return_attention=True)
        moved, moved_maps = model.extract_entity_sequence(entities, torch.tensor([True, False]),
                                                         moving, return_attention=True)
        torch.testing.assert_allclose(sequence, moved, atol=1e-6, rtol=1e-5)
        for t, perm in enumerate(permutations):
            torch.testing.assert_allclose(moved_maps[:, t], maps[:, t, perm], atol=1e-6, rtol=1e-5)
            self.assertEqual(moved_maps[0, t].argmax().item(), (perm == 0).nonzero().item())
            self.assertEqual(moved_maps[1, t].argmax().item(), (perm == 1).nonzero().item())
        # A grid-internal center moves with the content's location; identity
        # evidence is invariant to location while geometry retains movement.
        coordinates = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
        centers = moved_maps @ coordinates
        self.assertFalse(torch.allclose(centers[:, 0], centers[:, 1]))
        self.assertTrue((centers.abs() <= 1).all().item())

    def test_assignments_are_bounded_competitive_and_scale_invariant(self):
        queries = torch.randn(5, 16)
        _, weights, responsibilities = self.model.assign(queries, self.memory[0])
        torch.testing.assert_allclose(responsibilities.sum(0), torch.ones(6))
        torch.testing.assert_allclose(weights.sum(-1), torch.ones(5))
        with torch.no_grad():
            self.model.slot_query.weight.mul_(1000)
            self.model.key.weight.mul_(1000)
        _, scaled_weights, scaled_responsibilities = self.model.assign(queries, self.memory[0])
        torch.testing.assert_allclose(weights, scaled_weights, atol=1e-6, rtol=1e-5)
        torch.testing.assert_allclose(responsibilities, scaled_responsibilities, atol=1e-6, rtol=1e-5)

    def test_event_grid_geometry_is_retained_without_affecting_content_assignment(self):
        content, maps = self.model.extract_entity_sequence(self.entities, self.roles, self.memory,
                                                           return_attention=True)
        positioned, positioned_maps = self.model.extract_entity_sequence(
            self.entities, self.roles, self.memory, return_attention=True, spatial_grid=(2, 3))
        torch.testing.assert_allclose(maps, positioned_maps, atol=0, rtol=0)
        self.assertFalse(torch.allclose(content, positioned))
        positioned.square().sum().backward()
        self.assertGreater(self.model.geometry_proj.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            self.model.extract_entity_sequence(self.entities, self.roles, self.memory, spatial_grid=(3, 3))
        single = self.memory[:, :1]
        no_geometry = self.model.extract_entity_sequence(self.entities, self.roles, single)
        centered_geometry = self.model.extract_entity_sequence(self.entities, self.roles, single,
                                                               spatial_grid=(1, 1))
        torch.testing.assert_allclose(no_geometry, centered_geometry, atol=0, rtol=0)

    def test_chunk_parity_gradients_and_temporal_state(self):
        # Use double precision for the equality check: float32 accumulates
        # recurrent gradients in a different order when pair chunks change.
        self.model.double()
        pairs, entities = self.pairs.double(), self.entities.double()
        with torch.no_grad():
            self.model.residual.weight.normal_(0, .1)
        full = copy.deepcopy(self.model)
        full.pair_chunk_size = 100
        a_memory = self.memory.detach().double().clone().requires_grad_()
        b_memory = self.memory.detach().double().clone().requires_grad_()
        a = self.model(pairs, entities, self.roles, self.human, self.obj, a_memory)
        b = full(pairs, entities, self.roles, self.human, self.obj, b_memory)
        torch.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-8)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_allclose(a_memory.grad, b_memory.grad, atol=1e-10, rtol=1e-8)
        for parameter in [self.model.entity_proj.weight, self.model.key.weight,
                          self.model.update.weight_hh, self.model.pair_event_proj.weight]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for t in range(len(a_memory)):
            self.assertGreater(a_memory.grad[t].abs().sum().item(), 0)
        forward = self.model.extract_entity_sequence(entities, self.roles, a_memory)
        reverse = self.model.extract_entity_sequence(entities, self.roles, a_memory.flip(0)).flip(1)
        self.assertFalse(torch.allclose(forward, reverse, atol=1e-6))

    def test_empty_and_single_bin_and_zero_output(self):
        output = self.model(self.pairs, self.entities, self.roles, self.human, self.obj, self.memory[:1])
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        output.sum().backward()
        self.assertGreater(self.model.residual.weight.grad.abs().sum().item(), 0)
        empty = self.model.extract_entity_sequence(self.entities[:0], self.roles[:0], self.memory)
        self.assertEqual(empty.shape, (0, 5, 16))
        output = self.model(self.pairs[:0], self.entities[:0], self.roles[:0], self.human[:0],
                            self.obj[:0], self.memory)
        self.assertEqual(output.shape, (0, 32))

    def test_readout_is_event_evidence_not_copied_rgb_anchor(self):
        # If all spatial values are identical, content queries cannot invent
        # different entity evidence by copying their different RGB anchors.
        memory = torch.randn(5, 1, 16).expand(-1, 6, -1)
        sequence = self.model.extract_entity_sequence(self.entities, self.roles, memory)
        for entity in range(1, len(self.entities)):
            torch.testing.assert_allclose(sequence[0], sequence[entity], atol=1e-6, rtol=1e-5)


class TestEntityCheckpoint(unittest.TestCase):
    def model(self, mode=None):
        model = nn.Module()
        head = nn.Module()
        head.encoder = nn.Linear(8, 8)
        head.dvs_query = DVSPairActionQuery(16, 8, 3, query_dim=16)
        if mode == 'global':
            head.dvs_relation_query = DVSPairRelationQuery(16, 16)
        elif mode == 'entity-slots':
            head.dvs_entity_relation = DVSTemporalEntityRelation(8, 16, 16)
        model.interaction_head = head
        return model

    def test_modes_must_match_and_baseline_init_is_explicit(self):
        global_model = self.model('global')
        slots = self.model('entity-slots')
        with self.assertRaisesRegex(ValueError, 'relation mode mismatch'):
            load_hoi_weights(slots, global_model.state_dict())
        with self.assertRaisesRegex(ValueError, 'relation mode mismatch'):
            load_hoi_weights(global_model, slots.state_dict())
        with self.assertRaises(ValueError):
            validate_query_layout(global_model.state_dict(), False, True, relation_mode='entity-slots')
        load_hoi_weights(slots, self.model().state_dict(), init_query_baseline=True)
        self.assertEqual(load_hoi_weights(slots, slots.state_dict()), [])
        broken = dict(slots.state_dict())
        broken.pop('interaction_head.dvs_entity_relation.key.weight')
        with self.assertRaises(RuntimeError):
            load_hoi_weights(slots, broken)


if __name__ == '__main__':
    unittest.main()
