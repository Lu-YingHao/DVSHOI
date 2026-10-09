import unittest
from unittest.mock import patch

import torch
from torch import nn

from interaction_head import InteractionHead


class TestDVSQueryFusion(unittest.TestCase):
    def test_zero_initialised_query_score_receives_gradient(self):
        torch.manual_seed(1)
        head = InteractionHead(
            nn.Linear(64, 3),
            hidden_state_size=32,
            representation_size=32,
            num_channels=64,
            num_classes=3,
            human_idx=0,
            object_class_to_target_class=[[0], [0, 1], [1, 2]],
            use_dvs=True,
            dvs_variant='tiny',
            dvs_adjacent_changes=True,
            dvs_precomp_residual=True,
            dvs_relation_mode='global',
        )
        features = torch.rand(1, 64, 4, 4)
        image_shapes = torch.tensor([[64, 64]])
        region_props = [dict(
            boxes=torch.tensor([
                [1., 1., 20., 40.],
                [22., 8., 40., 35.],
                [10., 42., 30., 60.],
            ]),
            scores=torch.tensor([.9, .8, .7]),
            labels=torch.tensor([0, 1, 2]),
            hidden_states=torch.rand(3, 32),
        )]
        dvs_frames = (torch.rand(1, 4, 2, 32, 48) > .8).float()

        self.assertEqual(head.dvs_query.score.weight.abs().sum().item(), 0)
        logits = head(
            features, image_shapes, region_props, dvs_frames
        )[0]
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all().item())

        logits.sum().backward()
        gradient = head.dvs_query.score.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertGreater(head.dvs_relation_query.residual.weight.grad.abs().sum().item(), 0)

        # Once the score head has moved away from zero, task gradients must
        # reach the query parameters and the real spiking event encoder.
        with torch.no_grad():
            head.dvs_query.score.weight.normal_(0, .1)
            head.dvs_relation_query.residual.weight.normal_(0, .01)
        head.zero_grad()
        logits = head(features, image_shapes, region_props, dvs_frames)[0]
        logits.square().sum().backward()
        encoder_grad = head.dvs_encoder.patch_embed.stages[0].conv.weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all().item())
        self.assertGreater(encoder_grad.abs().sum().item(), 0)
        self.assertGreater(head.dvs_query.action_embeddings.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.dvs_query.change_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.dvs_relation_query.pair_proj.weight.grad.abs().sum().item(), 0)

    def test_initial_logits_equal_rgb_path_and_empty_image_keeps_alignment(self):
        torch.manual_seed(2)
        head = InteractionHead(nn.Linear(64, 3), 32, 32, 64, 3, 0,
                               [[0], [0, 1], [1, 2]], use_dvs=True, dvs_variant='tiny',
                               dvs_adjacent_changes=True, dvs_precomp_residual=True,
                               dvs_relation_mode='global').eval()
        features = torch.rand(2, 64, 4, 4)
        sizes = torch.tensor([[64, 64], [64, 64]])
        props = [dict(boxes=torch.tensor([[1., 1., 20., 40.], [22., 8., 40., 35.]]),
                      scores=torch.tensor([.9, .8]), labels=torch.tensor([1, 2]),
                      hidden_states=torch.rand(2, 32)),
                 dict(boxes=torch.tensor([[1., 1., 20., 40.], [22., 8., 40., 35.]]),
                      scores=torch.tensor([.9, .8]), labels=torch.tensor([0, 1]),
                      hidden_states=torch.rand(2, 32))]
        frames = torch.rand(2, 3, 2, 32, 48)
        with torch.no_grad():
            query = head(features, sizes, props, frames)
            head.use_dvs = False
            rgb = head(features, sizes, props)
        torch.testing.assert_allclose(query[0], rgb[0])
        self.assertEqual(query[0].shape, (1, 3))
        self.assertEqual(len(query[-1]), 2)
        self.assertEqual(len(query[2][0]), 0)

    def test_all_empty_pairs_support_backward(self):
        head = InteractionHead(nn.Linear(64, 3), 32, 32, 64, 3, 0,
                               [[0], [0, 1], [1, 2]], use_dvs=True, dvs_variant='tiny',
                               dvs_adjacent_changes=True, dvs_precomp_residual=True,
                               dvs_relation_mode='global').eval()
        props = [dict(boxes=torch.empty(0, 4), scores=torch.empty(0),
                      labels=torch.empty(0, dtype=torch.long), hidden_states=torch.empty(0, 32))]
        logits = head(torch.rand(1, 64, 4, 4), torch.tensor([[64, 64]]),
                      props, torch.rand(1, 2, 2, 32, 48))[0]
        self.assertEqual(logits.shape, (0, 3))
        logits.sum().backward()
        self.assertIsNotNone(head.box_pair_predictor.weight.grad)

    def test_event_residual_reaches_competitive_input_and_memory_is_reused(self):
        torch.manual_seed(3)
        head = InteractionHead(nn.Linear(64, 3), 32, 32, 64, 3, 0,
                               [[0], [0, 1], [1, 2]], use_dvs=True, dvs_variant='tiny',
                               dvs_adjacent_changes=True, dvs_precomp_residual=True,
                               dvs_relation_mode='global').eval()
        features = torch.rand(1, 64, 4, 4)
        sizes = torch.tensor([[64, 64]])
        props = [dict(boxes=torch.tensor([[1., 1., 20., 40.], [22., 8., 40., 35.]]),
                      scores=torch.tensor([.9, .8]), labels=torch.tensor([0, 1]),
                      hidden_states=torch.rand(2, 32))]
        comp_inputs, memories, projections = [], [], []
        hooks = [head.comp_layer.register_forward_pre_hook(
                    lambda module, inputs: comp_inputs.append(inputs[0].detach().clone())),
                 head.dvs_query.change_proj.register_forward_hook(
                    lambda module, inputs, output: projections.append(output.shape)),
                 head.dvs_query.attention.register_forward_pre_hook(
                    lambda module, inputs: memories.append(inputs[1])),
                 head.dvs_relation_query.attention.register_forward_pre_hook(
                    lambda module, inputs: memories.append(inputs[1]))]
        with torch.no_grad():
            head.dvs_relation_query.residual.bias.copy_(torch.linspace(-.2, .2, 64))
            event_logits = head(features, sizes, props, torch.rand(1, 4, 2, 32, 48))[0]
            head.use_dvs = False
            rgb_logits = head(features, sizes, props)[0]
        for hook in hooks:
            hook.remove()
        expected = head.dvs_relation_query.residual.bias.unsqueeze(0)
        torch.testing.assert_allclose(comp_inputs[0] - comp_inputs[1], expected, atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(event_logits, rgb_logits))
        self.assertEqual(len(projections), 1)
        self.assertEqual(len(memories), 2)
        self.assertIs(memories[0], memories[1])
        # tiny encoder produces a 2x3 grid: 4 original + 3 interval bins.
        self.assertEqual(memories[0].shape[0], 7 * 2 * 3)


class TestDVSEntityFusion(unittest.TestCase):
    def make_head(self):
        torch.manual_seed(18)
        return InteractionHead(nn.Linear(64, 3), 32, 32, 64, 3, 0,
                               [[0], [0, 1], [1, 2]], use_dvs=True, dvs_variant='tiny',
                               dvs_adjacent_changes=True, dvs_precomp_residual=True).eval()

    def inputs(self):
        # RGB 64x80 and DVS 48x32 are deliberately different resolutions.
        props = [dict(boxes=torch.tensor([[22., 8., 40., 35.], [1., 1., 20., 40.],
                                         [10., 42., 30., 60.], [42., 2., 60., 40.]]),
                      scores=torch.tensor([.8, .9, .7, .8]), labels=torch.tensor([1, 0, 2, 0]),
                      hidden_states=torch.rand(4, 32))]
        return torch.rand(1, 64, 4, 5), torch.tensor([[64, 80]]), props, torch.rand(1, 4, 2, 48, 32)

    def test_zero_init_rgb_equivalence_and_unique_entities_with_correct_roles(self):
        head = self.make_head()
        inputs = self.inputs()
        calls = []
        hook = head.dvs_entity_relation.register_forward_pre_hook(
            lambda module, values: calls.append(values))
        with patch.object(head.dvs_entity_relation, 'extract_entity_sequence',
                          wraps=head.dvs_entity_relation.extract_entity_sequence) as extract:
            logits = head(*inputs)[0]
            self.assertEqual(extract.call_count, 1)
        hook.remove()
        self.assertEqual(logits.shape, (6, 3))
        self.assertEqual(calls[0][1].shape, (4, 32))  # 4 entities, not 12 pair endpoints
        self.assertTrue(torch.equal(calls[0][2], torch.tensor([True, True, False, False])))
        self.assertEqual(calls[0][5].shape, (4, 6, 128))  # each original bin retained
        head.use_dvs = False
        rgb = head(*inputs[:3])[0]
        torch.testing.assert_allclose(logits, rgb, atol=0, rtol=0)
        head.use_dvs = True
        logits.sum().backward()
        self.assertGreater(head.dvs_entity_relation.residual.weight.grad.abs().sum().item(), 0)

    def test_nonzero_entity_residual_reaches_competitive_layer_and_encoder(self):
        head = self.make_head()
        inputs = self.inputs()
        residuals, comp_inputs = [], []
        hooks = [head.dvs_entity_relation.register_forward_hook(
                    lambda module, values, out: residuals.append(out)),
                 head.comp_layer.register_forward_pre_hook(
                    lambda module, values: comp_inputs.append(values[0]))]
        with torch.no_grad():
            head.dvs_entity_relation.residual.weight.normal_(0, .01)
        logits = head(*inputs)[0]
        logits.square().sum().backward()
        self.assertGreater(head.dvs_entity_relation.entity_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.dvs_entity_relation.update.weight_hh.grad.abs().sum().item(), 0)
        self.assertGreater(head.dvs_encoder.patch_embed.stages[0].conv.weight.grad.abs().sum().item(), 0)
        head.use_dvs = False
        head(*inputs[:3])
        torch.testing.assert_allclose(comp_inputs[0] - comp_inputs[1], residuals[0], atol=1e-6, rtol=1e-5)
        for hook in hooks:
            hook.remove()

    def test_empty_candidates(self):
        head = self.make_head()
        props = [dict(boxes=torch.empty(0, 4), scores=torch.empty(0),
                      labels=torch.empty(0, dtype=torch.long), hidden_states=torch.empty(0, 32))]
        output = head(torch.rand(1, 64, 4, 4), torch.tensor([[64, 64]]), props,
                      torch.rand(1, 4, 2, 48, 32))[0]
        self.assertEqual(output.shape, (0, 3))
        output.sum().backward()
        self.assertIsNotNone(head.box_pair_predictor.weight.grad)


if __name__ == '__main__':
    unittest.main()
