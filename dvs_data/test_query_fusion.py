import unittest

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

        # Once the score head has moved away from zero, task gradients must
        # reach the query parameters and the real spiking event encoder.
        with torch.no_grad():
            head.dvs_query.score.weight.normal_(0, .1)
        head.zero_grad()
        logits = head(features, image_shapes, region_props, dvs_frames)[0]
        logits.square().sum().backward()
        encoder_grad = head.dvs_encoder.patch_embed.stages[0].conv.weight.grad
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all().item())
        self.assertGreater(encoder_grad.abs().sum().item(), 0)
        self.assertGreater(head.dvs_query.action_embeddings.weight.grad.abs().sum().item(), 0)

    def test_initial_logits_equal_rgb_path_and_empty_image_keeps_alignment(self):
        torch.manual_seed(2)
        head = InteractionHead(nn.Linear(64, 3), 32, 32, 64, 3, 0,
                               [[0], [0, 1], [1, 2]], use_dvs=True, dvs_variant='tiny').eval()
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
                               [[0], [0, 1], [1, 2]], use_dvs=True, dvs_variant='tiny').eval()
        props = [dict(boxes=torch.empty(0, 4), scores=torch.empty(0),
                      labels=torch.empty(0, dtype=torch.long), hidden_states=torch.empty(0, 32))]
        logits = head(torch.rand(1, 64, 4, 4), torch.tensor([[64, 64]]),
                      props, torch.rand(1, 2, 2, 32, 48))[0]
        self.assertEqual(logits.shape, (0, 3))
        logits.sum().backward()
        self.assertIsNotNone(head.box_pair_predictor.weight.grad)


if __name__ == '__main__':
    unittest.main()
