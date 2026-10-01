import unittest

import torch
from torch import nn

from interaction_head import InteractionHead


class TestDVSResidualFusion(unittest.TestCase):
    def test_zero_initialised_adapter_receives_gradient(self):
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

        self.assertEqual(head.dvs_adapter.weight.abs().sum().item(), 0)
        logits = head(
            features, image_shapes, region_props, dvs_frames
        )[0]
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all().item())

        logits.sum().backward()
        gradient = head.dvs_adapter.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
