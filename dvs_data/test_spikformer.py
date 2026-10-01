import unittest

import numpy as np
import torch

from dvs_data import DVSSpikformer, events_to_frames


class TestDVSFrames(unittest.TestCase):
    def test_time_bins_and_polarity(self):
        events = {
            't': np.array([0, 1, 2, 3]),
            'x': np.array([0, 0, 1, 1]),
            'y': np.array([0, 0, 0, 0]),
            'p': np.array([0, 1, 0, 1]),
            'sensor_size': (2, 2),
        }
        frames = events_to_frames(events, num_bins=2)
        self.assertEqual(tuple(frames.shape), (2, 2, 2, 2))
        self.assertEqual(frames.sum().item(), 4)
        self.assertEqual(frames[0, 0, 0, 0].item(), 1)
        self.assertEqual(frames[0, 1, 0, 0].item(), 1)
        self.assertEqual(frames[1, 0, 0, 1].item(), 1)
        self.assertEqual(frames[1, 1, 0, 1].item(), 1)

    def test_same_timestamp_and_clipping(self):
        events = {
            't': np.array([5, 5, 5]),
            'x': np.array([0, 0, 0]),
            'y': np.array([0, 0, 0]),
            'p': np.array([1, 1, 1]),
            'sensor_size': (1, 1),
        }
        frames = events_to_frames(events, num_bins=3, count_clip=2)
        self.assertEqual(frames[0, 1, 0, 0].item(), 1)
        self.assertEqual(frames[1:].sum().item(), 0)


class TestDVSSpikformer(unittest.TestCase):
    def test_output_gradient_and_state_reset(self):
        torch.manual_seed(0)
        model = DVSSpikformer(
            variant='tiny', embed_dim=32, depth=1, num_heads=4,
            max_grid=(2, 3)
        )
        frames = (torch.rand(2, 4, 2, 32, 48) > 0.5).float()
        features = model(frames)
        self.assertEqual(tuple(features.shape), (2, 4, 32))
        self.assertTrue(torch.isfinite(features).all().item())
        features.sum().backward()
        first_conv = model.patch_embed.stages[0].conv
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertGreater(first_conv.weight.grad.abs().sum().item(), 0)

        model.eval()
        with torch.no_grad():
            first = model(frames)
            model(torch.zeros_like(frames))
            second = model(frames)
        self.assertTrue(torch.equal(first, second))

    def test_rejects_wrong_input_shape(self):
        model = DVSSpikformer(
            variant='tiny', embed_dim=32, depth=1, max_grid=(2, 2)
        )
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 32, 32))

    def test_base_variant_has_full_stack(self):
        model = DVSSpikformer()
        self.assertEqual(model.variant, 'base')
        self.assertEqual(model.embed_dim, 256)
        self.assertEqual(model.depth, 6)
        self.assertEqual(len(model.patch_embed.stages), 4)
        self.assertEqual(len(model.blocks), 6)

        model.eval()
        with torch.no_grad():
            features = model(torch.zeros(1, 2, 2, 32, 48))
        self.assertEqual(tuple(features.shape), (1, 2, 256))


if __name__ == '__main__':
    unittest.main()
