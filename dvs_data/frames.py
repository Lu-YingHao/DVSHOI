"""Convert a raw DVS event stream into polarity-separated time bins."""

import numpy as np
import torch


def events_to_frames(events, num_bins=8, count_clip=1):
    """Return float32 frames with shape [T, 2, H, W] in DVS coordinates.

    Bins cover equal durations between the first and last event. Channel 0
    contains negative events and channel 1 contains positive events. Counts
    are clipped and scaled to [0, 1]; count_clip=1 gives binary occupancy.
    """
    if num_bins <= 0 or count_clip <= 0:
        raise ValueError("num_bins and count_clip must be positive")

    height, width = events['sensor_size']
    t = np.asarray(events['t'], dtype=np.int64)
    x = np.asarray(events['x'], dtype=np.int64)
    y = np.asarray(events['y'], dtype=np.int64)
    p = np.asarray(events['p'], dtype=np.int64)

    if not len(t) or not (len(t) == len(x) == len(y) == len(p)):
        raise ValueError("DVS event arrays must be nonempty and have equal lengths")
    if np.any(t[1:] < t[:-1]):
        raise ValueError("DVS timestamps must be sorted")
    if (np.any(x < 0) or np.any(x >= width) or
            np.any(y < 0) or np.any(y >= height) or
            np.any((p != 0) & (p != 1))):
        raise ValueError("DVS coordinates or polarity are out of range")

    duration = int(t[-1] - t[0]) + 1
    time_bins = (t - t[0]) * num_bins // duration
    flat_index = ((time_bins * 2 + p) * height + y) * width + x
    frames = np.bincount(
        flat_index, minlength=num_bins * 2 * height * width
    ).reshape(num_bins, 2, height, width)
    np.minimum(frames, count_clip, out=frames)
    return torch.from_numpy(frames.astype(np.float32) / count_clip)
