import os
from typing import Dict, Optional, Tuple

import numpy as np


def load_event_npz(
    npz_path: str,
    sensor_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Load a DVS event file stored in NPZ format.

    Expected keys:
        t: timestamp, shape [N]
        x: x coordinate, shape [N]
        y: y coordinate, shape [N]
        p: polarity, shape [N], values in {0, 1}

    Args:
        npz_path:
            Path to the .npz file.

        sensor_size:
            Optional (H, W) of the DVS sensor.

            It is recommended to explicitly provide the real
            DVS resolution rather than infer it from max(x/y).

    Returns:
        events: dict
            {
                "t": np.ndarray [N], int64,
                "x": np.ndarray [N], int64,
                "y": np.ndarray [N], int64,
                "p": np.ndarray [N], int64,
                "sensor_size": (H, W),
                "num_events": int,
            }
    """

    # --------------------------------------------------
    # 1. Check file
    # --------------------------------------------------
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"DVS file does not exist: {npz_path}"
        )

    # --------------------------------------------------
    # 2. Load NPZ
    # --------------------------------------------------
    with np.load(npz_path, allow_pickle=False) as data:

        required_keys = {"t", "x", "y", "p"}

        missing_keys = required_keys - set(data.files)

        if missing_keys:
            raise KeyError(
                f"{npz_path} is missing keys: {missing_keys}. "
                f"Available keys: {data.files}"
            )

        t = np.asarray(data["t"]).reshape(-1)
        x = np.asarray(data["x"]).reshape(-1)
        y = np.asarray(data["y"]).reshape(-1)
        p = np.asarray(data["p"]).reshape(-1)

    # --------------------------------------------------
    # 3. Check event number
    # --------------------------------------------------
    num_events = len(t)

    if not (
        len(x) == num_events
        and len(y) == num_events
        and len(p) == num_events
    ):
        raise ValueError(
            f"Inconsistent event lengths in {npz_path}: "
            f"t={len(t)}, x={len(x)}, "
            f"y={len(y)}, p={len(p)}"
        )

    if num_events == 0:
        raise ValueError(
            f"Empty event stream: {npz_path}"
        )

    # --------------------------------------------------
    # 4. Use unified dtype
    # --------------------------------------------------
    t = t.astype(np.int64, copy=False)
    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)
    p = p.astype(np.int64, copy=False)

    # --------------------------------------------------
    # 5. Validate timestamp
    # --------------------------------------------------
    if not np.all(np.isfinite(t)):
        raise ValueError(
            f"Invalid timestamp in {npz_path}"
        )

    if np.any(t[1:] < t[:-1]):
        raise ValueError(
            f"Timestamps are not sorted in {npz_path}"
        )

    # --------------------------------------------------
    # 6. Validate polarity
    # --------------------------------------------------
    unique_p = np.unique(p)

    if not np.all(np.isin(unique_p, [0, 1])):
        raise ValueError(
            f"Unsupported polarity values in {npz_path}: "
            f"{unique_p}. Expected {{0, 1}}."
        )

    # --------------------------------------------------
    # 7. Determine sensor size
    # --------------------------------------------------
    if sensor_size is None:
        # Fallback only.
        # This is NOT necessarily the physical sensor resolution.
        height = int(y.max()) + 1
        width = int(x.max()) + 1

    else:
        height, width = sensor_size

        height = int(height)
        width = int(width)

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid sensor size: {sensor_size}"
            )

    # --------------------------------------------------
    # 8. Validate coordinates
    # --------------------------------------------------
    if x.min() < 0 or x.max() >= width:
        raise ValueError(
            f"x coordinate out of range in {npz_path}: "
            f"x=[{x.min()}, {x.max()}], W={width}"
        )

    if y.min() < 0 or y.max() >= height:
        raise ValueError(
            f"y coordinate out of range in {npz_path}: "
            f"y=[{y.min()}, {y.max()}], H={height}"
        )

    # --------------------------------------------------
    # 9. Return unified event representation
    # --------------------------------------------------
    events = {
        "t": t,
        "x": x,
        "y": y,
        "p": p,

        "sensor_size": (
            height,
            width
        ),

        "num_events": num_events,
    }

    return events