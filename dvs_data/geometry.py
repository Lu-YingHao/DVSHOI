from typing import Union

import torch


TensorLike = Union[torch.Tensor, int, float]


def identity() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float32)


def compose_transform(matrix: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    matrix = _as_matrix(matrix)
    previous = _as_matrix(previous).to(dtype=matrix.dtype, device=matrix.device)
    return matrix @ previous


def crop_matrix(left: TensorLike, top: TensorLike) -> torch.Tensor:
    matrix = identity()
    matrix[0, 2] = -float(left)
    matrix[1, 2] = -float(top)
    return matrix


def hflip_matrix(width: TensorLike) -> torch.Tensor:
    matrix = identity()
    matrix[0, 0] = -1.0
    matrix[0, 2] = float(width)
    return matrix


def resize_matrix(ratio_width: TensorLike, ratio_height: TensorLike) -> torch.Tensor:
    matrix = identity()
    matrix[0, 0] = float(ratio_width)
    matrix[1, 1] = float(ratio_height)
    return matrix


def _as_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 transform matrix, got {tuple(matrix.shape)}")
    return matrix
