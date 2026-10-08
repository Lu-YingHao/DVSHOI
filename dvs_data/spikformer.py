"""PyTorch 1.8 compatible Spikformer backbone for DVS time bins.

The implementation follows the full SPS -> SSA/MLP stack used by Spikformer,
but keeps the temporal features instead of attaching an image classifier. It
is self-contained and does not require SpikingJelly or CuPy.
"""

import torch
from torch import nn
from torch.nn import functional as F


_VARIANTS = {
    'tiny': dict(
        embed_dim=128, depth=2, num_heads=4, mlp_ratio=2.0,
        max_grid=(8, 11), drop_path_rate=0.0,
    ),
    'base': dict(
        embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
        max_grid=(17, 22), drop_path_rate=0.1,
    ),
}


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (voltage,) = ctx.saved_tensors
        return grad_output / (1 + voltage.abs()).pow(2)


class _MultiStepLIF(nn.Module):
    """Multi-step LIF with a differentiable surrogate spike gradient."""

    def __init__(self, tau=2.0, threshold=0.5):
        super().__init__()
        self.tau = float(tau)
        self.threshold = float(threshold)

    def forward(self, sequence):
        voltage = torch.zeros_like(sequence[0])
        spikes = []
        for current in sequence.unbind(0):
            voltage = voltage + (current - voltage) / self.tau
            spike = _SurrogateSpike.apply(voltage - self.threshold)
            voltage = voltage - spike.detach() * self.threshold
            spikes.append(spike)
        return torch.stack(spikes)


class _DropPath(nn.Module):
    """Stochastic depth shared across all time steps of one sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * mask.floor() / keep_prob


class _SpikingConvStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = _MultiStepLIF()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, sequence):
        steps, batch = sequence.shape[:2]
        x = self.bn(self.conv(sequence.flatten(0, 1)))
        x = self.lif(x.reshape(steps, batch, *x.shape[1:]))
        x = self.pool(x.flatten(0, 1))
        return x.reshape(steps, batch, *x.shape[1:])


class _SpikingPatchStem(nn.Module):
    """Full four-stage Spiking Patch Splitting with position encoding."""

    def __init__(self, in_channels, embed_dim):
        super().__init__()
        channels = [in_channels, embed_dim // 8, embed_dim // 4,
                    embed_dim // 2, embed_dim]
        self.stages = nn.ModuleList([
            _SpikingConvStage(channels[i], channels[i + 1])
            for i in range(4)
        ])
        self.rpe_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = _MultiStepLIF()

    def forward(self, sequence):
        for stage in self.stages:
            sequence = stage(sequence)

        steps, batch = sequence.shape[:2]
        rpe = self.rpe_bn(self.rpe_conv(sequence.flatten(0, 1)))
        rpe = rpe.reshape(steps, batch, *rpe.shape[1:])
        return sequence + self.rpe_lif(rpe)


class _SpikingLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = _MultiStepLIF()

    def forward(self, sequence):
        steps, batch = sequence.shape[:2]
        x = self.bn(self.conv(sequence.flatten(0, 1)))
        return self.lif(x.reshape(steps, batch, *x.shape[1:]))


class _SpikingSelfAttention(nn.Module):
    """Spike-form multi-head self-attention without softmax."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = _SpikingLinear(dim, dim)
        self.k = _SpikingLinear(dim, dim)
        self.v = _SpikingLinear(dim, dim)
        self.attn_lif = _MultiStepLIF()
        self.proj = _SpikingLinear(dim, dim)

    def forward(self, sequence):
        steps, batch, channels, tokens = sequence.shape
        head_dim = channels // self.num_heads

        def split_heads(x):
            return x.reshape(
                steps, batch, self.num_heads, head_dim, tokens
            )

        q = split_heads(self.q(sequence))
        k = split_heads(self.k(sequence))
        v = split_heads(self.v(sequence))
        # This multiplication order is algebraically equivalent to
        # (Q K^T) V in token-first notation, without materialising N x N.
        attention = v @ k.transpose(-1, -2)
        x = (attention @ q) * 0.25
        x = x.reshape(steps, batch, channels, tokens)
        return self.proj(self.attn_lif(x))


class _SpikingMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = _SpikingLinear(dim, hidden_dim)
        self.fc2 = _SpikingLinear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _SpikformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path):
        super().__init__()
        self.attn = _SpikingSelfAttention(dim, num_heads)
        self.mlp = _SpikingMLP(dim, int(dim * mlp_ratio))
        self.drop_path = _DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        return x + self.drop_path(self.mlp(x))


class DVSSpikformer(nn.Module):
    """Encode ``[B, T, 2, H, W]`` as ``[B, T, D, H', W']`` features.

    ``variant='base'`` is the full six-block model. ``variant='tiny'`` keeps
    the earlier lightweight configuration for ablation experiments. Explicit
    keyword arguments override values from the selected preset.
    """

    def __init__(self, variant='base', embed_dim=None, depth=None,
                 num_heads=None, mlp_ratio=None, max_grid=None,
                 drop_path_rate=None):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError("Unknown Spikformer variant: {}".format(variant))

        config = dict(_VARIANTS[variant])
        overrides = dict(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, max_grid=max_grid,
            drop_path_rate=drop_path_rate,
        )
        config.update({key: value for key, value in overrides.items()
                       if value is not None})

        embed_dim = config['embed_dim']
        depth = config['depth']
        num_heads = config['num_heads']
        mlp_ratio = config['mlp_ratio']
        max_grid = config['max_grid']
        drop_path_rate = config['drop_path_rate']
        if embed_dim < 32 or embed_dim % 8 or embed_dim % num_heads:
            raise ValueError(
                "embed_dim must be >= 32 and divisible by 8 and num_heads"
            )
        if depth < 1 or num_heads < 1 or mlp_ratio <= 0:
            raise ValueError("depth, num_heads and mlp_ratio must be positive")
        if len(max_grid) != 2 or min(max_grid) < 1:
            raise ValueError("max_grid must contain two positive integers")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")

        self.patch_embed = _SpikingPatchStem(2, embed_dim)
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            _SpikformerBlock(
                embed_dim, num_heads, mlp_ratio, drop_rates[index]
            )
            for index in range(depth)
        ])
        self.variant = variant
        self.embed_dim = embed_dim
        self.depth = depth
        self.max_grid = tuple(max_grid)

    def forward_features(self, frames):
        if frames.ndim != 5 or frames.shape[2] != 2 or frames.shape[1] < 1:
            raise ValueError("Expected DVS frames with shape [B, T, 2, H, W]")
        if not frames.is_floating_point():
            frames = frames.float()

        x = frames.permute(1, 0, 2, 3, 4).contiguous()
        x = self.patch_embed(x)

        steps, batch, channels, height, width = x.shape
        grid = (min(height, self.max_grid[0]),
                min(width, self.max_grid[1]))
        if grid != (height, width):
            x = F.adaptive_avg_pool2d(x.flatten(0, 1), grid)
            x = x.reshape(steps, batch, channels, *grid)

        x = x.flatten(start_dim=3)
        for block in self.blocks:
            x = block(x)
        return x.reshape(steps, batch, channels, *grid).permute(1, 0, 2, 3, 4).contiguous()

    def forward(self, frames):
        return self.forward_features(frames)
