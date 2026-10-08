"""Pair/action queries over an ordered sequence of DVS spatial tokens."""

import torch
from torch import nn
from torch.nn import functional as F


class DVSPairActionQuery(nn.Module):
    def __init__(self, pair_dim, event_dim, num_classes, query_dim=128,
                 num_heads=4, pair_chunk_size=16, spatial_grid=(4, 6)):
        super().__init__()
        if query_dim < 1 or num_heads < 1 or query_dim % num_heads:
            raise ValueError('query_dim must be positive and divisible by num_heads')
        if pair_chunk_size < 1 or min(spatial_grid) < 1:
            raise ValueError('Query chunk size and spatial grid must be positive')
        self.num_classes = num_classes
        self.query_dim = query_dim
        self.pair_chunk_size = pair_chunk_size
        self.spatial_grid = tuple(spatial_grid)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_proj = nn.Linear(pair_dim, query_dim)
        self.action_embeddings = nn.Embedding(num_classes, query_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.event_norm = nn.LayerNorm(event_dim)
        self.event_proj = nn.Linear(event_dim, query_dim)
        self.position_proj = nn.Linear(3, query_dim, bias=False)
        self.attention = nn.MultiheadAttention(query_dim, num_heads, dropout=0.0)
        self.output_norm = nn.LayerNorm(query_dim)
        self.score = nn.Linear(query_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def prepare_memory(self, sequence):
        """[T, C, H, W] -> [T * H' * W', 1, D]; never pool time."""
        if sequence.ndim != 4 or sequence.shape[0] < 1:
            raise ValueError('Expected DVS sequence [T, C, H, W]')
        steps, channels, height, width = sequence.shape
        grid = (min(height, self.spatial_grid[0]), min(width, self.spatial_grid[1]))
        # Compress space independently for each time bin. Every time bin is
        # retained, with explicit temporal and spatial coordinates.
        if grid != (height, width):
            sequence = F.adaptive_avg_pool2d(sequence, grid)
        tokens = sequence.permute(0, 2, 3, 1).reshape(-1, channels)
        t = torch.linspace(0, 1, steps, device=sequence.device, dtype=sequence.dtype)
        y = torch.linspace(-1, 1, grid[0], device=sequence.device, dtype=sequence.dtype)
        x = torch.linspace(-1, 1, grid[1], device=sequence.device, dtype=sequence.dtype)
        tt, yy, xx = torch.meshgrid(t, y, x)
        coordinates = torch.stack([tt, yy, xx], dim=-1).reshape(-1, 3)
        memory = self.event_proj(self.event_norm(tokens)) + self.position_proj(coordinates)
        return memory.unsqueeze(1)

    def extract_features(self, pairs, sequence):
        """Return a separate DVS feature for each candidate pair and action."""
        if pairs.ndim != 2:
            raise ValueError('Expected pair features [P, pair_dim]')
        if len(pairs) == 0:
            return pairs.new_empty((0, self.num_classes, self.query_dim))
        memory = self.prepare_memory(sequence)
        features = []
        for chunk in pairs.split(self.pair_chunk_size):
            queries = self.pair_proj(self.pair_norm(chunk))[:, None, :]
            queries = self.query_norm(queries + self.action_embeddings.weight[None, :, :])
            queries = queries.reshape(-1, self.query_dim).unsqueeze(1)
            attended, _ = self.attention(queries, memory, memory, need_weights=False)
            attended = attended.reshape(len(chunk), self.num_classes, self.query_dim)
            features.append(self.output_norm(attended))
        return torch.cat(features, dim=0)

    def forward(self, pairs, sequence):
        return self.score(self.extract_features(pairs, sequence)).squeeze(-1)
