"""Pair/action queries over an ordered sequence of DVS spatial tokens."""

import torch
from torch import nn
from torch.nn import functional as F


class DVSPairActionQuery(nn.Module):
    def __init__(self, pair_dim, event_dim, num_classes, query_dim=128,
                 num_heads=4, pair_chunk_size=16, spatial_grid=(4, 6),
                 adjacent_changes=False):
        super().__init__()
        if query_dim < 1 or num_heads < 1 or query_dim % num_heads:
            raise ValueError('query_dim must be positive and divisible by num_heads')
        if pair_chunk_size < 1 or min(spatial_grid) < 1:
            raise ValueError('Query chunk size and spatial grid must be positive')
        self.num_classes = num_classes
        self.query_dim = query_dim
        self.pair_chunk_size = pair_chunk_size
        self.spatial_grid = tuple(spatial_grid)
        self.adjacent_changes = adjacent_changes
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_proj = nn.Linear(pair_dim, query_dim)
        self.action_embeddings = nn.Embedding(num_classes, query_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.event_norm = nn.LayerNorm(event_dim)
        self.event_proj = nn.Linear(event_dim, query_dim)
        self.position_proj = nn.Linear(3, query_dim, bias=False)
        if adjacent_changes:
            # Learn from both interval endpoints and signed/magnitude changes.
            self.change_norm = nn.LayerNorm(event_dim * 4)
            self.change_proj = nn.Linear(event_dim * 4, query_dim)
            self.token_type = nn.Embedding(2, query_dim)
            self.interval_proj = nn.Linear(2, query_dim, bias=False)
            nn.init.zeros_(self.token_type.weight[0])
        self.attention = nn.MultiheadAttention(query_dim, num_heads, dropout=0.0)
        self.output_norm = nn.LayerNorm(query_dim)
        self.score = nn.Linear(query_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def prepare_memory(self, sequence):
        """Keep all original bins and optionally (T-1) adjacent intervals."""
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
        if self.adjacent_changes:
            memory = memory + self.token_type.weight[0]
            if steps > 1:
                endpoints = sequence.permute(0, 2, 3, 1)
                changes = self.adjacent_inputs(endpoints).reshape(-1, channels * 4)
                midpoints = (t[:-1] + t[1:]) / 2
                ct, cy, cx = torch.meshgrid(midpoints, y, x)
                change_coordinates = torch.stack([ct, cy, cx], dim=-1).reshape(-1, 3)
                intervals = torch.stack([t[:-1], t[1:]], dim=-1)
                intervals = intervals[:, None, None, :].expand(-1, *grid, -1).reshape(-1, 2)
                change_memory = (self.change_proj(self.change_norm(changes))
                                 + self.position_proj(change_coordinates)
                                 + self.interval_proj(intervals) + self.token_type.weight[1])
                memory = torch.cat([memory, change_memory], dim=0)
        return memory.unsqueeze(1)

    @staticmethod
    def adjacent_inputs(endpoints):
        """[T, ..., C] -> [T-1, ..., 4C]; signed differences retain direction."""
        delta = endpoints[1:] - endpoints[:-1]
        return torch.cat([endpoints[:-1], endpoints[1:], delta, delta.abs()], dim=-1)

    def extract_features(self, pairs, sequence=None, memory=None):
        """Return a separate DVS feature for each candidate pair and action."""
        if pairs.ndim != 2:
            raise ValueError('Expected pair features [P, pair_dim]')
        if len(pairs) == 0:
            return pairs.new_empty((0, self.num_classes, self.query_dim))
        if (sequence is None) == (memory is None):
            raise ValueError('Supply exactly one of sequence and prepared memory')
        if memory is None:
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

    def forward(self, pairs, sequence=None, memory=None):
        return self.score(self.extract_features(pairs, sequence, memory)).squeeze(-1)


class DVSPairRelationQuery(nn.Module):
    """Pair-only event readout injected before competitive relation reasoning."""
    def __init__(self, pair_dim, query_dim=128, num_heads=4, pair_chunk_size=16):
        super().__init__()
        if query_dim < 1 or num_heads < 1 or query_dim % num_heads or pair_chunk_size < 1:
            raise ValueError('Invalid relation query dimensions or chunk size')
        self.pair_chunk_size = pair_chunk_size
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_proj = nn.Linear(pair_dim, query_dim)
        self.attention = nn.MultiheadAttention(query_dim, num_heads, dropout=0.0)
        self.output_norm = nn.LayerNorm(query_dim)
        self.residual = nn.Linear(query_dim, pair_dim)
        # Only the final projection is zero, allowing it to learn immediately.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, pairs, memory):
        if pairs.ndim != 2:
            raise ValueError('Expected pair features [P, pair_dim]')
        if len(pairs) == 0:
            return pairs.new_empty(pairs.shape)
        outputs = []
        for chunk in pairs.split(self.pair_chunk_size):
            query = self.pair_proj(self.pair_norm(chunk)).unsqueeze(1)
            evidence, _ = self.attention(query, memory, memory, need_weights=False)
            outputs.append(self.residual(self.output_norm(evidence.squeeze(1))))
        return torch.cat(outputs, dim=0)
