"""Content-conditioned entity slots over time, without RGB-to-DVS box mapping."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DVSTemporalEntityRelation(nn.Module):
    """Read each detected entity once, then compose its incident HOI pairs.

    RGB features initialize entity identity queries, never event coordinates.
    Event tokens compete over entities plus a background slot at each time.
    Only event readouts (not the RGB anchors or GRU states) become pair memory.
    """
    def __init__(self, entity_dim, pair_dim, query_dim=128, num_heads=4,
                 pair_chunk_size=16, slot_iterations=2):
        super().__init__()
        if (query_dim < 1 or num_heads < 1 or query_dim % num_heads
                or pair_chunk_size < 1 or slot_iterations < 1):
            raise ValueError('Invalid entity relation dimensions or iteration count')
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.pair_chunk_size = pair_chunk_size
        self.slot_iterations = slot_iterations
        self.entity_norm = nn.LayerNorm(entity_dim)
        self.entity_proj = nn.Linear(entity_dim, query_dim)
        self.role_embeddings = nn.Embedding(2, query_dim)  # object / human
        nn.init.normal_(self.role_embeddings.weight, std=0.02)
        self.background = nn.Parameter(torch.randn(1, query_dim) / math.sqrt(query_dim))
        self.memory_norm = nn.LayerNorm(query_dim)
        self.slot_norm = nn.LayerNorm(query_dim)
        self.key = nn.Linear(query_dim, query_dim, bias=False)
        self.value = nn.Linear(query_dim, query_dim, bias=False)
        self.slot_query = nn.Linear(query_dim, query_dim, bias=False)
        self.update = nn.GRUCell(query_dim, query_dim)
        # Event-internal soft location/spread, used as evidence values only.
        self.geometry_proj = nn.Linear(4, query_dim, bias=False)
        # Cosine logits have a fixed bound, so norm growth cannot saturate
        # assignment in the way unbounded dot-product queries can.
        self.log_temperature = nn.Parameter(torch.full((num_heads,), math.log(4.0)))
        self.pair_event_norm = nn.LayerNorm(query_dim * 4)
        self.pair_event_proj = nn.Linear(query_dim * 4, query_dim)
        self.time_proj = nn.Linear(1, query_dim, bias=False)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_proj = nn.Linear(pair_dim, query_dim)
        self.temporal_key = nn.Linear(query_dim, query_dim, bias=False)
        self.temporal_value = nn.Linear(query_dim, query_dim, bias=False)
        self.output_norm = nn.LayerNorm(query_dim)
        self.residual = nn.Linear(query_dim, pair_dim, bias=False)
        nn.init.zeros_(self.residual.weight)

    def _heads(self, values):
        return values.reshape(-1, self.num_heads, self.query_dim // self.num_heads).transpose(0, 1)

    def _scale(self):
        return self.log_temperature.clamp(math.log(1.0), math.log(10.0)).exp()[:, None, None]

    def assign(self, queries, tokens):
        """Return content evidence and maps, including the background slot.

        Competition is across unique entities, not pairs: a human shared by
        several HOIs must not compete against duplicate copies of itself.
        """
        normal_tokens = self.memory_norm(tokens)
        q = F.normalize(self._heads(self.slot_query(self.slot_norm(queries))), dim=-1)
        k = F.normalize(self._heads(self.key(normal_tokens)), dim=-1)
        logits = (q @ k.transpose(-1, -2)) * self._scale()
        log_responsibilities = logits.log_softmax(dim=1)  # [heads, slots, spatial tokens]
        responsibilities = log_responsibilities.exp()
        # Each slot then pools the spatial tokens assigned to it. This is
        # spatial aggregation within one bin; there is no time mean.
        # Log-space renormalization preserves unit spatial mass even for a
        # slot with tiny responsibility, instead of changing it via epsilon.
        weights = log_responsibilities.softmax(dim=-1)
        values = self._heads(self.value(normal_tokens))
        evidence = (weights @ values).transpose(0, 1).reshape(len(queries), self.query_dim)
        return evidence, weights.mean(dim=0), responsibilities.mean(dim=0)

    def extract_entity_sequence(self, entities, is_human, temporal_memory, return_attention=False,
                                spatial_grid=None):
        """[N,U], [N], [T,S,D] -> event-only [N,T,D] identity readouts."""
        if entities.ndim != 2 or is_human.shape != (len(entities),):
            raise ValueError('Expected entity features [N,U] and roles [N]')
        if (temporal_memory.ndim != 3 or temporal_memory.shape[0] < 1
                or temporal_memory.shape[1] < 1 or temporal_memory.shape[2] != self.query_dim):
            raise ValueError('Expected original-bin memory [T,S,query_dim]')
        steps, spatial_tokens = temporal_memory.shape[:2]
        coordinates = None
        if spatial_grid is not None:
            if len(spatial_grid) != 2 or min(spatial_grid) < 1 or spatial_grid[0] * spatial_grid[1] != spatial_tokens:
                raise ValueError('Spatial grid must match original-bin token count')
            ys = (torch.linspace(-1, 1, spatial_grid[0], device=temporal_memory.device,
                                 dtype=temporal_memory.dtype) if spatial_grid[0] > 1
                  else temporal_memory.new_zeros(1))
            xs = (torch.linspace(-1, 1, spatial_grid[1], device=temporal_memory.device,
                                 dtype=temporal_memory.dtype) if spatial_grid[1] > 1
                  else temporal_memory.new_zeros(1))
            yy, xx = torch.meshgrid(ys, xs)
            coordinates = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        if len(entities) == 0:
            result = entities.new_empty((0, steps, self.query_dim))
            if return_attention:
                return result, entities.new_empty((0, steps, spatial_tokens))
            return result
        anchors = self.entity_proj(self.entity_norm(entities)) + self.role_embeddings(is_human.long())
        anchors = torch.cat([anchors, self.background], dim=0)
        state = anchors
        readouts, maps = [], []
        for tokens in temporal_memory.unbind(0):
            for _ in range(self.slot_iterations):
                evidence, _, _ = self.assign(anchors + state, tokens)
                state = self.update(evidence, state)
            # Read using the updated state; keep the original RGB anchor in
            # every query to resist identity drift under motion or occlusion.
            evidence, weights, _ = self.assign(anchors + state, tokens)
            if coordinates is not None:
                center = weights @ coordinates
                variance = (weights @ coordinates.square() - center.square()).clamp_min(0)
                evidence = evidence + self.geometry_proj(torch.cat([center, variance], dim=-1))
            readouts.append(evidence[:-1])
            if return_attention:
                maps.append(weights[:-1])
        result = torch.stack(readouts, dim=1)
        if return_attention:
            return result, torch.stack(maps, dim=1)
        return result

    def read_pairs(self, pairs, entity_sequence, human_indices, object_indices):
        """Compose per-bin person/object evidence, then learn a temporal readout."""
        if pairs.ndim != 2 or human_indices.shape != (len(pairs),) or object_indices.shape != (len(pairs),):
            raise ValueError('Expected pair features [P,R] and pair entity indices [P]')
        if len(pairs) == 0:
            return pairs.new_empty(pairs.shape)
        steps = entity_sequence.shape[1]
        times = torch.linspace(0, 1, steps, device=pairs.device, dtype=pairs.dtype).unsqueeze(-1)
        outputs = []
        for start in range(0, len(pairs), self.pair_chunk_size):
            end = start + self.pair_chunk_size
            chunk = pairs[start:end]
            human = entity_sequence[human_indices[start:end]]
            obj = entity_sequence[object_indices[start:end]]
            # Values originate solely from event evidence. RGB anchors select
            # evidence but are not copied into the supposed event residual.
            relation = torch.cat([human, obj, human - obj, human * obj], dim=-1)
            memory = self.pair_event_proj(self.pair_event_norm(relation)) + self.time_proj(times)
            q = self._heads(self.pair_proj(self.pair_norm(chunk))).unsqueeze(2)
            keys = self.temporal_key(self.memory_norm(memory))
            values = self.temporal_value(self.memory_norm(memory))
            shape = (len(chunk), steps, self.num_heads, self.query_dim // self.num_heads)
            keys = keys.reshape(shape).permute(2, 0, 1, 3)
            values = values.reshape(shape).permute(2, 0, 1, 3)
            logits = (F.normalize(q, dim=-1) * F.normalize(keys, dim=-1)).sum(-1)
            weights = (logits * self._scale()).softmax(dim=-1)
            evidence = (weights.unsqueeze(-1) * values).sum(dim=2)
            evidence = evidence.transpose(0, 1).reshape(len(chunk), self.query_dim)
            outputs.append(self.residual(self.output_norm(evidence)))
        return torch.cat(outputs, dim=0)

    def forward(self, pairs, entities, is_human, human_indices, object_indices, temporal_memory,
                spatial_grid=None):
        sequence = self.extract_entity_sequence(entities, is_human, temporal_memory,
                                                spatial_grid=spatial_grid)
        return self.read_pairs(pairs, sequence, human_indices, object_indices)
