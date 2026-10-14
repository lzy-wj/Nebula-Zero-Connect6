"""Throughput-oriented dense and whole-position MoE models for Connect6."""

from __future__ import annotations

import math

import torch
from torch import nn


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE

class FeedForward(nn.Module):
    def __init__(self, feature_dim: int, expansion: int = 2):
        super().__init__()
        hidden_dim = feature_dim * expansion
        self.norm = nn.LayerNorm(feature_dim)
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(self.norm(inputs))


class SelfAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            batch_first=True,
        )
        self.feed_forward = FeedForward(feature_dim, expansion)

    def mix(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        return inputs + attended

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mixed = self.mix(inputs)
        return mixed + self.feed_forward(mixed)


class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, expansion: int = 2):
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            batch_first=True,
        )
        self.feed_forward = FeedForward(feature_dim, expansion)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            normalized_queries,
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        outputs = queries + attended
        return outputs + self.feed_forward(outputs)


class RoutedExpert(nn.Module):
    def __init__(self, feature_dim: int, expansion: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList(
            FeedForward(feature_dim, expansion) for _ in range(depth)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = outputs + layer(outputs)
        return outputs - inputs


class WholePositionMoE(nn.Module):
    """Route every summary token from one position to the same top-1 expert."""

    def __init__(
        self,
        feature_dim: int,
        num_experts: int = 4,
        expert_expansion: int = 4,
        expert_depth: int = 1,
        shared_expansion: int = 0,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least two")
        self.num_experts = int(num_experts)
        self.router_norm = nn.LayerNorm(feature_dim)
        self.router = nn.Linear(feature_dim, num_experts)
        self.experts = nn.ModuleList(
            RoutedExpert(feature_dim, expert_expansion, expert_depth)
            for _ in range(num_experts)
        )
        self.shared_expert = (
            RoutedExpert(feature_dim, shared_expansion, 1)
            if shared_expansion > 0
            else None
        )

    def forward(self, inputs: torch.Tensor):
        pooled = self.router_norm(inputs.mean(dim=1))
        logits = self.router(pooled)
        probabilities = torch.softmax(logits, dim=-1)
        assignments = probabilities.argmax(dim=-1)
        routed = torch.zeros_like(inputs)
        for expert_index, expert in enumerate(self.experts):
            indices = torch.nonzero(
                assignments.eq(expert_index),
                as_tuple=False,
            ).flatten()
            if indices.numel() == 0:
                continue
            selected = inputs.index_select(0, indices)
            expert_outputs = expert(selected)
            # The forward scale is exactly one. Its straight-through gradient
            # lets the task loss teach the router despite hard top-1 dispatch.
            gate = probabilities.index_select(0, indices)[:, expert_index]
            scale = 1.0 + gate - gate.detach()
            expert_outputs = expert_outputs * scale[:, None, None]
            routed = routed.index_copy(0, indices, expert_outputs)

        outputs = inputs + routed
        if self.shared_expert is not None:
            outputs = outputs + self.shared_expert(inputs)

        importance = probabilities.mean(dim=0)
        load = torch.nn.functional.one_hot(
            assignments,
            num_classes=self.num_experts,
        ).to(probabilities.dtype).mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance * load.detach())
        router_z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        return outputs, {
            "balance_loss": balance_loss,
            "router_z_loss": router_z_loss,
            "assignments": assignments,
            "probabilities": probabilities,
            "load": load,
        }


class NebulaV4(nn.Module):
    """Compress all 361 points, process summaries, then decode all Pair scores."""

    def __init__(
        self,
        variant: str = "dense",
        input_planes: int = 5,
        stem_channels: int = 32,
        feature_dim: int = 160,
        summary_tokens: int = 64,
        depth: int = 4,
        num_heads: int = 5,
        pair_rank: int = 48,
        num_experts: int = 4,
        expert_expansion: int = 4,
        expert_depth: int = 1,
        shared_expansion: int = 0,
    ):
        super().__init__()
        if variant not in {"dense", "moe"}:
            raise ValueError("variant must be 'dense' or 'moe'")
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 1 <= summary_tokens <= BOARD_POINTS:
            raise ValueError("summary_tokens must be in [1, 361]")
        if depth < 1:
            raise ValueError("depth must be positive")
        self.variant = variant
        self.feature_dim = int(feature_dim)
        self.summary_tokens = int(summary_tokens)
        self.pair_rank = int(pair_rank)
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, stem_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1),
            nn.GELU(),
        )
        self.point_projection = nn.Linear(stem_channels, feature_dim)
        self.row_embedding = nn.Embedding(BOARD_SIZE, feature_dim)
        self.column_embedding = nn.Embedding(BOARD_SIZE, feature_dim)
        self.summary_queries = nn.Parameter(
            torch.empty(1, summary_tokens, feature_dim)
        )
        self.compressor = CrossAttentionBlock(feature_dim, num_heads)

        common_depth = depth if variant == "dense" else max(depth - 1, 0)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(feature_dim, num_heads)
            for _ in range(common_depth)
        )
        if variant == "moe":
            self.final_mixer = SelfAttentionBlock(feature_dim, num_heads)
            self.moe = WholePositionMoE(
                feature_dim,
                num_experts=num_experts,
                expert_expansion=expert_expansion,
                expert_depth=expert_depth,
                shared_expansion=shared_expansion,
            )
        else:
            self.final_mixer = None
            self.moe = None

        self.decoder = CrossAttentionBlock(feature_dim, num_heads)
        self.unary_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, pair_rank * 2),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
        )
        self.relative_bias = nn.Embedding((2 * BOARD_SIZE - 1) ** 2, 1)

        coordinates = torch.arange(BOARD_POINTS)
        rows = coordinates.div(BOARD_SIZE, rounding_mode="floor")
        columns = coordinates % BOARD_SIZE
        delta_rows = rows[:, None] - rows[None, :] + BOARD_SIZE - 1
        delta_columns = columns[:, None] - columns[None, :] + BOARD_SIZE - 1
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("columns", columns, persistent=False)
        self.register_buffer(
            "relative_indices",
            delta_rows * (2 * BOARD_SIZE - 1) + delta_columns,
            persistent=False,
        )
        self.register_buffer(
            "upper_triangle",
            torch.triu(
                torch.ones(BOARD_POINTS, BOARD_POINTS, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        nn.init.normal_(self.summary_queries, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.column_embedding.weight, std=0.02)
        nn.init.zeros_(self.relative_bias.weight)

    def positional_features(self) -> torch.Tensor:
        return self.row_embedding(self.rows) + self.column_embedding(self.columns)

    def forward(
        self,
        inputs: torch.Tensor,
        return_aux: bool = False,
        return_pair_scores: bool = False,
    ):
        batch = inputs.shape[0]
        dense_map = self.stem(inputs)
        points = dense_map.flatten(2).transpose(1, 2)
        points = self.point_projection(points) + self.positional_features()[None]
        summaries = self.compressor(
            self.summary_queries.expand(batch, -1, -1),
            points,
        )
        for block in self.blocks:
            summaries = block(summaries)

        auxiliary = {}
        if self.moe is not None:
            # The final attention remains a dense batched operation. Only its
            # feed-forward capacity is selected per complete position.
            summaries = self.final_mixer.mix(summaries)
            summaries, auxiliary = self.moe(summaries)

        decoded = self.decoder(points, summaries)
        unary_logits = self.unary_head(decoded).squeeze(-1)
        pair_embeddings = self.pair_head(decoded)
        pooled = torch.cat(
            (summaries.mean(dim=1), summaries.amax(dim=1)),
            dim=-1,
        )
        value = self.value_head(pooled).squeeze(-1)
        occupied = inputs[:, 2].flatten(1).lt(0.5)
        outputs = (unary_logits, pair_embeddings, value, occupied)
        if return_pair_scores:
            outputs = (*outputs, self.pair_scores(unary_logits, pair_embeddings, occupied))
        if return_aux:
            return (*outputs, auxiliary)
        return outputs

    def pair_scores(
        self,
        unary_logits: torch.Tensor,
        pair_embeddings: torch.Tensor,
        occupied: torch.Tensor,
    ) -> torch.Tensor:
        left, right = pair_embeddings.split(self.pair_rank, dim=-1)
        forward = torch.matmul(left, right.transpose(1, 2))
        interactions = (forward + forward.transpose(1, 2)) * (
            0.5 / math.sqrt(self.pair_rank)
        )
        scores = interactions + unary_logits[:, :, None] + unary_logits[:, None, :]
        bias = self.relative_bias(self.relative_indices).squeeze(-1)
        scores = scores + bias.to(dtype=scores.dtype)
        legal = (
            self.upper_triangle[None]
            & ~occupied[:, :, None]
            & ~occupied[:, None, :]
        )
        return scores.masked_fill(~legal, torch.finfo(scores.dtype).min)

    def pair_row_scores(
        self,
        unary_logits: torch.Tensor,
        pair_embeddings: torch.Tensor,
        occupied: torch.Tensor,
        first_moves: torch.Tensor,
    ) -> torch.Tensor:
        """Score every legal second stone for selected first stones.

        This is algebraically identical to rows of ``pair_scores`` after
        canonicalizing an unordered pair, but avoids materializing 361x361.
        ``first_moves`` has shape ``[batch, selected_first]``.
        """
        left, right = pair_embeddings.split(self.pair_rank, dim=-1)
        gather_index = first_moves[..., None].expand(-1, -1, self.pair_rank)
        first_left = left.gather(1, gather_index)
        first_right = right.gather(1, gather_index)
        interactions = (
            torch.einsum("bfr,bjr->bfj", first_left, right)
            + torch.einsum("bjr,bfr->bfj", left, first_right)
        ) * (0.5 / math.sqrt(self.pair_rank))
        first_unary = unary_logits.gather(1, first_moves)
        scores = interactions + first_unary[..., None] + unary_logits[:, None, :]

        seconds = torch.arange(BOARD_POINTS, device=first_moves.device)
        canonical_first = torch.minimum(first_moves[..., None], seconds)
        canonical_second = torch.maximum(first_moves[..., None], seconds)
        relative_indices = self.relative_indices[
            canonical_first,
            canonical_second,
        ]
        scores = scores + self.relative_bias(relative_indices).squeeze(-1).to(
            scores.dtype
        )

        first_occupied = occupied.gather(1, first_moves)
        legal = (
            ~first_occupied[..., None]
            & ~occupied[:, None, :]
            & first_moves[..., None].ne(seconds)
        )
        return scores.masked_fill(~legal, torch.finfo(scores.dtype).min)

    def forward_topk(self, inputs: torch.Tensor, top_k: int = 64):
        unary, pair_embeddings, value, occupied = self(inputs)
        scores = self.pair_scores(unary, pair_embeddings, occupied)
        top_scores, flat_indices = torch.topk(scores.flatten(1), top_k, dim=1)
        first = flat_indices.div(BOARD_POINTS, rounding_mode="floor")
        second = flat_indices % BOARD_POINTS
        return first, second, top_scores, value, unary


DEFAULT_CONFIG = {
    "variant": "dense",
    "input_planes": 5,
    "stem_channels": 32,
    "feature_dim": 160,
    "summary_tokens": 64,
    "depth": 4,
    "num_heads": 5,
    "pair_rank": 48,
    "num_experts": 4,
    "expert_expansion": 4,
    "expert_depth": 1,
    "shared_expansion": 0,
}

CANDIDATE_CONFIGS = {
    "dense": {
        **DEFAULT_CONFIG,
        "variant": "dense",
        "feature_dim": 160,
        "depth": 4,
        "num_heads": 5,
    },
    "moe": {
        **DEFAULT_CONFIG,
        "variant": "moe",
        "feature_dim": 128,
        "depth": 2,
        "num_heads": 4,
        "expert_expansion": 8,
        "expert_depth": 2,
    },
}


def build_model(config: dict | None = None) -> NebulaV4:
    values = dict(DEFAULT_CONFIG)
    if config:
        values.update(config)
    return NebulaV4(**values)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def active_parameter_count(model: NebulaV4) -> int:
    total = parameter_count(model)
    if model.moe is None:
        return total
    inactive_experts = sum(
        parameter_count(expert) for expert in list(model.moe.experts)[1:]
    )
    return total - inactive_experts
