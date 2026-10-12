"""Small pair-native sparse model prototype.

The network predicts a value, one unary score per board point, and one pair
embedding per point.  Pair scores are symmetric, so the MCTS action is an
unordered canonical pair from the start rather than two sequential actions.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE


class AttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, expansion: int = 2):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * expansion),
            nn.GELU(),
            nn.Linear(feature_dim * expansion, feature_dim),
        )

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        outputs = inputs + attended
        return outputs + self.feed_forward(self.norm2(outputs))


class CrossAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.norm_query = nn.LayerNorm(feature_dim)
        self.norm_context = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.norm_query(queries),
            self.norm_context(context),
            self.norm_context(context),
            key_padding_mask=~valid,
            need_weights=False,
        )
        outputs = queries + attended
        return outputs + self.feed_forward(outputs)


class PairNativeSparseNet(nn.Module):
    """Fixed-shape sparse-token trunk with a symmetric unordered-pair head."""

    def __init__(
        self,
        input_planes: int = 5,
        feature_dim: int = 192,
        max_tokens: int = 64,
        depth: int = 4,
        num_heads: int = 6,
        pair_rank: int = 32,
        latent_tokens: int = 4,
        dense_channels: int = 0,
    ):
        super().__init__()
        if not 1 <= max_tokens <= BOARD_POINTS:
            raise ValueError("max_tokens must be in [1, 361]")
        self.feature_dim = int(feature_dim)
        self.max_tokens = int(max_tokens)
        self.pair_rank = int(pair_rank)
        self.dense_channels = int(dense_channels)
        self.input_projection = nn.Linear(input_planes, feature_dim)
        self.position_embedding = nn.Embedding(BOARD_POINTS, feature_dim)
        self.latents = nn.Parameter(torch.empty(1, latent_tokens, feature_dim))
        self.token_blocks = nn.ModuleList([
            AttentionBlock(feature_dim, num_heads) for _ in range(depth)
        ])
        self.query_attention = CrossAttention(feature_dim, num_heads)
        self.unary_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.pair_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, pair_rank),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
        )
        self.relative_bias = nn.Embedding(37 * 37, 1)
        if self.dense_channels > 0:
            self.dense_stem = nn.Sequential(
                nn.Conv2d(input_planes, dense_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(dense_channels, dense_channels, 3, padding=1),
                nn.GELU(),
            )
            self.dense_to_token = nn.Linear(dense_channels, feature_dim)
            self.dense_to_query = nn.Linear(dense_channels, feature_dim)
            self.dense_to_latent = nn.Linear(dense_channels, feature_dim)
        else:
            self.dense_stem = None

        selection_order = torch.arange(BOARD_POINTS, dtype=torch.float32)
        self.register_buffer("selection_order", selection_order.view(1, -1), persistent=False)
        coordinates = torch.arange(BOARD_POINTS)
        rows, columns = coordinates.div(BOARD_SIZE, rounding_mode="floor"), coordinates % BOARD_SIZE
        delta_rows = rows[:, None] - rows[None, :] + BOARD_SIZE - 1
        delta_columns = columns[:, None] - columns[None, :] + BOARD_SIZE - 1
        self.register_buffer(
            "relative_indices",
            delta_rows * (2 * BOARD_SIZE - 1) + delta_columns,
            persistent=False,
        )
        self.register_buffer(
            "upper_triangle",
            torch.triu(torch.ones(BOARD_POINTS, BOARD_POINTS, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        nn.init.normal_(self.latents, std=0.02)
        nn.init.zeros_(self.relative_bias.weight)

    def _select_tokens(
        self,
        inputs: torch.Tensor,
        dense_points: torch.Tensor | None = None,
    ):
        point_features = inputs.flatten(2).transpose(1, 2)
        occupancy = point_features[:, :, 0] + point_features[:, :, 1]
        priority = self.selection_order.to(dtype=occupancy.dtype) * 1e-5
        indices = torch.topk(
            occupancy + priority,
            k=self.max_tokens,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        gather = indices.unsqueeze(2).expand(-1, -1, inputs.shape[1])
        selected_features = torch.gather(point_features, 1, gather)
        selected_occupancy = torch.gather(occupancy, 1, indices)
        tokens = self.input_projection(selected_features) + self.position_embedding(indices)
        if dense_points is not None:
            dense_gather = indices.unsqueeze(2).expand(
                -1,
                -1,
                dense_points.shape[2],
            )
            selected_dense = torch.gather(dense_points, 1, dense_gather)
            tokens = tokens + self.dense_to_token(selected_dense)
        valid = selected_occupancy > 0.5
        return tokens, valid, occupancy > 0.5

    def forward(self, inputs: torch.Tensor):
        batch = inputs.shape[0]
        dense_points = None
        dense_latents = None
        if self.dense_stem is not None:
            dense_map = self.dense_stem(inputs)
            dense_points = dense_map.flatten(2).transpose(1, 2)
            pooled = F.adaptive_avg_pool1d(
                dense_map.flatten(2),
                self.latents.shape[1],
            ).transpose(1, 2)
            dense_latents = self.dense_to_latent(pooled)
        tokens, valid, occupied = self._select_tokens(inputs, dense_points)
        latents = self.latents.expand(batch, -1, -1)
        if dense_latents is not None:
            latents = latents + dense_latents
        tokens = torch.cat((latents, tokens), dim=1)
        latent_valid = torch.ones(
            (batch, self.latents.shape[1]),
            dtype=torch.bool,
            device=inputs.device,
        )
        valid = torch.cat((latent_valid, valid), dim=1)
        for block in self.token_blocks:
            tokens = block(tokens, valid)

        coordinates = torch.arange(BOARD_POINTS, device=inputs.device)
        queries = self.position_embedding(coordinates).unsqueeze(0).expand(batch, -1, -1)
        if dense_points is not None:
            queries = queries + self.dense_to_query(dense_points)
        point_features = self.query_attention(queries, tokens, valid)
        unary_logits = self.unary_head(point_features).squeeze(-1)
        pair_embeddings = self.pair_head(point_features)
        global_features = tokens[:, : self.latents.shape[1]].mean(dim=1)
        value = self.value_head(global_features).squeeze(-1)
        return unary_logits, pair_embeddings, value, occupied

    def pair_scores(
        self,
        unary_logits: torch.Tensor,
        pair_embeddings: torch.Tensor,
        occupied: torch.Tensor,
    ) -> torch.Tensor:
        interactions = torch.matmul(
            pair_embeddings,
            pair_embeddings.transpose(1, 2),
        ) / math.sqrt(self.pair_rank)
        scores = interactions + unary_logits[:, :, None] + unary_logits[:, None, :]
        bias = self.relative_bias(self.relative_indices).squeeze(-1)
        scores = scores + bias.to(dtype=scores.dtype)
        legal = (
            self.upper_triangle.unsqueeze(0)
            & ~occupied[:, :, None]
            & ~occupied[:, None, :]
        )
        return scores.masked_fill(~legal, torch.finfo(scores.dtype).min)

    def forward_topk(self, inputs: torch.Tensor, top_k: int = 64):
        unary_logits, pair_embeddings, value, occupied = self(inputs)
        scores = self.pair_scores(unary_logits, pair_embeddings, occupied)
        top_scores, flat_indices = torch.topk(
            scores.flatten(1),
            k=min(int(top_k), BOARD_POINTS * (BOARD_POINTS - 1) // 2),
            dim=1,
        )
        first_moves = flat_indices.div(BOARD_POINTS, rounding_mode="floor")
        second_moves = flat_indices % BOARD_POINTS
        return first_moves, second_moves, top_scores, value, unary_logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


DEFAULT_MODEL_CONFIG = {
    "input_planes": 5,
    "feature_dim": 192,
    "max_tokens": 96,
    "depth": 4,
    "num_heads": 6,
    "pair_rank": 32,
    "latent_tokens": 4,
    "dense_channels": 32,
}


def build_model(config: dict | None = None) -> PairNativeSparseNet:
    values = dict(DEFAULT_MODEL_CONFIG)
    if config:
        values.update(config)
    return PairNativeSparseNet(**values)


def save_random_checkpoint(
    path: str | Path,
    config: dict | None = None,
    seed: int = 2026,
) -> Path:
    torch.manual_seed(seed)
    model_config = dict(DEFAULT_MODEL_CONFIG)
    if config:
        model_config.update(config)
    model = build_model(model_config)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({
        "schema_version": 1,
        "model_config": model_config,
        "seed": int(seed),
        "model_state_dict": model.state_dict(),
    }, temporary)
    temporary.replace(output)
    return output


def load_checkpoint(path: str | Path, map_location="cpu"):
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
