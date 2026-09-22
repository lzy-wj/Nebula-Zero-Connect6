"""Inference-first policy/value backbones for the fixed 19x19 board."""

import torch
from torch import nn
from torch.nn import functional as F


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE


class PolicyValueMixin:
    """Common policy/value interface used by the experimental exporter."""

    feature_dim: int

    def forward_features(self, inputs):
        raise NotImplementedError

    def forward(self, inputs, move1_idx=None, return_features=False):
        del move1_idx
        features, global_features = self.forward_features(inputs)
        policy = self.policy_head(features).squeeze(-1)
        value = self.value_head(global_features)
        if return_features:
            return policy, None, value, features
        return policy, None, value

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class BottleneckBlock(nn.Module):
    """TensorRT-friendly 1x1 -> 3x3 -> 1x1 residual bottleneck."""

    def __init__(self, channels, hidden_channels, dilation=1):
        super().__init__()
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.spatial = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(self, inputs):
        outputs = F.relu(self.reduce(inputs))
        outputs = F.relu(self.spatial(outputs))
        outputs = self.expand(outputs)
        return F.relu(outputs + inputs)


class GlobalPoolBlock(nn.Module):
    """Inject whole-board mean/max context without quadratic attention."""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, inputs):
        mean = inputs.mean(dim=(2, 3))
        maximum = inputs.amax(dim=(2, 3))
        context = self.project(torch.cat((mean, maximum), dim=1))
        return F.relu(inputs + context.unsqueeze(2).unsqueeze(3))


class BottleneckGlobalNet(PolicyValueMixin, nn.Module):
    """Dense convolutional candidate with cheap periodic global context."""

    def __init__(
        self,
        input_planes=5,
        channels=192,
        hidden_channels=96,
        depth=12,
        global_every=4,
        dilations=(1, 2, 3, 1),
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        if global_every <= 0:
            raise ValueError("global_every must be positive")
        if not dilations:
            raise ValueError("dilations must not be empty")

        self.feature_dim = int(channels)
        self.stem = nn.Conv2d(
            input_planes,
            channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        blocks = []
        for index in range(depth):
            blocks.append(
                BottleneckBlock(
                    channels,
                    hidden_channels,
                    dilation=int(dilations[index % len(dilations)]),
                )
            )
            if (index + 1) % global_every == 0:
                blocks.append(GlobalPoolBlock(channels, hidden_channels))
        self.blocks = nn.ModuleList(blocks)
        self.policy_head = nn.Sequential(
            nn.Linear(channels, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward_features(self, inputs):
        features = F.relu(self.stem(inputs))
        for block in self.blocks:
            features = block(features)
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, tokens.mean(dim=1)


class DenseAttention(nn.Module):
    """Small fixed-shape attention expressed using basic TensorRT operators."""

    def __init__(self, feature_dim, num_heads):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.feature_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.output = nn.Linear(feature_dim, feature_dim)

    def forward(self, inputs, key_mask=None):
        batch, tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, None, :],
                -10_000.0,
            )
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        outputs = torch.matmul(attention, value)
        outputs = outputs.transpose(1, 2).reshape(batch, tokens, channels)
        return self.output(outputs)


class CrossAttention(nn.Module):
    """Cross-attention used by sparse stone queries and dual-scale fusion."""

    def __init__(self, feature_dim, num_heads):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.feature_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key_value = nn.Linear(feature_dim, feature_dim * 2)
        self.output = nn.Linear(feature_dim, feature_dim)

    def forward(self, queries, context, key_mask=None):
        batch, query_count, channels = queries.shape
        context_count = context.shape[1]
        query = self.query(queries).reshape(
            batch,
            query_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_value = self.key_value(context).reshape(
            batch,
            context_count,
            2,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            scores = scores.masked_fill(
                ~key_mask[:, None, None, :],
                -10_000.0,
            )
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        outputs = torch.matmul(attention, value)
        outputs = outputs.transpose(1, 2).reshape(batch, query_count, channels)
        return self.output(outputs)


class AttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attention = DenseAttention(feature_dim, num_heads)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * expansion),
            nn.GELU(),
            nn.Linear(feature_dim * expansion, feature_dim),
        )

    def forward(self, inputs, key_mask=None):
        outputs = inputs + self.attention(self.norm1(inputs), key_mask=key_mask)
        return outputs + self.feed_forward(self.norm2(outputs))


class DualScaleNet(PolicyValueMixin, nn.Module):
    """Full-resolution local trunk with attention only on a 5x5 global map."""

    def __init__(
        self,
        input_planes=5,
        channels=192,
        hidden_channels=96,
        local_depth=10,
        global_depth=2,
        num_heads=6,
    ):
        super().__init__()
        if local_depth < 2:
            raise ValueError("local_depth must be at least two")
        self.feature_dim = int(channels)
        self.stem = nn.Conv2d(input_planes, channels, 3, padding=1, bias=True)
        split = local_depth - 2
        self.local_blocks = nn.ModuleList([
            BottleneckBlock(
                channels,
                hidden_channels,
                dilation=(1, 2, 3, 1)[index % 4],
            )
            for index in range(split)
        ])
        self.refine_blocks = nn.ModuleList([
            BottleneckBlock(channels, hidden_channels, dilation=1)
            for _ in range(2)
        ])
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.global_blocks = nn.ModuleList([
            AttentionBlock(channels, num_heads=num_heads, expansion=2)
            for _ in range(global_depth)
        ])
        self.global_projection = nn.Conv2d(channels, channels, 1, bias=True)
        self.policy_head = nn.Sequential(
            nn.Linear(channels, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward_features(self, inputs):
        local = F.relu(self.stem(inputs))
        for block in self.local_blocks:
            local = block(local)

        global_map = self.downsample(local)
        global_tokens = global_map.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            global_tokens = block(global_tokens)
        global_map = global_tokens.transpose(1, 2).reshape(
            inputs.shape[0],
            self.feature_dim,
            5,
            5,
        )
        global_map = F.interpolate(
            global_map,
            size=(BOARD_SIZE, BOARD_SIZE),
            mode="nearest",
        )
        local = F.relu(local + self.global_projection(global_map))
        for block in self.refine_blocks:
            local = block(local)

        tokens = local.flatten(2).transpose(1, 2)
        global_features = global_tokens.mean(dim=1)
        return tokens, global_features


class PyramidNet(PolicyValueMixin, nn.Module):
    """Keep the policy grid shallow while moving capacity to cheaper scales."""

    def __init__(
        self,
        input_planes=5,
        high_channels=192,
        mid_channels=320,
        low_channels=512,
        feature_dim=320,
        high_depth=2,
        mid_depth=5,
        low_depth=7,
        global_depth=1,
        num_heads=8,
    ):
        super().__init__()
        if min(high_depth, mid_depth, low_depth) < 1:
            raise ValueError("pyramid stage depths must be positive")
        if global_depth < 0:
            raise ValueError("global_depth must not be negative")
        if low_channels % num_heads:
            raise ValueError("low_channels must be divisible by num_heads")

        self.feature_dim = int(feature_dim)
        self.stem = nn.Conv2d(
            input_planes,
            high_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.high_blocks = nn.ModuleList([
            BottleneckBlock(
                high_channels,
                high_channels // 2,
                dilation=(1, 2)[index % 2],
            )
            for index in range(high_depth)
        ])
        self.down_mid = nn.Conv2d(
            high_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        self.mid_blocks = nn.ModuleList([
            BottleneckBlock(
                mid_channels,
                mid_channels // 2,
                dilation=(1, 2, 3, 1)[index % 4],
            )
            for index in range(mid_depth)
        ])
        self.down_low = nn.Conv2d(
            mid_channels,
            low_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        self.low_blocks = nn.ModuleList([
            BottleneckBlock(low_channels, low_channels // 2)
            for _ in range(low_depth)
        ])
        self.global_blocks = nn.ModuleList([
            AttentionBlock(low_channels, num_heads=num_heads, expansion=2)
            for _ in range(global_depth)
        ])

        self.low_to_mid = nn.Conv2d(low_channels, mid_channels, 1, bias=True)
        self.mid_refine = BottleneckBlock(mid_channels, mid_channels // 2)
        self.mid_to_high = nn.Conv2d(
            mid_channels,
            feature_dim,
            1,
            bias=True,
        )
        self.high_skip = nn.Conv2d(
            high_channels,
            feature_dim,
            1,
            bias=True,
        )
        self.high_refine = BottleneckBlock(feature_dim, feature_dim // 2)
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(low_channels, low_channels),
            nn.ReLU(),
            nn.Linear(low_channels, 1),
            nn.Tanh(),
        )

    def forward_features(self, inputs):
        high = F.relu(self.stem(inputs))
        for block in self.high_blocks:
            high = block(high)

        mid = F.relu(self.down_mid(high))
        for block in self.mid_blocks:
            mid = block(mid)

        low = F.relu(self.down_low(mid))
        for block in self.low_blocks:
            low = block(low)

        low_tokens = low.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            low_tokens = block(low_tokens)
        low = low_tokens.transpose(1, 2).reshape(
            inputs.shape[0],
            low_tokens.shape[2],
            5,
            5,
        )

        decoded_mid = F.interpolate(low, size=(10, 10), mode="nearest")
        decoded_mid = F.relu(self.low_to_mid(decoded_mid) + mid)
        decoded_mid = self.mid_refine(decoded_mid)
        decoded_high = F.interpolate(
            decoded_mid,
            size=(BOARD_SIZE, BOARD_SIZE),
            mode="nearest",
        )
        decoded_high = F.relu(
            self.mid_to_high(decoded_high) + self.high_skip(high)
        )
        decoded_high = self.high_refine(decoded_high)

        tokens = decoded_high.flatten(2).transpose(1, 2)
        return tokens, low_tokens.mean(dim=1)


class SparseStoneNet(PolicyValueMixin, nn.Module):
    """Represent occupied points sparsely and query all policy coordinates."""

    def __init__(
        self,
        input_planes=5,
        feature_dim=192,
        max_stones=64,
        stone_depth=4,
        num_heads=6,
    ):
        super().__init__()
        if not 1 <= max_stones <= BOARD_POINTS:
            raise ValueError("max_stones must be in [1, 361]")
        self.feature_dim = int(feature_dim)
        self.max_stones = int(max_stones)
        self.register_buffer(
            "selection_order",
            torch.arange(BOARD_POINTS, dtype=torch.int64).view(1, -1),
            persistent=False,
        )
        self.input_projection = nn.Linear(input_planes, feature_dim)
        self.position_embedding = nn.Embedding(BOARD_POINTS, feature_dim)
        self.global_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.stone_blocks = nn.ModuleList([
            AttentionBlock(feature_dim, num_heads=num_heads, expansion=2)
            for _ in range(stone_depth)
        ])
        self.query_norm = nn.LayerNorm(feature_dim)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.policy_cross_attention = CrossAttention(feature_dim, num_heads)
        self.policy_feed_forward = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
        )
        nn.init.normal_(self.global_token, std=0.02)

    def _select_stone_indices(self, occupancy):
        # TensorRT and PyTorch may break equal TopK ties differently.  A tiny,
        # fixed positional priority makes export validation reproducible while
        # remaining far below the occupied/empty score gap.
        priority = self.selection_order.float() * 1e-5
        scores = occupancy.float() + priority
        return torch.topk(
            scores,
            k=self.max_stones,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

    def forward_features(self, inputs):
        batch = inputs.shape[0]
        point_features = inputs.flatten(2).transpose(1, 2)
        occupancy = point_features[:, :, 0] + point_features[:, :, 1]
        indices = self._select_stone_indices(occupancy)
        gather_index = indices.unsqueeze(2).expand(-1, -1, inputs.shape[1])
        selected_features = torch.gather(point_features, 1, gather_index)
        selected_occupancy = torch.gather(occupancy, 1, indices)
        stones = (
            self.input_projection(selected_features)
            + self.position_embedding(indices)
        )

        global_token = self.global_token.expand(batch, -1, -1)
        tokens = torch.cat((global_token, stones), dim=1)
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=inputs.device),
                selected_occupancy.gt(0.5),
            ),
            dim=1,
        )
        for block in self.stone_blocks:
            tokens = block(tokens, key_mask=valid)

        coordinate_indices = torch.arange(
            BOARD_POINTS,
            dtype=torch.long,
            device=inputs.device,
        )
        queries = self.position_embedding(coordinate_indices).unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )
        queries = queries + self.policy_cross_attention(
            self.query_norm(queries),
            self.context_norm(tokens),
            key_mask=valid,
        )
        queries = queries + self.policy_feed_forward(queries)
        return queries, tokens[:, 0]


ARCHITECTURES = {
    "bottleneck_c192": lambda: BottleneckGlobalNet(
        channels=192,
        hidden_channels=96,
        depth=12,
    ),
    "bottleneck_c256": lambda: BottleneckGlobalNet(
        channels=256,
        hidden_channels=128,
        depth=12,
    ),
    "bottleneck_c320_d16": lambda: BottleneckGlobalNet(
        channels=320,
        hidden_channels=160,
        depth=16,
    ),
    "dual_scale_c192": lambda: DualScaleNet(
        channels=192,
        hidden_channels=96,
        local_depth=10,
        global_depth=2,
        num_heads=6,
    ),
    "dual_scale_c256_d14": lambda: DualScaleNet(
        channels=256,
        hidden_channels=128,
        local_depth=14,
        global_depth=3,
        num_heads=8,
    ),
    "dual_scale_c320_d16": lambda: DualScaleNet(
        channels=320,
        hidden_channels=160,
        local_depth=16,
        global_depth=3,
        num_heads=10,
    ),
    "pyramid_c256_d12": lambda: PyramidNet(
        high_channels=160,
        mid_channels=256,
        low_channels=384,
        feature_dim=256,
        high_depth=2,
        mid_depth=4,
        low_depth=6,
        global_depth=1,
        num_heads=6,
    ),
    "pyramid_c320_d14": lambda: PyramidNet(
        high_channels=192,
        mid_channels=320,
        low_channels=512,
        feature_dim=320,
        high_depth=2,
        mid_depth=5,
        low_depth=7,
        global_depth=1,
        num_heads=8,
    ),
    "pyramid_wide_c256_d17": lambda: PyramidNet(
        high_channels=160,
        mid_channels=320,
        low_channels=512,
        feature_dim=256,
        high_depth=2,
        mid_depth=5,
        low_depth=10,
        global_depth=0,
        num_heads=8,
    ),
    "sparse_stone_c192_k64": lambda: SparseStoneNet(
        feature_dim=192,
        max_stones=64,
        stone_depth=4,
        num_heads=6,
    ),
    "sparse_stone_c192_k96": lambda: SparseStoneNet(
        feature_dim=192,
        max_stones=96,
        stone_depth=4,
        num_heads=6,
    ),
    "sparse_stone_c256_k96_d6": lambda: SparseStoneNet(
        feature_dim=256,
        max_stones=96,
        stone_depth=6,
        num_heads=8,
    ),
    "sparse_stone_c320_k96_d8": lambda: SparseStoneNet(
        feature_dim=320,
        max_stones=96,
        stone_depth=8,
        num_heads=10,
    ),
}


def build_architecture(name):
    try:
        factory = ARCHITECTURES[name]
    except KeyError as error:
        choices = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unknown architecture {name!r}; choose from {choices}") from error
    model = factory()
    model.architecture_name = name
    return model
