"""轻量条件第二子策略头。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def normalize_tokens(features, feature_dim=256):
    """按棋盘点归一化，抑制旧主干中幅度很大的少量通道。"""

    # ONNX 旧导出器要求 normalized_shape 是编译期常量；本项目特征维固定为 256。
    return F.layer_norm(features, (feature_dim,))


def safe_atanh(values):
    """使用 ONNX/TensorRT 都支持的基础 Log 算子实现稳定 atanh。"""

    # BF16 中 0.999 会舍入为 1.0，必须先转 FP32 再截断，否则 log(1-x) 为 -inf。
    values = values.float().flatten().clamp(-0.999, 0.999)
    return 0.5 * (torch.log(1.0 + values) - torch.log(1.0 - values))


class PairPolicyHead(nn.Module):
    """用第一落点特征、候选点特征和相对位移预测第二颗子。"""

    def __init__(
        self,
        feature_dim=256,
        rank=64,
        board_size=19,
        tied_factors=False,
        projection_hidden=0,
        relative_gating=False,
    ):
        super().__init__()
        self.board_size = board_size
        self.rank = rank
        self.tied_factors = tied_factors
        self.projection_hidden = projection_hidden
        self.relative_gating = relative_gating

        def make_projection():
            if projection_hidden > 0:
                return nn.Sequential(
                    nn.Linear(feature_dim, projection_hidden),
                    nn.GELU(),
                    nn.Linear(projection_hidden, rank, bias=False),
                )
            return nn.Linear(feature_dim, rank, bias=False)

        self.candidate_projection = make_projection()
        self.first_move_projection = (
            self.candidate_projection
            if tied_factors
            else make_projection()
        )
        self.relative_bias = nn.Parameter(
            torch.zeros((board_size * 2 - 1) ** 2)
        )
        self.base_scale = nn.Parameter(torch.ones(()))
        self.relative_gate = (
            nn.Parameter(
                torch.ones((board_size * 2 - 1) ** 2, rank)
            )
            if relative_gating
            else None
        )

        positions = torch.arange(board_size * board_size)
        rows = positions // board_size
        columns = positions % board_size
        self.register_buffer("candidate_rows", rows, persistent=False)
        self.register_buffer("candidate_columns", columns, persistent=False)

        # 初始时主要复用 parent policy1，再逐渐学习成对落子的兼容性。
        candidate_output = (
            self.candidate_projection[-1]
            if projection_hidden > 0
            else self.candidate_projection
        )
        nn.init.normal_(candidate_output.weight, std=0.02)
        if not tied_factors:
            first_output = (
                self.first_move_projection[-1]
                if projection_hidden > 0
                else self.first_move_projection
            )
            nn.init.normal_(first_output.weight, std=0.02)

    def forward(self, features, first_move, parent_policy_logits):
        features = normalize_tokens(features)
        batch_indices = torch.arange(features.shape[0], device=features.device)
        first_features = features[batch_indices, first_move]
        candidate_factors = self.candidate_projection(features)
        first_factors = self.first_move_projection(first_features)
        first_rows = first_move // self.board_size
        first_columns = first_move % self.board_size
        relative_rows = self.candidate_rows.unsqueeze(0) - first_rows.unsqueeze(1)
        relative_columns = self.candidate_columns.unsqueeze(0) - first_columns.unsqueeze(1)
        relative_index = (
            (relative_rows + self.board_size - 1) * (self.board_size * 2 - 1)
            + relative_columns
            + self.board_size
            - 1
        )
        pair_products = candidate_factors * first_factors.unsqueeze(1)
        if self.relative_gate is not None:
            pair_products = pair_products * self.relative_gate[relative_index]
        compatibility = pair_products.sum(dim=2) / math.sqrt(self.rank)
        return (
            self.base_scale * parent_policy_logits
            + compatibility
            + self.relative_bias[relative_index]
        )


class PairValueHead(nn.Module):
    """预测第一颗子落下后的条件价值 q(s, a1)。"""

    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 初始严格退化为 parent value，训练只学习动作带来的残差。
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, features, first_move, parent_value):
        features = normalize_tokens(features)
        batch_indices = torch.arange(features.shape[0], device=features.device)
        first_features = features[batch_indices, first_move]
        global_features = features.mean(dim=1)
        inputs = torch.cat(
            (
                global_features,
                first_features,
                first_features - global_features,
            ),
            dim=1,
        )
        residual = self.delta(inputs).squeeze(1)
        parent_logit = safe_atanh(parent_value)
        return torch.tanh(parent_logit + residual)


class VectorPairValueHead(nn.Module):
    """一次生成所有第一落点的 q(s, a)，与旧条件价值头代数等价。"""

    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.global_projection = nn.Linear(feature_dim, hidden_dim)
        self.action_projection = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def residuals_from_normalized(self, features):
        """输入已做逐点 LayerNorm 时复用，避免导出图重复归一化。"""

        global_features = features.mean(dim=1)
        hidden = (
            self.global_projection(global_features).unsqueeze(1)
            + self.action_projection(features)
        )
        return self.output(F.gelu(hidden)).squeeze(2)

    def residuals(self, features):
        return self.residuals_from_normalized(normalize_tokens(features))

    def forward_all_from_normalized(self, features, parent_value):
        residual = self.residuals_from_normalized(features)
        parent_logit = safe_atanh(parent_value)
        return torch.tanh(parent_logit.unsqueeze(1) + residual)

    def forward_all(self, features, parent_value):
        """返回每个第一落点对应的条件价值。"""

        return self.forward_all_from_normalized(
            normalize_tokens(features),
            parent_value,
        )

    def forward(self, features, first_move, parent_value):
        values = self.forward_all(features, parent_value)
        batch_indices = torch.arange(values.shape[0], device=values.device)
        return values[batch_indices, first_move]

    def load_legacy_head(self, legacy_head):
        """把 concat 形式的旧价值头严格转换成可向量化形式。"""

        first_layer = legacy_head.delta[0]
        output_layer = legacy_head.delta[2]
        feature_dim = first_layer.weight.shape[1] // 3
        global_weight, first_weight, difference_weight = first_layer.weight.split(
            feature_dim,
            dim=1,
        )
        with torch.no_grad():
            self.global_projection.weight.copy_(global_weight - difference_weight)
            self.global_projection.bias.copy_(first_layer.bias)
            self.action_projection.weight.copy_(first_weight + difference_weight)
            self.output.weight.copy_(output_layer.weight)
            self.output.bias.copy_(output_layer.bias)
        return self
