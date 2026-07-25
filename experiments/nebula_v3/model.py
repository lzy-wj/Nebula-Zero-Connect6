"""面向六子棋逐子 MCTS 的高效策略价值网络。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE


class DropPath(nn.Module):
    """按样本随机丢弃残差分支，推理阶段完全等价于恒等缩放。"""

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, inputs):
        if not self.training or self.probability == 0.0:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        random_tensor.floor_()
        return inputs * random_tensor / keep_probability


class LayerNorm2d(nn.Module):
    """对每个棋盘点的通道做归一化，不依赖 batch 运行统计。"""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        # ONNX/TensorRT 要求 LayerNorm 的归一化维度是编译期常量，不能从
        # 动态 batch 图里的 shape 节点回推。
        self.channels = int(channels)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs):
        channels_last = inputs.permute(0, 2, 3, 1)
        channels_last = F.layer_norm(
            channels_last,
            (self.channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        return channels_last.permute(0, 3, 1, 2)


class ConvNeXtBoardBlock(nn.Module):
    """大核深度卷积提取横竖斜局部棋形，再用逐点 MLP 融合通道。"""

    def __init__(self, channels, expansion=4, drop_path=0.0):
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, hidden)
        self.project = nn.Linear(hidden, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = DropPath(drop_path)

    def forward(self, inputs):
        residual = inputs
        outputs = self.depthwise(inputs).permute(0, 2, 3, 1)
        outputs = self.norm(outputs)
        outputs = self.expand(outputs)
        outputs = F.gelu(outputs, approximate='tanh')
        outputs = self.project(outputs)
        outputs = outputs * self.layer_scale
        outputs = outputs.permute(0, 3, 1, 2)
        return residual + self.drop_path(outputs)


def rotate_half(inputs):
    """把相邻偶奇维组成二维向量并旋转 90 度。"""
    even = inputs[..., 0::2]
    odd = inputs[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding2D(nn.Module):
    """二维 RoPE：一半维度编码行，另一半编码列。"""

    def __init__(self, head_dim, board_size=BOARD_SIZE, theta=10_000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError('注意力 head_dim 必须能被 4 整除，以便拆分行列 RoPE')

        axis_dim = head_dim // 2
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        )
        positions = torch.arange(board_size, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        cosine = angles.cos().repeat_interleave(2, dim=-1)
        sine = angles.sin().repeat_interleave(2, dim=-1)

        rows = torch.arange(board_size).repeat_interleave(board_size)
        columns = torch.arange(board_size).repeat(board_size)
        self.register_buffer(
            'row_cosine',
            cosine[rows].view(1, 1, BOARD_POINTS, axis_dim),
            persistent=False,
        )
        self.register_buffer(
            'row_sine',
            sine[rows].view(1, 1, BOARD_POINTS, axis_dim),
            persistent=False,
        )
        self.register_buffer(
            'column_cosine',
            cosine[columns].view(1, 1, BOARD_POINTS, axis_dim),
            persistent=False,
        )
        self.register_buffer(
            'column_sine',
            sine[columns].view(1, 1, BOARD_POINTS, axis_dim),
            persistent=False,
        )

    def forward(self, query, key):
        query_row, query_column = query.chunk(2, dim=-1)
        key_row, key_column = key.chunk(2, dim=-1)
        row_cosine = self.row_cosine.to(dtype=query.dtype)
        row_sine = self.row_sine.to(dtype=query.dtype)
        column_cosine = self.column_cosine.to(dtype=query.dtype)
        column_sine = self.column_sine.to(dtype=query.dtype)

        query = torch.cat(
            (
                query_row * row_cosine + rotate_half(query_row) * row_sine,
                query_column * column_cosine + rotate_half(query_column) * column_sine,
            ),
            dim=-1,
        )
        key = torch.cat(
            (
                key_row * row_cosine + rotate_half(key_row) * row_sine,
                key_column * column_cosine + rotate_half(key_column) * column_sine,
            ),
            dim=-1,
        )
        return query, key


class FlashGlobalAttention(nn.Module):
    """没有稠密 bias mask 的全局注意力，可进入 PyTorch 高效 SDPA 内核。"""

    def __init__(self, channels, num_heads, board_size=BOARD_SIZE):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError('channels 必须能被 num_heads 整除')
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.rotary = RotaryEmbedding2D(self.head_dim, board_size=board_size)
        self.output = nn.Linear(channels, channels, bias=True)

    def forward(self, inputs):
        batch, tokens, channels = inputs.shape
        query, key, value = self.qkv(inputs).view(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.rotary(query, key)
        outputs = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        outputs = outputs.transpose(1, 2).reshape(batch, tokens, channels)
        return self.output(outputs)


class SwiGLU(nn.Module):
    """参数利用率更高的门控前馈层。"""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.gate_and_value = nn.Linear(channels, hidden_channels * 2)
        self.output = nn.Linear(hidden_channels, channels)

    def forward(self, inputs):
        gate, value = self.gate_and_value(inputs).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class TransformerBoardBlock(nn.Module):
    def __init__(self, channels, num_heads, drop_path=0.0):
        super().__init__()
        hidden = int(math.ceil((channels * 8 / 3) / 64) * 64)
        self.norm_attention = nn.LayerNorm(channels, eps=1e-6)
        self.attention = FlashGlobalAttention(channels, num_heads)
        self.norm_mlp = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = SwiGLU(channels, hidden)
        self.attention_scale = nn.Parameter(torch.full((channels,), 1e-4))
        self.mlp_scale = nn.Parameter(torch.full((channels,), 1e-4))
        self.drop_path = DropPath(drop_path)

    def forward(self, inputs):
        inputs = inputs + self.drop_path(
            self.attention(self.norm_attention(inputs)) * self.attention_scale
        )
        return inputs + self.drop_path(self.mlp(self.norm_mlp(inputs)) * self.mlp_scale)


class NebulaNetV3(nn.Module):
    """单策略头 + 单价值头，与现有逐子 MCTS 的真实消费方式保持一致。"""

    def __init__(
        self,
        input_planes=5,
        channels=320,
        conv_depth=8,
        transformer_depth=4,
        num_heads=10,
        drop_path_rate=0.08,
    ):
        super().__init__()
        self.input_planes = input_planes
        self.channels = channels
        self.board_size = BOARD_SIZE
        self.num_points = BOARD_POINTS
        self.model_config = {
            'input_planes': input_planes,
            'channels': channels,
            'conv_depth': conv_depth,
            'transformer_depth': transformer_depth,
            'num_heads': num_heads,
            'drop_path_rate': drop_path_rate,
        }

        total_depth = conv_depth + transformer_depth
        drop_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=True),
            LayerNorm2d(channels),
        )
        self.spatial_blocks = nn.Sequential(*[
            ConvNeXtBoardBlock(channels, drop_path=drop_rates[index])
            for index in range(conv_depth)
        ])
        self.global_blocks = nn.ModuleList([
            TransformerBoardBlock(
                channels,
                num_heads,
                drop_path=drop_rates[conv_depth + index],
            )
            for index in range(transformer_depth)
        ])
        self.final_norm = nn.LayerNorm(channels, eps=1e-6)

        self.policy_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(approximate='tanh'),
            nn.Linear(channels // 2, 1),
        )
        self.value_attention = nn.Linear(channels, 1)
        self.value_head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, LayerNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, inputs):
        if inputs.shape[1] != self.input_planes:
            raise ValueError(
                f'NebulaNetV3 需要 {self.input_planes} 个输入平面，实际收到 {inputs.shape[1]}'
            )
        features = self.spatial_blocks(self.stem(inputs))
        tokens = features.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)

    def forward(self, inputs, move1_idx=None):
        del move1_idx  # 保留兼容签名；逐子 MCTS 不需要未训练的第二策略头。
        tokens = self.forward_features(inputs)
        policy_logits = self.policy_head(tokens).squeeze(-1)

        attention = torch.softmax(self.value_attention(tokens).squeeze(-1), dim=1)
        salient = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
        average = tokens.mean(dim=1)
        value = self.value_head(torch.cat((salient, average), dim=-1))
        return policy_logits, None, value

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


if __name__ == '__main__':
    model = NebulaNetV3()
    sample = torch.randn(2, 5, BOARD_SIZE, BOARD_SIZE)
    policy, _, value = model(sample)
    print(f'参数量: {model.parameter_count:,}')
    print(f'策略形状: {tuple(policy.shape)} | 价值形状: {tuple(value.shape)}')
