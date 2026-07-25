"""从生产 V2 蒸馏而来的低延迟策略价值网络。"""

import copy
import os
import sys

import torch
import torch.nn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RL_DIR = os.path.join(PROJECT_ROOT, 'reinforcement_learning')
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from core.model import RelativeGlobalAttention, SEBlock


BOARD_SIZE = 19
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE


class FusedDilatedResBlock(nn.Module):
    """去掉 BatchNorm 的 V2 残差块；初始化时把 BN 精确折入卷积。"""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.se = SEBlock(channels)

    def forward(self, inputs):
        outputs = self.relu(self.conv1(inputs))
        outputs = self.se(self.conv2(outputs))
        return self.relu(outputs + inputs)


def fold_batch_norm_into_conv(destination, source_conv, source_bn):
    """把 eval 模式 BatchNorm 等价折叠到带 bias 的卷积中。"""

    with torch.no_grad():
        scale = source_bn.weight / torch.sqrt(source_bn.running_var + source_bn.eps)
        source_bias = source_conv.bias
        if source_bias is None:
            source_bias = torch.zeros_like(source_bn.running_mean)
        destination.weight.copy_(source_conv.weight * scale[:, None, None, None])
        destination.bias.copy_(
            (source_bias - source_bn.running_mean) * scale + source_bn.bias
        )


class FastC6NetV4(nn.Module):
    """保留 V2 强策略知识、裁剪全局层并移除无效第二策略头。"""

    def __init__(
        self,
        input_planes=5,
        embed_dim=256,
        num_heads=8,
        source_block_indices=(0, 2, 4, 5),
        bridge_attention_source=None,
    ):
        super().__init__()
        source_block_indices = tuple(int(index) for index in source_block_indices)
        if not source_block_indices:
            raise ValueError('source_block_indices 不能为空')
        if len(set(source_block_indices)) != len(source_block_indices):
            raise ValueError('source_block_indices 不能重复')
        if bridge_attention_source is not None:
            bridge_attention_source = int(bridge_attention_source)
            if bridge_attention_source in source_block_indices:
                raise ValueError('桥接注意力层不能与完整保留层重复')

        self.input_planes = int(input_planes)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.source_block_indices = source_block_indices
        self.bridge_attention_source = bridge_attention_source
        self.model_config = {
            'input_planes': self.input_planes,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'source_block_indices': self.source_block_indices,
            'bridge_attention_source': self.bridge_attention_source,
        }

        # BN 已折入 stem 和残差卷积，训练时不再维护易漂移的运行统计。
        self.conv_in = nn.Conv2d(
            self.input_planes,
            self.embed_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.res_stack = nn.Sequential(
            FusedDilatedResBlock(self.embed_dim, dilation=1),
            FusedDilatedResBlock(self.embed_dim, dilation=2),
            FusedDilatedResBlock(self.embed_dim, dilation=3),
            FusedDilatedResBlock(self.embed_dim, dilation=1),
            FusedDilatedResBlock(self.embed_dim, dilation=1),
        )

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(self.embed_dim),
                'attn': RelativeGlobalAttention(
                    self.embed_dim,
                    num_heads=self.num_heads,
                ),
                'norm2': nn.LayerNorm(self.embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.embed_dim * 4, self.embed_dim),
                ),
            })
            for _ in self.source_block_indices
        ])
        if self.bridge_attention_source is not None:
            # 只恢复被裁剪层的全局信息交换，省去该层计算更重的 4 倍 FFN。
            self.bridge_norm = nn.LayerNorm(self.embed_dim)
            self.bridge_attention = RelativeGlobalAttention(
                self.embed_dim,
                num_heads=self.num_heads,
            )
            self.bridge_insert_after = sum(
                index < self.bridge_attention_source
                for index in self.source_block_indices
            )
        else:
            self.bridge_norm = None
            self.bridge_attention = None
            self.bridge_insert_after = -1
        self.policy_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def initialize_from_v2(self, source_model):
        """迁移 V2 权重，并让 5 平面输入与旧 17 平面语义初始等价。"""

        if source_model.conv_in.out_channels != self.embed_dim:
            raise ValueError('V2 embed_dim 与 FastC6NetV4 不一致')
        if max(self.source_block_indices) >= len(source_model.blocks):
            raise ValueError('待迁移的注意力层编号超出 V2 深度')
        if (
            self.bridge_attention_source is not None
            and self.bridge_attention_source >= len(source_model.blocks)
        ):
            raise ValueError('桥接注意力层编号超出 V2 深度')

        # 旧网络真正使用的只有己方、对方和黑方身份三个平面。新加入的
        # 空位与第二颗子平面从零权重开始，后续再由训练学习增量信息。
        adapted_stem = copy.deepcopy(source_model.conv_in)
        adapted_stem.weight = nn.Parameter(
            torch.zeros(
                (self.embed_dim, self.input_planes, 3, 3),
                dtype=source_model.conv_in.weight.dtype,
                device=source_model.conv_in.weight.device,
            )
        )
        with torch.no_grad():
            adapted_stem.weight[:, 0].copy_(source_model.conv_in.weight[:, 0])
            adapted_stem.weight[:, 1].copy_(source_model.conv_in.weight[:, 1])
            adapted_stem.weight[:, 3].copy_(source_model.conv_in.weight[:, 16])
        fold_batch_norm_into_conv(
            self.conv_in,
            adapted_stem,
            source_model.bn_in,
        )

        for destination, source in zip(self.res_stack, source_model.res_stack):
            fold_batch_norm_into_conv(destination.conv1, source.conv1, source.bn1)
            fold_batch_norm_into_conv(destination.conv2, source.conv2, source.bn2)
            destination.se.load_state_dict(source.se.state_dict())

        for destination, source_index in zip(
            self.blocks,
            self.source_block_indices,
        ):
            destination.load_state_dict(source_model.blocks[source_index].state_dict())
        if self.bridge_attention_source is not None:
            bridge_source = source_model.blocks[self.bridge_attention_source]
            self.bridge_norm.load_state_dict(bridge_source['norm1'].state_dict())
            self.bridge_attention.load_state_dict(bridge_source['attn'].state_dict())
        self.policy_head.load_state_dict(source_model.head_move1.state_dict())
        self.value_head.load_state_dict(source_model.value_head.state_dict())
        return self

    def forward_features(self, inputs):
        if inputs.shape[1] != self.input_planes:
            raise ValueError(
                f'FastC6NetV4 需要 {self.input_planes} 个输入平面，'
                f'实际收到 {inputs.shape[1]}'
            )
        features = self.res_stack(self.relu(self.conv_in(inputs)))
        tokens = features.flatten(2).transpose(1, 2)
        if self.bridge_insert_after == 0:
            tokens = self.bridge_attention(self.bridge_norm(tokens)) + tokens
        for block_index, block in enumerate(self.blocks, start=1):
            residual = tokens
            tokens = block['attn'](block['norm1'](tokens)) + residual
            residual = tokens
            tokens = block['mlp'](block['norm2'](tokens)) + residual
            if block_index == self.bridge_insert_after:
                tokens = self.bridge_attention(self.bridge_norm(tokens)) + tokens
        return tokens

    def forward(self, inputs, move1_idx=None, return_features=False):
        del move1_idx
        tokens = self.forward_features(inputs)
        policy_logits = self.policy_head(tokens).squeeze(-1)
        value = self.value_head(tokens.mean(dim=1))
        if return_features:
            return policy_logits, None, value, tokens
        return policy_logits, None, value

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
