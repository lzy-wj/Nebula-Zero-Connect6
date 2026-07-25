"""把 NebulaNet V3 导出为接收原始棋盘的 TensorRT 友好 ONNX。"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import NebulaNetV3
from model_fast import FastC6NetV4


class FusedV3SelfPlayWrapper(nn.Module):
    """在图内完成棋盘编码、合法点 mask、softmax 与价值规则修正。"""

    def __init__(self, model, compute_dtype=torch.float16):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        kernels = torch.zeros((4, 1, 6, 6), dtype=torch.float32)
        kernels[0, 0, 0, :] = 1
        kernels[1, 0, :, 0] = 1
        kernels[2, 0] = torch.eye(6)
        kernels[3, 0] = torch.rot90(torch.eye(6), 1, [0, 1])
        self.register_buffer('threat_kernels', kernels)

    def detect_threats(self, stones, blockers):
        padded_stones = F.pad(stones, (5, 5, 5, 5), value=0)
        counts = F.conv2d(padded_stones, self.threat_kernels)
        padded_blockers = F.pad(blockers, (5, 5, 5, 5), value=1)
        blocked = F.conv2d(padded_blockers, self.threat_kernels)
        counts = counts * blocked.eq(0).float()
        return (
            counts.ge(6).float().flatten(1).amax(1),
            counts.eq(5).float().flatten(1).amax(1),
            counts.eq(4).float().flatten(1).amax(1),
        )

    def forward(self, board):
        board = torch.where(board == 2, -torch.ones_like(board), board)
        stone_count = board.ne(0).sum(dim=(1, 2))
        rank = (stone_count + 1) // 2
        current_player = torch.where(rank.remainder(2).eq(1), -1, 1)
        player_view = current_player.view(-1, 1, 1)

        self_stones = board.eq(player_view).float().unsqueeze(1)
        opponent_stones = board.eq(-player_view).float().unsqueeze(1)
        empty = board.eq(0).float().unsqueeze(1)
        black_plane = current_player.eq(1).float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)
        second_stone = (
            stone_count.gt(0) & stone_count.remainder(2).eq(0)
        ).float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)
        features = torch.cat(
            (self_stones, opponent_stones, empty, black_plane, second_stone),
            dim=1,
        )

        policy_logits, _, value = self.model(features.to(self.compute_dtype))
        policy_logits = policy_logits.float().masked_fill(empty.flatten(1).eq(0), -10_000.0)
        policy = torch.softmax(policy_logits, dim=1)
        value = value.float().flatten()

        my_win, my_five, my_four = self.detect_threats(self_stones, opponent_stones)
        opponent_win, opponent_five, opponent_four = self.detect_threats(
            opponent_stones,
            self_stones,
        )
        stones_remaining = torch.where(
            stone_count.gt(0) & stone_count.remainder(2).eq(1),
            2,
            1,
        )
        value = torch.where(opponent_four.bool(), value.clamp(max=-0.2), value)
        value = torch.where(opponent_five.bool(), value.clamp(max=-0.25), value)
        can_win = my_five.bool() | (my_four.bool() & stones_remaining.ge(2))
        value = torch.where(can_win, torch.ones_like(value), value)
        value = torch.where(opponent_win.bool(), -torch.ones_like(value), value)
        value = torch.where(my_win.bool(), torch.ones_like(value), value)
        return policy, value.unsqueeze(1)


def checkpoint_state(checkpoint):
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    return {key.removeprefix('module.'): value for key, value in state.items()}


def build_checkpoint_model(checkpoint):
    """按 checkpoint 中记录的架构恢复候选网络。"""

    architecture = checkpoint.get('architecture', 'NebulaNetV3')
    model_classes = {
        'NebulaNetV3': NebulaNetV3,
        'FastC6NetV4': FastC6NetV4,
    }
    if architecture not in model_classes:
        raise ValueError(f'不支持导出的网络架构: {architecture}')
    model = model_classes[architecture](**checkpoint['model_config'])
    model.load_state_dict(checkpoint_state(checkpoint))
    return model


def main():
    parser = argparse.ArgumentParser(description='导出 NebulaNet V3 ONNX')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_checkpoint_model(checkpoint)
    compute_dtype = torch.float16 if args.precision == 'fp16' else torch.bfloat16
    model = model.to(device=device, dtype=compute_dtype).eval()
    wrapper = FusedV3SelfPlayWrapper(model, compute_dtype).to(device).eval()
    dummy_board = torch.zeros(
        (args.batch_size, 19, 19),
        dtype=torch.int32,
        device=device,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_board,),
        args.output,
        opset_version=18,
        do_constant_folding=True,
        input_names=['board'],
        output_names=['policy1', 'value'],
        dynamic_axes={
            'board': {0: 'batch_size'},
            'policy1': {0: 'batch_size'},
            'value': {0: 'batch_size'},
        },
        dynamo=False,
    )
    print(f'ONNX 已保存: {args.output}')


if __name__ == '__main__':
    main()
