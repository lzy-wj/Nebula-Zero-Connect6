"""导出带成对落子辅助输出的融合自对弈 ONNX，仅供候选架构基准。"""

import argparse
import os
import sys


os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    os.environ.get("NEBULA_BUILD_GPU", "6"),
)

import torch
import torch.nn.functional as F


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RL_DIR = os.path.join(ROOT, "reinforcement_learning")
sys.path.insert(0, RL_DIR)
sys.path.insert(0, os.path.dirname(__file__))

import config
from core.model import C6TransNet
from pipeline.export_onnx import FusedSelfPlayExportWrapper
from model import PairPolicyHead, PairValueHead, VectorPairValueHead, normalize_tokens


class PairFusedExportWrapper(FusedSelfPlayExportWrapper):
    """完整评估时顺便生成第二子低秩因子和全部条件价值。"""

    def __init__(self, model, pair_policy, pair_value, compute_dtype):
        super().__init__(model, compute_dtype)
        self.pair_policy = pair_policy
        self.pair_value = pair_value

    def forward(self, board):
        # 与生产融合图保持同一棋盘编码，避免候选引擎的主输出发生变化。
        board = torch.where(board == 2, -torch.ones_like(board), board)
        stone_count = board.ne(0).sum(dim=(1, 2))
        rank = (stone_count + 1) // 2
        current_player = torch.where(rank.remainder(2).eq(1), -1, 1)
        player_view = current_player.view(-1, 1, 1)

        self_stones = board.eq(player_view).float().unsqueeze(1)
        opponent_stones = board.eq(-player_view).float().unsqueeze(1)
        empty_planes = torch.zeros(
            (board.shape[0], 14, 19, 19),
            dtype=torch.float32,
            device=board.device,
        )
        color_plane = current_player.eq(1).float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)
        inputs = torch.cat(
            (self_stones, opponent_stones, empty_planes, color_plane),
            dim=1,
        ).to(self.compute_dtype)

        features_2d = self.model.relu(self.model.bn_in(self.model.conv_in(inputs)))
        features_2d = self.model.res_stack(features_2d)
        features = self.model.forward_transformer(features_2d.flatten(2).transpose(1, 2))
        raw_value = self.model.value_head(features.mean(dim=1)).flatten()
        policy_logits = self.model.head_move1(features).squeeze(-1)

        # 三个辅助输出共享一次无参数 LayerNorm；主 policy/value 保持原图数值路径。
        normalized = normalize_tokens(features)
        candidate_factors = self.pair_policy.candidate_projection(normalized)
        first_factors = self.pair_policy.first_move_projection(normalized)
        pair_values = self.pair_value.forward_all_from_normalized(normalized, raw_value)

        policy = torch.softmax(policy_logits.float(), dim=1)
        value = raw_value.float()
        my_win, my_c5, my_c4 = self._detect_threats(self_stones, opponent_stones)
        opp_win, opp_c5, opp_c4 = self._detect_threats(opponent_stones, self_stones)
        stones_remaining = torch.where(
            stone_count.gt(0) & stone_count.remainder(2).eq(1),
            2,
            1,
        )
        value = torch.where(opp_c4.bool(), value.clamp(max=-0.2), value)
        value = torch.where(opp_c5.bool(), value.clamp(max=-0.25), value)
        can_win = my_c5.bool() | (my_c4.bool() & stones_remaining.ge(2))
        value = torch.where(can_win, torch.ones_like(value), value)
        value = torch.where(opp_win.bool(), -torch.ones_like(value), value)
        value = torch.where(my_win.bool(), torch.ones_like(value), value)

        return (
            policy,
            value.unsqueeze(1),
            candidate_factors,
            first_factors,
            pair_values.to(self.compute_dtype),
        )


def load_main_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    model = C6TransNet(input_planes=17).to(device)
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}
    )
    return model.eval()


def export_pair_onnx(checkpoint_path, pair_heads_path, output_path, batch_size):
    device = torch.device("cuda")
    saved_heads = torch.load(pair_heads_path, map_location=device, weights_only=True)
    rank = int(saved_heads["args"]["rank"])
    if saved_heads["args"].get("tied_factors", False):
        raise ValueError("增强引擎原型要求非共享因子")

    main_model = load_main_model(checkpoint_path, device)
    pair_policy = PairPolicyHead(
        rank=rank,
        projection_hidden=int(
            saved_heads["args"].get("projection_hidden", 0)
        ),
        relative_gating=bool(
            saved_heads["args"].get("relative_gating", False)
        ),
    ).to(device).eval()
    pair_policy.load_state_dict(saved_heads["pair_head"])
    legacy_value = PairValueHead().to(device).eval()
    legacy_value.load_state_dict(saved_heads["pair_value_head"])
    pair_value = VectorPairValueHead().to(device).eval().load_legacy_head(legacy_value)

    precision_to_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    compute_dtype = precision_to_dtype[config.TRT_PRECISION]
    main_model = main_model.to(dtype=compute_dtype)
    pair_policy = pair_policy.to(dtype=compute_dtype)
    pair_value = pair_value.to(dtype=compute_dtype)
    wrapper = PairFusedExportWrapper(
        main_model,
        pair_policy,
        pair_value,
        compute_dtype,
    ).to(device).eval()

    dummy_board = torch.zeros(
        (batch_size, 19, 19),
        dtype=torch.int32,
        device=device,
    )
    output_names = [
        "policy1",
        "value",
        "pair_candidate",
        "pair_first",
        "pair_value",
    ]
    dynamic_axes = {"board": {0: "batch_size"}}
    dynamic_axes.update({name: {0: "batch_size"} for name in output_names})
    torch.onnx.export(
        wrapper,
        (dummy_board,),
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["board"],
        output_names=output_names,
        dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    print(
        f"已导出候选增强图: {output_path} | rank={rank} | "
        f"precision={config.TRT_PRECISION}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("pair_heads")
    parser.add_argument("output")
    parser.add_argument("--batch-size", type=int, default=config.MCTS_BATCH_SIZE)
    args = parser.parse_args()
    export_pair_onnx(
        args.checkpoint,
        args.pair_heads,
        args.output,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
