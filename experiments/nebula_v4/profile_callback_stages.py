"""Isolate evaluator input, network, Pair selection, and output-copy costs."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from experiments.pair_native.arena import load_arena_checkpoint
from experiments.pair_native.model import BOARD_POINTS
from experiments.pair_native.pair_mcts import TorchPairEvaluator


class GumbelSelection(torch.nn.Module):
    def __init__(self, model, top_k: int, scale: float):
        super().__init__()
        self.model = model
        self.top_k = int(top_k)
        self.scale = float(scale)

    def forward(self, unary, embeddings, occupied, uniform):
        scores = self.model.pair_scores(unary, embeddings, occupied)
        noise = -torch.log(-torch.log(uniform)) * self.scale
        scores = scores + noise.to(scores.dtype)
        top_scores, flat_pair = torch.topk(scores.flatten(1), self.top_k, dim=1)
        first = flat_pair.div(BOARD_POINTS, rounding_mode="floor")
        second = flat_pair % BOARD_POINTS
        return first, second, torch.softmax(top_scores.float(), dim=1)


def elapsed_ms(function, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--stones", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model, checkpoint = load_arena_checkpoint(args.checkpoint)
    evaluator = TorchPairEvaluator(
        model,
        device,
        top_k=args.top_k,
        candidate_gumbel=1.0e-12,
        compile_model=True,
        deduplicate=False,
        value_scale=1.0,
    )
    score_model = getattr(evaluator.model, "_orig_mod", evaluator.model)

    boards_numpy = np.zeros((args.batch, BOARD_POINTS), dtype=np.int8)
    for row in range(args.batch):
        indices = (np.arange(args.stones) * 37 + row * 13) % BOARD_POINTS
        boards_numpy[row, indices[0::2]] = 1
        boards_numpy[row, indices[1::2]] = -1
    players_numpy = np.where(np.arange(args.batch) % 2, -1, 1).astype(np.int8)

    def input_stage():
        boards = torch.from_numpy(boards_numpy).to(device)
        players = torch.from_numpy(players_numpy).to(device)
        return evaluator._features(boards, players)

    features = input_stage()

    def trunk_stage():
        with torch.autocast("cuda", dtype=torch.float16):
            return evaluator.model(features)

    with torch.inference_mode():
        unary, embeddings, values, occupied = (
            output.clone() for output in trunk_stage()
        )

    def pair_stage():
        with torch.autocast("cuda", dtype=torch.float16):
            return score_model.pair_scores(unary, embeddings, occupied)

    with torch.inference_mode():
        pair_scores = pair_stage()

    def gumbel_stage():
        uniform = torch.rand(
            pair_scores.shape,
            device=device,
            dtype=torch.float32,
            generator=evaluator.generator,
        ).clamp_(1.0e-6, 1.0 - 1.0e-6)
        return pair_scores + (-torch.log(-torch.log(uniform)) * 0.25).to(pair_scores.dtype)

    def gumbel_inplace_stage():
        uniform = torch.empty(
            pair_scores.shape,
            device=device,
            dtype=torch.float32,
        ).uniform_(generator=evaluator.generator)
        uniform.clamp_(1.0e-6, 1.0 - 1.0e-6)
        uniform.log_().neg_().log_().neg_().mul_(0.25)
        return pair_scores.add_(uniform.to(pair_scores.dtype))

    with torch.inference_mode():
        noisy_scores = gumbel_stage()

    def topk_stage():
        top_scores, flat_pair = torch.topk(
            noisy_scores.flatten(1), args.top_k, dim=1
        )
        first = flat_pair.div(BOARD_POINTS, rounding_mode="floor")
        second = flat_pair % BOARD_POINTS
        priors = torch.softmax(top_scores.float(), dim=1)
        return first, second, priors

    with torch.inference_mode():
        first, second, priors = topk_stage()

    compiled_selection = torch.compile(
        GumbelSelection(score_model, args.top_k, 0.25).eval(),
        fullgraph=True,
        mode="reduce-overhead",
    )

    def compiled_selection_stage():
        uniform = torch.rand(
            pair_scores.shape,
            device=device,
            dtype=torch.float32,
            generator=evaluator.generator,
        ).clamp_(1.0e-6, 1.0 - 1.0e-6)
        return compiled_selection(unary, embeddings, occupied, uniform)

    def output_stage():
        return (
            first.to(dtype=torch.int16).cpu().numpy(),
            second.to(dtype=torch.int16).cpu().numpy(),
            priors.cpu().numpy(),
            values.float().cpu().numpy(),
        )

    result = {
        "device": torch.cuda.get_device_name(device),
        "model_config": checkpoint["model_config"],
        "batch": args.batch,
        "stones": args.stones,
        "top_k": args.top_k,
        "input_h2d_and_features_ms": elapsed_ms(input_stage, args.warmup, args.iterations),
        "compiled_network_ms": elapsed_ms(trunk_stage, args.warmup, args.iterations),
        "pair_scores_ms": elapsed_ms(pair_stage, args.warmup, args.iterations),
        "full_matrix_gumbel_ms": elapsed_ms(gumbel_stage, args.warmup, args.iterations),
        "full_matrix_gumbel_inplace_ms": elapsed_ms(
            gumbel_inplace_stage, args.warmup, args.iterations
        ),
        "topk_softmax_ms": elapsed_ms(topk_stage, args.warmup, args.iterations),
        "compiled_pair_gumbel_topk_ms": elapsed_ms(
            compiled_selection_stage, args.warmup, args.iterations
        ),
        "compact_d2h_ms": elapsed_ms(output_stage, args.warmup, args.iterations),
    }
    result["isolated_stage_sum_ms"] = sum(
        result[key]
        for key in (
            "input_h2d_and_features_ms",
            "compiled_network_ms",
            "pair_scores_ms",
            "full_matrix_gumbel_ms",
            "topk_softmax_ms",
            "compact_d2h_ms",
        )
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
