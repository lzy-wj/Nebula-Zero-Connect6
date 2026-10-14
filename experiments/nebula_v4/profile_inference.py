"""Time trunk, Pair scoring, and Top-K separately on one CUDA device."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.nebula_v4.benchmark import make_inputs
from experiments.nebula_v4.model import CANDIDATE_CONFIGS, build_model
from experiments.pair_native.model import PairNativeSparseNet


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
    parser.add_argument("--model", choices=("v3", "dense", "moe"), required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--stones", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--summary-tokens", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=0)
    parser.add_argument("--pair-rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--compile-trunk", action="store_true")
    arguments = parser.parse_args()
    device = torch.device("cuda:0")
    if arguments.model == "v3":
        model = PairNativeSparseNet(
            feature_dim=192,
            max_tokens=96,
            depth=4,
            num_heads=6,
            pair_rank=32,
            latent_tokens=4,
            dense_channels=32,
        )
    else:
        config = {
            **CANDIDATE_CONFIGS[arguments.model],
            "summary_tokens": arguments.summary_tokens,
        }
        if arguments.feature_dim:
            config["feature_dim"] = arguments.feature_dim
        if arguments.num_heads:
            config["num_heads"] = arguments.num_heads
        if arguments.pair_rank:
            config["pair_rank"] = arguments.pair_rank
        model = build_model(config)
    score_model = model.eval().to(device)
    model = (
        torch.compile(score_model, fullgraph=True, mode="reduce-overhead")
        if arguments.compile_trunk
        else score_model
    )
    inputs = make_inputs(arguments.batch, arguments.stones, device)

    def trunk():
        if arguments.compile_trunk:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(inputs)

    with torch.inference_mode():
        unary, embeddings, value, occupied = (
            output.clone() for output in trunk()
        )

    def pair():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return score_model.pair_scores(unary, embeddings, occupied)

    with torch.inference_mode():
        scores = pair()

    def topk():
        return torch.topk(scores.flatten(1), arguments.top_k, dim=1)

    def complete():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            local_unary, local_embeddings, local_value, local_occupied = model(inputs)
            local_scores = score_model.pair_scores(
                local_unary,
                local_embeddings,
                local_occupied,
            )
            top_scores, flat_indices = torch.topk(
                local_scores.flatten(1),
                arguments.top_k,
                dim=1,
            )
            return flat_indices, top_scores, local_value

    result = {
        "model": arguments.model,
        "batch": arguments.batch,
        "summary_tokens": arguments.summary_tokens,
        "feature_dim": getattr(score_model, "feature_dim", None),
        "num_heads": arguments.num_heads or None,
        "pair_rank": getattr(score_model, "pair_rank", None),
        "compile_trunk": arguments.compile_trunk,
        "trunk_ms": elapsed_ms(trunk, arguments.warmup, arguments.iterations),
        "pair_ms": elapsed_ms(pair, arguments.warmup, arguments.iterations),
        "topk_ms": elapsed_ms(topk, arguments.warmup, arguments.iterations),
        "complete_ms": elapsed_ms(complete, arguments.warmup, arguments.iterations),
        "unary_dtype": str(unary.dtype),
        "pair_embedding_dtype": str(embeddings.dtype),
        "pair_score_dtype": str(scores.dtype),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
