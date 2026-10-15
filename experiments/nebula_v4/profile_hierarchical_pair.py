"""Compare flat Pair Top-K with a hierarchical first/second candidate path."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.nebula_v4.benchmark import make_inputs
from experiments.pair_native.arena import load_arena_checkpoint


def elapsed_ms(function, warmup: int, iterations: int) -> float:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
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
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--stones", type=int, default=96)
    parser.add_argument("--first-candidates", type=int, default=16)
    parser.add_argument("--seconds-per-first", type=int, default=4)
    parser.add_argument("--gumbel", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    score_model, checkpoint = load_arena_checkpoint(args.checkpoint)
    score_model = score_model.eval().to(device)
    model = torch.compile(score_model, fullgraph=True, mode="reduce-overhead")
    inputs = make_inputs(args.batch, args.stones, device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        unary, embeddings, value, occupied = (
            output.clone() for output in model(inputs)
        )

    full_noise = torch.randn(
        args.batch, 361, 361, device=device, dtype=torch.float16
    )
    first_noise = torch.randn(
        args.batch, 361, device=device, dtype=torch.float16
    )

    def flat_candidates():
        scores = score_model.pair_scores(unary, embeddings, occupied)
        return torch.topk(
            (scores + args.gumbel * full_noise).flatten(1),
            args.first_candidates * args.seconds_per_first,
            dim=1,
        )

    def first_candidates():
        first_logits = unary.masked_fill(
            occupied,
            torch.finfo(unary.dtype).min,
        )
        return torch.topk(
            first_logits + args.gumbel * first_noise,
            args.first_candidates,
            dim=1,
        ).indices

    with torch.inference_mode():
        selected_first = first_candidates()
        second_noise = torch.randn(
            args.batch,
            args.first_candidates,
            361,
            device=device,
            dtype=torch.float16,
        )

    def conditional_rows():
        return score_model.pair_row_scores(
            unary,
            embeddings,
            occupied,
            selected_first,
        )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        row_scores = conditional_rows()

    def second_candidates():
        return torch.topk(
            row_scores + args.gumbel * second_noise,
            args.seconds_per_first,
            dim=2,
        )

    flat_ms = elapsed_ms(flat_candidates, args.warmup, args.iterations)
    first_ms = elapsed_ms(first_candidates, args.warmup, args.iterations)
    rows_ms = elapsed_ms(conditional_rows, args.warmup, args.iterations)
    second_ms = elapsed_ms(second_candidates, args.warmup, args.iterations)
    hierarchical_ms = first_ms + rows_ms + second_ms
    print(json.dumps({
        "device": torch.cuda.get_device_name(device),
        "model_config": checkpoint["model_config"],
        "batch": args.batch,
        "first_candidates": args.first_candidates,
        "seconds_per_first": args.seconds_per_first,
        "returned_pairs": args.first_candidates * args.seconds_per_first,
        "flat_pair_candidates_ms": flat_ms,
        "hierarchical_first_ms": first_ms,
        "hierarchical_conditional_rows_ms": rows_ms,
        "hierarchical_second_ms": second_ms,
        "hierarchical_total_ms": hierarchical_ms,
        "candidate_stage_speedup": flat_ms / hierarchical_ms,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
