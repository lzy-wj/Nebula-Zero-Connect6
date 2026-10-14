"""Measure V4 Dense architectural stages with fixed CUDA tensors."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.nebula_v4.benchmark import make_inputs
from experiments.pair_native.model import load_checkpoint


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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--stones", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model, checkpoint = load_checkpoint(args.checkpoint)
    model = model.eval().to(device)
    inputs = make_inputs(args.batch, args.stones, device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        dense_map = model.stem(inputs)
        points = model.point_projection(dense_map.flatten(2).transpose(1, 2))
        points = points + model.positional_features()[None]
        summaries = model.compressor(
            model.summary_queries.expand(args.batch, -1, -1), points
        )
        block_outputs = summaries
        for block in model.blocks:
            block_outputs = block(block_outputs)
        decoded = model.decoder(points, block_outputs)

    def stem():
        return model.stem(inputs)

    def projection_position():
        projected = model.point_projection(dense_map.flatten(2).transpose(1, 2))
        return projected + model.positional_features()[None]

    def compressor():
        return model.compressor(
            model.summary_queries.expand(args.batch, -1, -1), points
        )

    def blocks():
        outputs = summaries
        for block in model.blocks:
            outputs = block(outputs)
        return outputs

    def decoder():
        return model.decoder(points, block_outputs)

    def policy_heads():
        return model.unary_head(decoded), model.pair_head(decoded)

    def value_head():
        pooled = torch.cat(
            (block_outputs.mean(dim=1), block_outputs.amax(dim=1)), dim=-1
        )
        return model.value_head(pooled)

    stages = {
        "stem_ms": elapsed_ms(stem, args.warmup, args.iterations),
        "projection_and_position_ms": elapsed_ms(
            projection_position, args.warmup, args.iterations
        ),
        "compressor_361_to_summary_ms": elapsed_ms(
            compressor, args.warmup, args.iterations
        ),
        "summary_transformer_blocks_ms": elapsed_ms(
            blocks, args.warmup, args.iterations
        ),
        "decoder_summary_to_361_ms": elapsed_ms(
            decoder, args.warmup, args.iterations
        ),
        "unary_and_pair_heads_ms": elapsed_ms(
            policy_heads, args.warmup, args.iterations
        ),
        "value_head_ms": elapsed_ms(value_head, args.warmup, args.iterations),
    }
    stage_sum = sum(stages.values())
    result = {
        "device": torch.cuda.get_device_name(device),
        "model_config": checkpoint["model_config"],
        "batch": args.batch,
        "stones": args.stones,
        **stages,
        "isolated_stage_sum_ms": stage_sum,
        "share_percent": {
            key.removesuffix("_ms"): 100.0 * value / stage_sum
            for key, value in stages.items()
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
