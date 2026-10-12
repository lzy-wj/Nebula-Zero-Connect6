"""Benchmark pair-native inference including pair scoring and GPU Top-K."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.pair_native.model import BOARD_POINTS, PairNativeSparseNet, parameter_count


def make_inputs(batch: int, stones: int, device: torch.device) -> torch.Tensor:
    inputs = torch.zeros((batch, 5, 19, 19), dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(2026 + batch * 1009 + stones)
    random_scores = torch.rand((batch, BOARD_POINTS), generator=generator, device=device)
    indices = random_scores.topk(min(stones, BOARD_POINTS), dim=1).indices
    flat = inputs.flatten(2)
    split = (indices.shape[1] + 1) // 2
    flat[:, 0].scatter_(1, indices[:, :split], 1.0)
    flat[:, 1].scatter_(1, indices[:, split:], 1.0)
    flat[:, 2] = 1.0 - flat[:, 0] - flat[:, 1]
    flat[:, 3].fill_(1.0)
    return inputs


def benchmark_batch(model, inputs, top_k, dtype, warmup, iterations, copy_outputs):
    def run_once():
        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model.forward_topk(inputs, top_k=top_k)
        if copy_outputs:
            return tuple(output.detach().cpu() for output in outputs[:4])
        return outputs

    with torch.inference_mode():
        for _ in range(warmup):
            run_once()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(iterations):
            run_once()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    latency_ms = elapsed * 1000.0 / iterations
    return {
        "batch": int(inputs.shape[0]),
        "latency_ms": latency_ms,
        "positions_per_second": inputs.shape[0] * 1000.0 / latency_ms,
        "topk_pairs_per_second": inputs.shape[0] * top_k * 1000.0 / latency_ms,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,16,32,64,128")
    parser.add_argument("--stones", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--pair-rank", type=int, default=32)
    parser.add_argument("--dense-channels", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--copy-outputs", action="store_true")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.bfloat16
    model = PairNativeSparseNet(
        feature_dim=arguments.feature_dim,
        max_tokens=arguments.max_tokens,
        depth=arguments.depth,
        num_heads=arguments.heads,
        pair_rank=arguments.pair_rank,
        dense_channels=arguments.dense_channels,
    ).eval().to(device)
    metadata = {
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "dtype": arguments.dtype,
        "parameters": parameter_count(model),
        "stones": arguments.stones,
        "top_k": arguments.top_k,
        "copy_outputs": arguments.copy_outputs,
    }
    results = []
    for batch in (int(item) for item in arguments.batches.split(",") if item.strip()):
        inputs = make_inputs(batch, arguments.stones, device)
        result = benchmark_batch(
            model,
            inputs,
            arguments.top_k,
            dtype,
            arguments.warmup,
            arguments.iterations,
            arguments.copy_outputs,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"metadata": metadata, "results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
