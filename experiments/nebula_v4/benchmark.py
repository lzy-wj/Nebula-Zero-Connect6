"""Benchmark V4 inference including complete Pair scoring and GPU Top-K."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.nebula_v4.model import (
    BOARD_POINTS,
    CANDIDATE_CONFIGS,
    active_parameter_count,
    build_model,
    parameter_count,
)


def make_inputs(batch: int, stones: int, device: torch.device) -> torch.Tensor:
    inputs = torch.zeros((batch, 5, 19, 19), device=device)
    generator = torch.Generator(device=device).manual_seed(2026 + batch * 1009 + stones)
    scores = torch.rand((batch, BOARD_POINTS), generator=generator, device=device)
    indices = scores.topk(min(stones, BOARD_POINTS), dim=1).indices
    flat = inputs.flatten(2)
    split = (indices.shape[1] + 1) // 2
    flat[:, 0].scatter_(1, indices[:, :split], 1.0)
    flat[:, 1].scatter_(1, indices[:, split:], 1.0)
    flat[:, 2] = 1.0 - flat[:, 0] - flat[:, 1]
    flat[:, 3].fill_(1.0)
    return inputs


def benchmark(model, inputs, dtype, top_k, warmup, iterations, copy_outputs):
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
    latency_ms = (time.perf_counter() - started) * 1000.0 / iterations
    return {
        "batch": int(inputs.shape[0]),
        "latency_ms": latency_ms,
        "positions_per_second": inputs.shape[0] * 1000.0 / latency_ms,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dense", "moe"), required=True)
    parser.add_argument("--shared-expansion", type=int, default=0)
    parser.add_argument("--summary-tokens", type=int, default=64)
    parser.add_argument("--batches", default="1,8,16,32,64,128")
    parser.add_argument("--stones", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--copy-outputs", action="store_true")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = {
        **CANDIDATE_CONFIGS[arguments.variant],
        "summary_tokens": arguments.summary_tokens,
        "shared_expansion": arguments.shared_expansion,
    }
    device = torch.device("cuda:0")
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.bfloat16
    model = build_model(config).eval().to(device)
    metadata = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": arguments.dtype,
        "model_config": config,
        "parameters": parameter_count(model),
        "active_parameters": active_parameter_count(model),
        "stones": arguments.stones,
        "top_k": arguments.top_k,
        "copy_outputs": arguments.copy_outputs,
    }
    results = []
    for batch in (int(value) for value in arguments.batches.split(",") if value):
        inputs = make_inputs(batch, arguments.stones, device)
        result = benchmark(
            model,
            inputs,
            dtype,
            arguments.top_k,
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
