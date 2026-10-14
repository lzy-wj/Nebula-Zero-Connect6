"""Benchmark the complete Python/CUDA evaluator callback at a fixed batch."""

from __future__ import annotations

import argparse
import ctypes
import json
import time

import numpy as np
import torch

from experiments.pair_native.arena import load_arena_checkpoint
from experiments.pair_native.model import BOARD_POINTS
from experiments.pair_native.pair_mcts import TorchPairEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--stones", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--candidate-gumbel", type=float, default=0.25)
    parser.add_argument("--force-model-only-compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model, checkpoint = load_arena_checkpoint(args.checkpoint)
    construction_gumbel = args.candidate_gumbel
    if args.force_model_only_compile and construction_gumbel == 0.0:
        construction_gumbel = 1.0e-12
    evaluator = TorchPairEvaluator(
        model,
        device,
        top_k=args.top_k,
        candidate_gumbel=construction_gumbel,
        compile_model=True,
        deduplicate=False,
        pad_batch_to=0,
        batch_buckets=(),
        value_scale=1.0,
    )
    evaluator.candidate_gumbel = args.candidate_gumbel

    boards = np.zeros((args.batch, BOARD_POINTS), dtype=np.int8)
    for row in range(args.batch):
        indices = (np.arange(args.stones) * 37 + row * 13) % BOARD_POINTS
        boards[row, indices[0::2]] = 1
        boards[row, indices[1::2]] = -1
    players = np.where(np.arange(args.batch) % 2, -1, 1).astype(np.int8)
    first = np.empty((args.batch, args.top_k), dtype=np.int16)
    second = np.empty_like(first)
    priors = np.empty((args.batch, args.top_k), dtype=np.float32)
    values = np.empty(args.batch, dtype=np.float32)
    call_args = (
        args.batch,
        boards.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        players.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        first.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        second.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        priors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    for _ in range(args.warmup):
        if evaluator._callback(*call_args) != 0:
            raise RuntimeError("callback warmup failed") from evaluator.last_error
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(args.iterations):
        if evaluator._callback(*call_args) != 0:
            raise RuntimeError("callback failed") from evaluator.last_error
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "device": torch.cuda.get_device_name(device),
        "model_config": checkpoint["model_config"],
        "batch": args.batch,
        "stones": args.stones,
        "top_k": args.top_k,
        "candidate_gumbel": args.candidate_gumbel,
        "force_model_only_compile": args.force_model_only_compile,
        "callback_ms": elapsed * 1000.0 / args.iterations,
        "positions_per_second": args.batch * args.iterations / elapsed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
