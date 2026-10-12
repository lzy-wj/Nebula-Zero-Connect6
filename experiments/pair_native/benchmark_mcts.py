"""End-to-end H20 callback + C++ turn-level MCTS benchmark."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.pair_native.model import build_model, parameter_count
from experiments.pair_native.pair_mcts import PairMCTSEngine, TorchPairEvaluator


def timed_search(engine, contexts, simulations):
    engine.reset_global_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.run(contexts, simulations)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    total_simulations = simulations * len(contexts)
    return {
        "seconds": elapsed,
        "simulations": total_simulations,
        "simulations_per_second": total_simulations / elapsed,
        "turns_per_second": len(contexts) / elapsed,
        "search_ms_per_turn": elapsed * 1000.0,
        "stats": engine.global_stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--dense-channels", type=int, default=32)
    parser.add_argument("--candidate-gumbel", type=float, default=0.25)
    parser.add_argument("--random-candidates", type=int, default=0)
    parser.add_argument("--mcts-cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pad-batch-to", type=int, default=0)
    parser.add_argument("--batch-buckets", default="")
    parser.add_argument("--value-scale", type=float, default=0.0)
    arguments = parser.parse_args()
    batch_buckets = [
        int(value)
        for value in arguments.batch_buckets.split(",")
        if value.strip()
    ]

    torch.manual_seed(arguments.seed)
    device = torch.device("cuda:0")
    model_config = {
        "max_tokens": arguments.max_tokens,
        "dense_channels": arguments.dense_channels,
    }
    model = build_model(model_config)
    evaluator = TorchPairEvaluator(
        model,
        device,
        top_k=arguments.top_k,
        candidate_gumbel=arguments.candidate_gumbel,
        seed=arguments.seed,
        compile_model=arguments.compile_model,
        deduplicate=arguments.deduplicate,
        pad_batch_to=arguments.pad_batch_to or arguments.contexts,
        batch_buckets=batch_buckets,
        value_scale=arguments.value_scale,
    )
    engine = PairMCTSEngine(
        evaluator,
        top_k=arguments.top_k,
        random_candidates=arguments.random_candidates,
        cpu_threads=arguments.mcts_cpu_threads,
    )
    contexts = [engine.create_context(arguments.seed + index) for index in range(arguments.contexts)]
    try:
        # Warm the kernels with independent temporary trees.
        warm_contexts = [
            engine.create_context(arguments.seed + 1_000_000 + index)
            for index in range(len(contexts))
        ]
        try:
            engine.run(warm_contexts, 2)
        finally:
            for context in warm_contexts:
                context.close()
        opening = timed_search(engine, contexts, arguments.simulations)
        for context in contexts:
            context.play(*context.choose_action(temperature=1.0))
        pair_turn = timed_search(engine, contexts, arguments.simulations)
        output = {
            "gpu": torch.cuda.get_device_name(device),
            "parameters": parameter_count(model),
            "contexts": arguments.contexts,
            "simulations_per_turn": arguments.simulations,
            "model_config": model_config,
            "opening": opening,
            "pair_turn": pair_turn,
        }
        print(json.dumps(output, indent=2))
    finally:
        for context in contexts:
            context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
