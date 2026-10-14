"""Profile a production-shaped V4 MCTS search on one GPU.

The report separates evaluator callback time from the C++ MCTS/search
remainder.  It deliberately uses real search contexts instead of estimating
end-to-end behavior from model-only throughput.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time

import torch

from experiments.pair_native.arena import load_arena_checkpoint
from experiments.pair_native.pair_mcts import PairMCTSEngine, TorchPairEvaluator
from experiments.pair_native.selfplay import parse_linux_cpu_list


class TimedEvaluator(TorchPairEvaluator):
    def __init__(self, *args, **kwargs):
        self.callback_seconds = 0.0
        self.callback_batches: Counter[int] = Counter()
        super().__init__(*args, **kwargs)

    def _callback(self, batch_size, *args):
        started = time.perf_counter()
        try:
            return super()._callback(batch_size, *args)
        finally:
            self.callback_seconds += time.perf_counter() - started
            self.callback_batches[int(batch_size)] += 1

    def reset_profile(self):
        self.callback_seconds = 0.0
        self.callback_batches.clear()
        self.requested_positions = 0
        self.unique_positions = 0
        self.eval_batches = 0


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library", default="")
    parser.add_argument("--contexts", type=int, default=128)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--hierarchical-first", type=int, default=0)
    parser.add_argument("--hierarchical-second", type=int, default=0)
    parser.add_argument("--progressive-widening-base", type=int, default=0)
    parser.add_argument("--progressive-widening-scale", type=float, default=0.0)
    parser.add_argument("--async-pipeline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--batch-buckets", default="16,32,64,128")
    parser.add_argument("--candidate-gumbel", type=float, default=0.25)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-model-only-compile",
        action="store_true",
        help="keep Pair scoring outside the compiled graph even when Gumbel is zero",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="")
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--start-file", default="")
    parser.add_argument("--barrier-timeout-seconds", type=float, default=600.0)
    arguments = parser.parse_args()
    if min(arguments.contexts, arguments.turns, arguments.simulations, arguments.top_k) < 1:
        raise ValueError("contexts, turns, simulations, and top-k must be positive")
    if arguments.barrier_timeout_seconds <= 0.0:
        raise ValueError("barrier-timeout-seconds must be positive")
    affinity = parse_linux_cpu_list(arguments.cpu_affinity)
    if affinity and hasattr(os, "sched_getaffinity"):
        unavailable = sorted(set(affinity) - os.sched_getaffinity(0))
        if unavailable:
            raise ValueError(f"CPU affinity contains unavailable CPUs: {unavailable}")
        os.sched_setaffinity(0, affinity)

    device = torch.device("cuda:0")
    buckets = tuple(
        int(item) for item in arguments.batch_buckets.split(",") if item.strip()
    )
    model, checkpoint = load_arena_checkpoint(arguments.checkpoint)
    construction_gumbel = arguments.candidate_gumbel
    if (
        arguments.force_model_only_compile
        and arguments.compile_model
        and construction_gumbel == 0.0
    ):
        construction_gumbel = 1.0e-12
    evaluator = TimedEvaluator(
        model,
        device,
        top_k=arguments.top_k,
        candidate_gumbel=construction_gumbel,
        compile_model=arguments.compile_model,
        deduplicate=False,
        pad_batch_to=0,
        batch_buckets=buckets,
        value_scale=1.0,
        seed=arguments.seed,
        hierarchical_first_candidates=arguments.hierarchical_first,
        hierarchical_seconds_per_first=arguments.hierarchical_second,
    )
    evaluator.candidate_gumbel = arguments.candidate_gumbel

    # Compile every production bucket before starting the measured interval.
    # Candidate Gumbel keeps pair selection outside the compiled model, which
    # matches the current self-play evaluator.
    if arguments.compile_model:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for batch in buckets:
                if evaluator.inference is None:
                    features = torch.zeros(batch, 5, 19, 19, device=device)
                    evaluator.model(features)
                else:
                    evaluator.warmup(batch)
        torch.cuda.synchronize(device)

    engine = PairMCTSEngine(
        evaluator,
        library_path=arguments.library or None,
        top_k=arguments.top_k,
        root_noise_epsilon=0.25,
        root_dirichlet_alpha=0.15,
        cpu_threads=arguments.mcts_cpu_threads,
        progressive_widening_base=arguments.progressive_widening_base,
        progressive_widening_scale=arguments.progressive_widening_scale,
        async_pipeline=arguments.async_pipeline,
    )
    contexts = [
        engine.create_context(arguments.seed + index)
        for index in range(arguments.contexts)
    ]
    context_ids = {id(context): index for index, context in enumerate(contexts)}
    game_turns = [0] * len(contexts)
    game_hashes = [hashlib.sha256() for _ in contexts]
    openings = [-1] * len(contexts)
    winners: list[int | None] = [None] * len(contexts)

    if bool(arguments.ready_file) != bool(arguments.start_file):
        raise ValueError("ready-file and start-file must be provided together")
    if arguments.ready_file:
        ready_path = Path(arguments.ready_file).expanduser().resolve()
        start_path = Path(arguments.start_file).expanduser().resolve()
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ready_path.with_suffix(ready_path.suffix + ".tmp")
        temporary.write_text(f"{os.getpid()}\n", encoding="utf-8")
        temporary.replace(ready_path)
        deadline = time.monotonic() + arguments.barrier_timeout_seconds
        while not start_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for start barrier: {start_path}")
            time.sleep(0.01)
        torch.cuda.synchronize(device)

    search_seconds = 0.0
    action_seconds = 0.0
    cpu_started = cpu_seconds()
    wall_started = time.perf_counter()
    completed_turns = 0
    action_digest = hashlib.sha256()
    try:
        engine.reset_global_stats()
        evaluator.reset_profile()
        active = contexts
        for _ in range(arguments.turns):
            if not active:
                break
            started = time.perf_counter()
            engine.run(active, arguments.simulations)
            torch.cuda.synchronize(device)
            search_seconds += time.perf_counter() - started

            started = time.perf_counter()
            next_active = []
            for context in active:
                context_index = context_ids[id(context)]
                first, second = context.choose_action(temperature=0.0)
                action_digest.update(
                    struct.pack(
                        "<iiii",
                        context_index,
                        game_turns[context_index],
                        first,
                        second,
                    )
                )
                game_hashes[context_index].update(struct.pack("<ii", first, second))
                if game_turns[context_index] == 0:
                    openings[context_index] = first
                context.play(first, second)
                completed_turns += 1
                game_turns[context_index] += 1
                if not context.terminal:
                    next_active.append(context)
                else:
                    winners[context_index] = context.winner
            action_seconds += time.perf_counter() - started
            active = next_active
    finally:
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_started
        used_cpu_seconds = cpu_seconds() - cpu_started
        for context in contexts:
            context.close()

    callback_seconds = evaluator.callback_seconds
    mcts_remainder_seconds = max(0.0, search_seconds - callback_seconds)
    accounted = callback_seconds + mcts_remainder_seconds + action_seconds
    finished_lengths = [
        game_turns[index]
        for index, winner in enumerate(winners)
        if winner is not None
    ]
    finished_fingerprints = [
        game_hashes[index].hexdigest()
        for index, winner in enumerate(winners)
        if winner is not None
    ]
    result = {
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "model_config": checkpoint["model_config"],
        "device": torch.cuda.get_device_name(device),
        "contexts": arguments.contexts,
        "requested_turns": arguments.turns,
        "completed_turns": completed_turns,
        "simulations_per_turn": arguments.simulations,
        "top_k": arguments.top_k,
        "hierarchical_first": arguments.hierarchical_first,
        "hierarchical_second": arguments.hierarchical_second,
        "progressive_widening_base": arguments.progressive_widening_base,
        "progressive_widening_scale": arguments.progressive_widening_scale,
        "async_pipeline": arguments.async_pipeline,
        "mcts_cpu_threads": arguments.mcts_cpu_threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "candidate_gumbel": arguments.candidate_gumbel,
        "compile_model": arguments.compile_model,
        "force_model_only_compile": arguments.force_model_only_compile,
        "batch_buckets": buckets,
        "wall_seconds": wall_seconds,
        "search_seconds": search_seconds,
        "evaluator_callback_seconds": callback_seconds,
        "mcts_cpu_and_coordination_seconds": mcts_remainder_seconds,
        "action_and_tree_advance_seconds": action_seconds,
        "unaccounted_seconds": max(0.0, wall_seconds - accounted),
        "search_share_percent": 100.0 * search_seconds / wall_seconds,
        "callback_share_of_search_percent": 100.0 * callback_seconds / search_seconds,
        "mcts_remainder_share_of_search_percent": 100.0 * mcts_remainder_seconds / search_seconds,
        "turns_per_second": completed_turns / wall_seconds,
        "simulations_per_second": (
            engine.global_stats()["simulations"] / search_seconds
        ),
        "process_cpu_seconds": used_cpu_seconds,
        "average_cpu_cores": used_cpu_seconds / wall_seconds,
        "callback_batch_histogram": dict(sorted(evaluator.callback_batches.items())),
        "evaluator": evaluator.stats(),
        "mcts": engine.global_stats(),
        "pid": os.getpid(),
        "action_sha256": action_digest.hexdigest(),
        "games_finished": len(finished_lengths),
        "winner_counts": {
            "black": sum(winner == 1 for winner in winners),
            "white": sum(winner == -1 for winner in winners),
            "draw": sum(winner == 0 for winner in winners),
        },
        "mean_finished_turns": (
            sum(finished_lengths) / len(finished_lengths)
            if finished_lengths
            else 0.0
        ),
        "unique_finished_games": len(set(finished_fingerprints)),
        "unique_openings": len({move for move in openings if move >= 0}),
    }
    payload = json.dumps(result, indent=2)
    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
