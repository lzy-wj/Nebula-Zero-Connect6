"""Isolated one-GPU self-play systems sweep.

The generated shards live outside an async-loop replay root and are performance
artifacts only.  Search quality settings remain fixed across variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.pair_native.pair_replay import sha256_file


VARIANTS = (
    {
        "name": "c128_t24_physical",
        "concurrent_games": 128,
        "mcts_cpu_threads": 24,
        "include_smt_affinity": False,
        "batch_buckets": "16,32,64,128",
    },
    {
        "name": "c192_t24_physical",
        "concurrent_games": 192,
        "mcts_cpu_threads": 24,
        "include_smt_affinity": False,
        "batch_buckets": "16,32,64,128,192",
    },
    {
        "name": "c192_t32_smt",
        "concurrent_games": 192,
        "mcts_cpu_threads": 32,
        "include_smt_affinity": True,
        "batch_buckets": "16,32,64,128,192",
    },
)


def checkpoint_from_arguments(arguments) -> Path:
    if arguments.state:
        state_path = Path(arguments.state).expanduser().resolve()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") not in {"paused", "complete"}:
            raise RuntimeError(
                f"benchmark state must be paused or complete, got {state.get('status')!r}"
            )
        return Path(state["accepted_checkpoint"]).expanduser().resolve()
    return Path(arguments.checkpoint).expanduser().resolve()


def validate_benchmark_output_root(output_root: Path) -> None:
    """Refuse paths that could be discovered as async training replay."""
    for path in (output_root, *output_root.parents):
        if path.name in {"replay", "candidates", "gates"}:
            raise ValueError(f"benchmark output cannot be under {path.name}/")
        if (path / "state.json").is_file():
            raise ValueError(f"benchmark output cannot be under async-loop root {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--state")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026092405)
    arguments = parser.parse_args()

    if arguments.games < 256:
        raise ValueError("systems sweep requires at least 256 games per variant")
    checkpoint = checkpoint_from_arguments(arguments)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_root = Path(arguments.output_dir).expanduser().resolve()
    validate_benchmark_output_root(output_root)
    if arguments.state:
        state_root = Path(arguments.state).expanduser().resolve().parent
        if output_root == state_root or state_root in output_root.parents:
            raise ValueError("benchmark output must be outside the async-loop state root")
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for index, variant in enumerate(VARIANTS):
        variant_root = output_root / variant["name"]
        generation = 9000 + index
        summary_path = variant_root / f"gen-{generation:04d}-summary.json"
        request = {
            "schema_version": 1,
            "purpose": "performance-only; never training-visible",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "variant": variant,
            "games": arguments.games,
            "simulations": arguments.simulations,
            "top_k": arguments.top_k,
            "gpu": arguments.gpu,
            "generation": generation,
            "seed": arguments.seed,
        }
        request_path = variant_root / "request.json"
        if variant_root.exists():
            if not request_path.is_file():
                raise RuntimeError(
                    f"existing variant has no provenance request: {variant_root}"
                )
            existing_request = json.loads(request_path.read_text(encoding="utf-8"))
            if existing_request != request:
                raise RuntimeError(f"benchmark request mismatch in {variant_root}")
            if not summary_path.is_file():
                raise RuntimeError(
                    f"incomplete prior variant; use a new output root: {variant_root}"
                )
        else:
            variant_root.mkdir(parents=True)
            temporary_request = request_path.with_suffix(".json.tmp")
            temporary_request.write_text(
                json.dumps(request, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_request.replace(request_path)
        if not summary_path.exists():
            command = [
                sys.executable,
                "-m",
                "experiments.pair_native.selfplay",
                "--output-dir",
                str(variant_root),
                "--checkpoint",
                str(checkpoint),
                "--games",
                str(arguments.games),
                "--gpus",
                str(arguments.gpu),
                "--concurrent-games",
                str(variant["concurrent_games"]),
                "--simulations",
                str(arguments.simulations),
                "--top-k",
                str(arguments.top_k),
                "--candidate-gumbel",
                "0.25",
                "--value-scale",
                "1.0",
                "--forced-opening-radius",
                "-1",
                "--temperature-turns",
                "4",
                "--early-temperature",
                "0.35",
                "--opening-temperature",
                "1.25",
                "--mcts-cpu-threads",
                str(variant["mcts_cpu_threads"]),
                "--batch-buckets",
                variant["batch_buckets"],
                "--generation",
                str(generation),
                "--seed",
                str(arguments.seed),
            ]
            if variant["include_smt_affinity"]:
                command.append("--include-smt-affinity")
            log_path = variant_root / "run.log"
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    check=True,
                    cwd=Path(__file__).resolve().parents[2],
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if not summary_path.is_file():
                raise RuntimeError(f"self-play did not publish {summary_path}")
        else:
            started = summary_path.stat().st_mtime

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["games"] != arguments.games:
            raise RuntimeError(f"unexpected game count in {summary_path}")
        if not all(worker["audit"]["valid"] for worker in summary["workers"]):
            raise RuntimeError(f"invalid replay in {summary_path}")
        for worker in summary["workers"]:
            metadata = worker["audit"]["metadata"]
            if Path(metadata["checkpoint"]).resolve() != checkpoint:
                raise RuntimeError(f"checkpoint provenance mismatch in {summary_path}")
            execution = metadata.get("execution", {})
            expected_execution = {
                "concurrent_games": variant["concurrent_games"],
                "include_smt_affinity": variant["include_smt_affinity"],
                "pad_batch_to": 0,
            }
            if execution != expected_execution:
                raise RuntimeError(f"execution provenance mismatch in {summary_path}")
            search = metadata["search"]
            if (
                search["simulations"] != arguments.simulations
                or search["top_k"] != arguments.top_k
                or search["mcts_cpu_threads"] != variant["mcts_cpu_threads"]
                or search["batch_buckets"]
                != [int(value) for value in variant["batch_buckets"].split(",")]
            ):
                raise RuntimeError(f"search provenance mismatch in {summary_path}")
        requested = sum(
            worker["evaluator"]["requested_positions"]
            for worker in summary["workers"]
        )
        batches = sum(worker["evaluator"]["eval_batches"] for worker in summary["workers"])
        results.append(
            variant
            | {
                "games": summary["games"],
                "turns": summary["turns"],
                "seconds": summary["seconds"],
                "games_per_second": summary["games_per_second"],
                "turns_per_second": summary["turns_per_second"],
                "average_eval_batch": requested / max(batches, 1),
                "unique_complete_game_rate": summary["diversity"][
                    "unique_complete_game_rate"
                ],
                "black_wins": summary["winner_counts"]["black"],
                "white_wins": summary["winner_counts"]["white"],
                "draws": summary["winner_counts"]["draw"],
                "summary": str(summary_path),
                "completed_unix": max(started, summary_path.stat().st_mtime),
            }
        )

    baseline = results[0]["turns_per_second"]
    for result in results:
        result["turns_per_second_relative_to_baseline"] = (
            result["turns_per_second"] / baseline
        )
    payload = {
        "schema_version": 1,
        "purpose": "performance-only; never training-visible",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "games_per_variant": arguments.games,
        "simulations": arguments.simulations,
        "top_k": arguments.top_k,
        "gpu": arguments.gpu,
        "seed": arguments.seed,
        "results": results,
    }
    result_path = output_root / "sweep.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
