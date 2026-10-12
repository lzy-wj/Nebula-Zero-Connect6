"""Isolated wall-time benchmark for strict replay validation strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from experiments.pair_native.async_loop import validate_replay
from experiments.pair_native.train import parse_cpu_affinity


def summary_sha256(summary: dict) -> str:
    payload = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--games", required=True, type=int)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--black-simulations", required=True, type=int)
    parser.add_argument("--white-simulations", required=True, type=int)
    parser.add_argument("--opening-temperature", required=True, type=float)
    parser.add_argument("--maximum-opening-top-share", type=float, default=0.5)
    parser.add_argument("--minimum-unique-game-rate", type=float, default=0.9)
    parser.add_argument("--audit-workers", default="1,7")
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()

    affinity = parse_cpu_affinity(arguments.cpu_affinity)
    if affinity:
        os.sched_setaffinity(0, affinity)
    worker_counts = [
        int(value)
        for value in arguments.audit_workers.split(",")
        if value.strip()
    ]
    if not worker_counts or any(value < 1 for value in worker_counts):
        raise ValueError("audit-workers must contain positive integers")

    directory = Path(arguments.directory).expanduser().resolve()
    results = []
    expected_summary_sha256 = None
    for worker_count in worker_counts:
        started = time.perf_counter()
        summary = validate_replay(
            directory,
            arguments.generation,
            arguments.games,
            arguments.model_sha256,
            arguments.black_simulations,
            arguments.white_simulations,
            arguments.opening_temperature,
            arguments.maximum_opening_top_share,
            arguments.minimum_unique_game_rate,
            audit_workers=worker_count,
        )
        seconds = time.perf_counter() - started
        digest = summary_sha256(summary)
        if expected_summary_sha256 is None:
            expected_summary_sha256 = digest
        elif digest != expected_summary_sha256:
            raise RuntimeError("validation strategies produced different summaries")
        results.append({
            "audit_workers": worker_count,
            "seconds": seconds,
            "summary_sha256": digest,
        })

    payload = {
        "schema_version": 1,
        "purpose": "performance-only strict replay validation benchmark",
        "directory": str(directory),
        "generation": arguments.generation,
        "games": arguments.games,
        "model_sha256": arguments.model_sha256,
        "cpu_affinity": affinity,
        "results": results,
    }
    if arguments.output:
        atomic_json(payload, Path(arguments.output).expanduser().resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
