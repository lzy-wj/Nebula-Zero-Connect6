"""Benchmark strict learner replay loading without starting a GPU training job."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from experiments.pair_native.train import load_data, parse_cpu_affinity


def update_digest(digest, value):
    # Pickle bytes are not a semantic digest: equal dictionaries arriving from
    # another process can have different memo/reference encodings.  Canonical
    # JSON makes the cross-process benchmark compare values instead.
    digest.update(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: (
                {"__bytes_hex__": bytes(item).hex()}
                if isinstance(item, (bytes, bytearray, memoryview))
                else _raise_unserializable(item)
            ),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _raise_unserializable(value):
    raise TypeError(f"unsupported digest value: {type(value).__name__}")


def data_summary(payload) -> dict:
    games, game_lengths, training_rows, validation_rows, audits = payload
    digest = hashlib.sha256()
    update_digest(digest, games)
    update_digest(digest, game_lengths)
    for rows in (training_rows, validation_rows):
        for row in rows:
            update_digest(digest, row)
    update_digest(digest, audits)
    return {
        "games": len(games),
        "training_turns": len(training_rows),
        "validation_turns": len(validation_rows),
        "audited_shards": len(audits),
        "content_sha256": digest.hexdigest(),
    }


def atomic_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--order",
        default="legacy,single,single,legacy",
        help="comma-separated legacy/single/parallel sequence",
    )
    parser.add_argument("--parallel-workers", type=int, default=7)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()

    affinity = parse_cpu_affinity(arguments.cpu_affinity)
    if affinity:
        os.sched_setaffinity(0, affinity)
    modes = [value.strip() for value in arguments.order.split(",") if value.strip()]
    if not modes or any(value not in {"legacy", "single", "parallel"} for value in modes):
        raise ValueError("order must contain only legacy, single, and parallel")
    if arguments.parallel_workers < 1:
        raise ValueError("parallel-workers must be positive")
    directories = [Path(value).expanduser().resolve() for value in arguments.data_dir]

    results = []
    expected_content_sha256 = None
    for mode in modes:
        started = time.perf_counter()
        payload = load_data(
            directories,
            arguments.validation_fraction,
            single_pass_audit_load=mode == "single",
            replay_audit_workers=(
                arguments.parallel_workers if mode == "parallel" else 1
            ),
        )
        seconds = time.perf_counter() - started
        summary = data_summary(payload)
        if expected_content_sha256 is None:
            expected_content_sha256 = summary["content_sha256"]
        elif summary["content_sha256"] != expected_content_sha256:
            raise RuntimeError("replay loading strategies produced different content")
        results.append({"mode": mode, "seconds": seconds, **summary})
        del payload
        gc.collect()

    medians = {
        mode: statistics.median(
            item["seconds"] for item in results if item["mode"] == mode
        )
        for mode in ("legacy", "single", "parallel")
        if any(item["mode"] == mode for item in results)
    }
    payload = {
        "schema_version": 1,
        "purpose": "performance-only strict learner replay-load benchmark",
        "data_dirs": [str(value) for value in directories],
        "validation_fraction": arguments.validation_fraction,
        "cpu_affinity": affinity,
        "results": results,
        "median_seconds": medians,
        "single_pass_speedup": (
            medians["legacy"] / medians["single"]
            if {"legacy", "single"} <= medians.keys()
            else None
        ),
        "parallel_speedup": (
            medians["legacy"] / medians["parallel"]
            if {"legacy", "parallel"} <= medians.keys()
            else None
        ),
    }
    if arguments.output:
        atomic_json(payload, Path(arguments.output).expanduser().resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
