"""Run isolated, same-opening gate evaluator scheduling benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "compiled_buckets": {
        "compile_model": True,
        "batch_buckets": "16,32,64,128",
        "pad_batch_to": 0,
    },
    "eager_buckets": {
        "compile_model": False,
        "batch_buckets": "16,32,64,128",
        "pad_batch_to": 0,
    },
    "compiled_fixed": {
        "compile_model": True,
        "batch_buckets": "",
        "pad_batch_to": None,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--opening-radius", type=int, default=4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--seed", type=int, default=2026092407)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
    )
    arguments = parser.parse_args()

    initial = Path(arguments.initial_checkpoint).expanduser().resolve()
    trained = Path(arguments.trained_checkpoint).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"gate sweep output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [value.strip() for value in arguments.variants.split(",") if value.strip()]
    if not names or any(name not in VARIANTS for name in names):
        raise ValueError(f"unknown gate variants: {names}")

    results = []
    expected_fingerprints = None
    for name in names:
        variant = VARIANTS[name]
        result_path = output_dir / f"{name}.json"
        log_path = output_dir / f"{name}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.pair_native.arena",
            "--initial-checkpoint",
            str(initial),
            "--trained-checkpoint",
            str(trained),
            "--pairs",
            str(arguments.pairs),
            "--simulations",
            str(arguments.simulations),
            "--top-k",
            str(arguments.top_k),
            "--opening-radius",
            str(arguments.opening_radius),
            "--value-scale",
            str(arguments.value_scale),
            "--mcts-cpu-threads",
            str(arguments.mcts_cpu_threads),
            "--batch-buckets",
            variant["batch_buckets"],
            "--pad-batch-to",
            str(arguments.pairs if variant["pad_batch_to"] is None else variant["pad_batch_to"]),
            "--cpu-affinity",
            arguments.cpu_affinity,
            "--seed",
            str(arguments.seed),
            "--output",
            str(result_path),
            "--compile-model" if variant["compile_model"] else "--no-compile-model",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        arena = json.loads(result_path.read_text(encoding="utf-8"))
        fingerprints = arena["game_fingerprints"]
        if expected_fingerprints is None:
            expected_fingerprints = fingerprints
        results.append({
            "name": name,
            "seconds": arena["seconds"],
            "games_per_second": arena["games_per_second"],
            "turns_per_second": arena["turns_per_second"],
            "mean_turns": arena["mean_turns"],
            "same_games_as_first": fingerprints == expected_fingerprints,
            "evaluators": arena["evaluators"],
            "result": str(result_path),
        })

    payload = {
        "schema_version": 1,
        "purpose": "performance-only gate evaluator scheduling benchmark",
        "initial_checkpoint": str(initial),
        "initial_sha256": sha256_file(initial),
        "trained_checkpoint": str(trained),
        "trained_sha256": sha256_file(trained),
        "pairs": arguments.pairs,
        "simulations": arguments.simulations,
        "gpu": arguments.gpu,
        "results": results,
    }
    atomic_json(payload, output_dir / "sweep.json")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
