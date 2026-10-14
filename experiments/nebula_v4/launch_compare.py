"""Launch a fair two-GPU Dense64 versus MoE64 supervision comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


GENERATION_PATTERN = re.compile(r"^gen-(\d+)-attempt-")


def discover_generations(root: Path) -> list[tuple[int, Path]]:
    generations = []
    for directory in root.iterdir():
        match = GENERATION_PATTERN.match(directory.name)
        if directory.is_dir() and match and list(directory.glob("*.manifest.json")):
            generations.append((int(match.group(1)), directory.resolve()))
    generations.sort()
    duplicate = [
        generation
        for (generation, _), (next_generation, _) in zip(generations, generations[1:])
        if generation == next_generation
    ]
    if duplicate:
        raise ValueError(f"multiple attempts found for generations: {duplicate}")
    return generations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--validation-generations", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=12_000_000)
    parser.add_argument("--validation-samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--dense-gpu", default="0")
    parser.add_argument("--moe-gpu", default="1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261013)
    arguments = parser.parse_args()
    if not 0 < arguments.validation_generations < arguments.generations:
        raise ValueError("validation-generations must be inside the selected window")

    data_root = Path(arguments.data_root).expanduser().resolve()
    selected = discover_generations(data_root)[-arguments.generations:]
    if len(selected) != arguments.generations:
        raise ValueError(
            f"expected {arguments.generations} generations, found {len(selected)}"
        )
    expected = list(range(selected[0][0], selected[-1][0] + 1))
    actual = [generation for generation, _ in selected]
    if actual != expected:
        raise ValueError("selected generations are not contiguous")
    split = len(selected) - arguments.validation_generations
    training = selected[:split]
    validation = selected[split:]

    output_root = Path(arguments.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "-m",
        "experiments.nebula_v4.train_cpu",
        "--device",
        "cuda",
        "--dtype",
        "bf16",
        "--summary-tokens",
        "64",
        "--epochs",
        str(arguments.epochs),
        "--batch-size",
        str(arguments.batch_size),
        "--workers",
        str(arguments.workers),
        "--torch-threads",
        "16",
        "--train-samples",
        str(arguments.train_samples),
        "--validation-samples",
        str(arguments.validation_samples),
        "--learning-rate",
        str(arguments.learning_rate),
        "--minimum-learning-rate",
        str(arguments.minimum_learning_rate),
        "--warmup-ratio",
        str(arguments.warmup_ratio),
        "--log-interval",
        "50",
        "--replay-audit-workers",
        str(arguments.audit_workers),
        "--seed",
        str(arguments.seed),
        "--final-full-validation",
    ]
    for _, directory in training:
        common.extend(("--data-dir", str(directory)))
    for _, directory in validation:
        common.extend(("--validation-data-dir", str(directory)))

    jobs = {
        "dense64": {
            "gpu": arguments.dense_gpu,
            "command": common + [
                "--variant",
                "dense",
                "--output-dir",
                str(output_root / "dense64"),
            ],
        },
        "moe64": {
            "gpu": arguments.moe_gpu,
            "command": common + [
                "--variant",
                "moe",
                "--router-balance-weight",
                "0.1",
                "--output-dir",
                str(output_root / "moe64"),
            ],
        },
    }
    record = {
        "created_unix": time.time(),
        "training_generations": [training[0][0], training[-1][0]],
        "validation_generations": [validation[0][0], validation[-1][0]],
        "training_directories": [str(path) for _, path in training],
        "validation_directories": [str(path) for _, path in validation],
        "config": vars(arguments),
        "gradient_checkpointing": False,
        "jobs": {
            name: {"gpu": item["gpu"], "command": item["command"]}
            for name, item in jobs.items()
        },
    }
    (output_root / "launch.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    processes = {}
    logs = []
    try:
        for name, item in jobs.items():
            log = (output_root / f"{name}.log").open("w", encoding="utf-8")
            logs.append(log)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = item["gpu"]
            processes[name] = subprocess.Popen(
                item["command"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        return_codes = {name: process.wait() for name, process in processes.items()}
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    record["finished_unix"] = time.time()
    record["return_codes"] = return_codes
    (output_root / "launch.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(return_codes, ensure_ascii=False))
    return max(return_codes.values())


if __name__ == "__main__":
    raise SystemExit(main())
