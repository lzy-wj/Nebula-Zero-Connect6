"""Launch Dense and MoE cache training with two GPUs per candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dense-gpus", default="2,3")
    parser.add_argument("--moe-gpus", default="6,7")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers-per-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261013)
    arguments = parser.parse_args()
    cache_dir = Path(arguments.cache_dir).expanduser().resolve()
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if not metadata.get("strict_audit"):
        raise ValueError("cache is not strictly audited")
    output_root = Path(arguments.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "experiments.nebula_v4.train_cpu",
        "--cache-dir",
        str(cache_dir),
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
        str(arguments.workers_per_rank),
        "--torch-threads",
        "8",
        "--learning-rate",
        "0.0006",
        "--minimum-learning-rate",
        "0.00006",
        "--warmup-ratio",
        "0.03",
        "--log-interval",
        "50",
        "--seed",
        str(arguments.seed),
        "--final-full-validation",
    ]
    jobs = {
        "dense64": {
            "gpus": arguments.dense_gpus,
            "command": common
            + ["--variant", "dense", "--output-dir", str(output_root / "dense64")],
        },
        "moe64": {
            "gpus": arguments.moe_gpus,
            "command": common
            + [
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
        "cache_metadata": metadata,
        "config": vars(arguments),
        "gradient_checkpointing": False,
        "jobs": jobs,
    }
    launch_path = output_root / "launch.json"
    launch_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    processes = {}
    logs = []
    try:
        for name, job in jobs.items():
            log = (output_root / f"{name}.log").open("w", encoding="utf-8")
            logs.append(log)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = job["gpus"]
            processes[name] = subprocess.Popen(
                job["command"], stdout=log, stderr=subprocess.STDOUT, env=environment
            )
        return_codes = {name: process.wait() for name, process in processes.items()}
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    record.update({"finished_unix": time.time(), "return_codes": return_codes})
    launch_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(return_codes), flush=True)
    return max(return_codes.values())


if __name__ == "__main__":
    raise SystemExit(main())
