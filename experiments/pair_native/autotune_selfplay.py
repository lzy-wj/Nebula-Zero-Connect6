"""Hardware-aware single-GPU self-play autotuning.

The quick stages rank production-shaped MCTS service configurations.  An
optional full stage writes audited replay shards outside the training tree and
uses the second committed shard for steady-state ranking.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from experiments.pair_native.pair_replay import sha256_file
from experiments.pair_native.selfplay import parse_linux_cpu_list
from experiments.pair_native.selfplay import topology_worker_affinities
from experiments.pair_native.selfplay_sweep import validate_benchmark_output_root


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CPU_LIST = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")


def parse_positive_csv(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values or min(values) < 1:
        raise ValueError("candidate lists must contain positive integers")
    return values


def compress_cpu_list(cpus: list[int]) -> str:
    if not cpus:
        return ""
    values = sorted(set(int(cpu) for cpu in cpus))
    ranges = []
    start = previous = values[0]
    for cpu in values[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def topology_cpu_affinity(text: str, gpu_id: int) -> list[int]:
    """Extract the largest numeric CPU-list field from an nvidia-smi row."""
    target = f"GPU{int(gpu_id)}"
    clean = ANSI_ESCAPE.sub("", text)
    for line in clean.splitlines():
        fields = line.split()
        if not fields or fields[0] != target:
            continue
        candidates = []
        for field in fields[1:]:
            if CPU_LIST.fullmatch(field):
                cpus = parse_linux_cpu_list(field)
                candidates.append(cpus)
        if candidates:
            return max(candidates, key=len)
    return []


def physical_cpu_representatives(cpus: list[int]) -> list[int]:
    """Return one allowed logical CPU per physical core."""
    allowed = set(cpus)
    representatives = []
    seen: set[tuple[int, ...]] = set()
    for cpu in sorted(allowed):
        path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        try:
            siblings = tuple(
                value
                for value in parse_linux_cpu_list(path.read_text(encoding="utf-8"))
                if value in allowed
            )
        except (OSError, ValueError):
            siblings = (cpu,)
        key = siblings or (cpu,)
        if key in seen:
            continue
        seen.add(key)
        representatives.append(cpu)
    return representatives


def partition_cpu_affinity(cpus: list[int], workers: int) -> list[list[int]]:
    if workers < 1 or len(cpus) < workers:
        raise ValueError("CPU affinity must provide at least one CPU per worker")
    return [
        cpus[len(cpus) * worker // workers:len(cpus) * (worker + 1) // workers]
        for worker in range(workers)
    ]


def detect_cpu_affinity(
    gpu_id: int,
    requested: str,
    deployment_gpu_ids: list[int] | None = None,
) -> list[int]:
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if requested:
        affinity = parse_linux_cpu_list(requested)
    elif deployment_gpu_ids:
        if gpu_id not in deployment_gpu_ids:
            raise ValueError("benchmark GPU must be present in deployment GPU list")
        plans = topology_worker_affinities(deployment_gpu_ids)
        affinity = plans[deployment_gpu_ids.index(gpu_id)]
    else:
        try:
            topology = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            affinity = topology_cpu_affinity(topology, gpu_id)
        except (OSError, subprocess.CalledProcessError):
            affinity = []
        affinity = physical_cpu_representatives(affinity)
    if available:
        affinity = [cpu for cpu in affinity if cpu in set(available)]
    if not affinity:
        affinity = available
    if not affinity:
        raise RuntimeError("no usable CPUs were discovered")
    return affinity


def detect_gpu(gpu_id: int) -> dict:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, memory, driver = [item.strip() for item in query.split(",", 2)]
    return {
        "index": int(gpu_id),
        "name": name,
        "memory_mib": int(memory),
        "driver": driver,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_or_write_request(root: Path, request: dict) -> None:
    """Prevent benchmark cache reuse across models, hardware, or settings."""
    path = root / "autotune-request.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != request:
            raise ValueError(
                f"autotune output belongs to a different request: {path}; "
                "use a new --output-dir"
            )
        return
    existing = [item for item in root.iterdir() if item.name != path.name]
    if existing:
        raise ValueError(
            f"autotune output is non-empty but has no request manifest: {root}; "
            "use a new --output-dir"
        )
    atomic_json(path, request)


def run_command(
    command: list[str],
    log_path: Path,
    environment: dict,
    process_affinity: list[int] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    preexec_fn = None
    if process_affinity and hasattr(os, "sched_setaffinity"):
        allowed = set(process_affinity)
        preexec_fn = lambda: os.sched_setaffinity(0, allowed)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec_fn,
        )


def quick_profile(
    arguments,
    root: Path,
    affinity_text: str,
    contexts: int,
    threads: int,
    repeat: int,
) -> dict:
    output = root / f"quick-c{contexts}-t{threads}-r{repeat}.json"
    log = output.with_suffix(".log")
    if not output.is_file():
        command = [
            sys.executable,
            "-m",
            "experiments.nebula_v4.profile_selfplay",
            "--checkpoint",
            str(arguments.checkpoint),
            "--contexts",
            str(contexts),
            "--turns",
            str(arguments.profile_turns),
            "--simulations",
            str(arguments.simulations),
            "--top-k",
            str(arguments.top_k),
            "--hierarchical-first",
            str(arguments.hierarchical_first),
            "--hierarchical-second",
            str(arguments.hierarchical_second),
            "--progressive-widening-base",
            str(arguments.progressive_widening_base),
            "--progressive-widening-scale",
            str(arguments.progressive_widening_scale),
            "--mcts-cpu-threads",
            str(threads),
            "--cpu-affinity",
            affinity_text,
            "--batch-buckets",
            str(contexts),
            "--candidate-gumbel",
            "0.25",
            "--compile-model",
            "--seed",
            str(arguments.seed + repeat),
            "--output",
            str(output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu)
        run_command(command, log, environment)
    result = json.loads(output.read_text(encoding="utf-8"))
    result["repeat"] = repeat
    return result


def aggregate_quick(rows: list[dict]) -> dict:
    return {
        "contexts": rows[0]["contexts"],
        "mcts_cpu_threads": rows[0]["mcts_cpu_threads"],
        "turns_per_second": statistics.median(row["turns_per_second"] for row in rows),
        "simulations_per_second": statistics.median(
            row["simulations_per_second"] for row in rows
        ),
        "callback_share_percent": statistics.median(
            row["callback_share_of_search_percent"] for row in rows
        ),
        "average_cpu_cores": statistics.median(
            row["average_cpu_cores"] for row in rows
        ),
        "runs": rows,
    }


def distinct_context_candidates(ranked: list[dict], limit: int) -> list[dict]:
    """Keep the fastest thread setting for each distinct concurrency level."""
    selected = []
    seen_contexts = set()
    for candidate in ranked:
        if candidate["contexts"] in seen_contexts:
            continue
        seen_contexts.add(candidate["contexts"])
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def full_profile(arguments, root: Path, affinity_text: str, candidate: dict, index: int) -> dict:
    contexts = int(candidate["contexts"])
    threads = int(candidate["mcts_cpu_threads"])
    shard_games = int(arguments.full_shard_games)
    burn_in_shards = (
        int(arguments.full_burn_in_shards)
        if arguments.full_burn_in_shards >= 0
        else (contexts + shard_games - 1) // shard_games
    )
    measure_shards = int(arguments.full_measure_shards)
    stop_after_shards = burn_in_shards + measure_shards
    worker_count = int(arguments.workers_per_gpu)
    games = max(
        int(arguments.full_games),
        worker_count * (contexts + stop_after_shards * shard_games),
    )
    run_root = root / (
        f"full-w{worker_count}-c{contexts}-t{threads}"
        f"-b{burn_in_shards}-m{measure_shards}"
    )
    generation = 9900 + index
    summary_path = run_root / f"gen-{generation:04d}-summary.json"
    log_path = run_root / "run.log"
    if not summary_path.is_file():
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.pair_native.selfplay",
            "--output-dir",
            str(run_root),
            "--checkpoint",
            str(arguments.checkpoint),
            "--games",
            str(games),
            "--shard-games",
            str(shard_games),
            "--stop-after-shards",
            str(stop_after_shards),
            "--gpus",
            "0",
            "--workers-per-gpu",
            str(worker_count),
            "--concurrent-games",
            str(contexts),
            "--simulations",
            str(arguments.simulations),
            "--black-simulations",
            str(arguments.simulations),
            "--white-simulations",
            str(arguments.simulations),
            "--top-k",
            str(arguments.top_k),
            "--hierarchical-first-candidates",
            str(arguments.hierarchical_first),
            "--hierarchical-seconds-per-first",
            str(arguments.hierarchical_second),
            "--progressive-widening-base",
            str(arguments.progressive_widening_base),
            "--progressive-widening-scale",
            str(arguments.progressive_widening_scale),
            "--candidate-gumbel",
            "0.25",
            "--mcts-cpu-threads",
            str(threads),
            "--cpu-affinity",
            "auto" if worker_count > 1 else affinity_text,
            "--batch-buckets",
            str(contexts),
            "--value-scale",
            "0",
            "--forced-opening-radius",
            "4",
            "--temperature-turns",
            "6",
            "--early-temperature",
            "1.0",
            "--opening-temperature",
            "1.0",
            "--generation",
            str(generation),
            "--seed",
            str(arguments.seed),
            "--compile-model",
            "--no-async-pipeline",
            "--no-deduplicate",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu)
        run_command(command, log_path, environment, parse_linux_cpu_list(affinity_text))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events_by_worker = {worker: [] for worker in range(worker_count)}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("event") == "selfplay_shard_committed":
            events_by_worker[int(item["worker"])].append(item)
    stable = {}
    if (
        burn_in_shards > 0
        and all(len(events) >= stop_after_shards for events in events_by_worker.values())
    ):
        worker_measurements = []
        for worker, events in events_by_worker.items():
            interval = (
                events[stop_after_shards - 1]["elapsed_seconds"]
                - events[burn_in_shards - 1]["elapsed_seconds"]
            )
            manifests = [
                json.loads(
                    (
                        run_root
                        / f"gen-{generation:04d}-worker-{worker:02d}"
                        f"-part-{part:04d}.manifest.json"
                    ).read_text(encoding="utf-8")
                )
                for part in range(burn_in_shards, stop_after_shards)
            ]
            measured_games = sum(item["games"] for item in manifests)
            measured_turns = sum(item["turns"] for item in manifests)
            worker_measurements.append({
                "worker": worker,
                "seconds": interval,
                "games": measured_games,
                "turns": measured_turns,
                "games_per_second": measured_games / interval,
                "turns_per_second": measured_turns / interval,
            })
        stable = {
            "games_per_second": sum(
                item["games_per_second"] for item in worker_measurements
            ),
            "turns_per_second": sum(
                item["turns_per_second"] for item in worker_measurements
            ),
            "games": sum(item["games"] for item in worker_measurements),
            "turns": sum(item["turns"] for item in worker_measurements),
            "burn_in_shards": burn_in_shards,
            "measure_shards": measure_shards,
            "workers": worker_measurements,
        }
    if not all(worker["audit"]["valid"] for worker in summary["workers"]):
        raise RuntimeError(f"full autotune replay audit failed: {summary_path}")
    return candidate | {
        "summary": str(summary_path),
        "games_per_second": summary["games_per_second"],
        "turns_per_second": summary["turns_per_second"],
        "stable_shard": stable,
        "unique_complete_game_rate": summary["diversity"]["unique_complete_game_rate"],
        "winner_counts": summary["winner_counts"],
        "length_buckets": summary["length_buckets"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--deployment-gpus",
        default="",
        help="physical GPU list whose CPU sharing should be reproduced, e.g. 0-7 is written as 0,1,2,3,4,5,6,7",
    )
    parser.add_argument("--contexts", default="128,256,384,512,768")
    parser.add_argument("--threads", default="8,12,16,24,32")
    parser.add_argument("--profile-turns", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=96)
    parser.add_argument("--hierarchical-first", type=int, default=24)
    parser.add_argument("--hierarchical-second", type=int, default=4)
    parser.add_argument("--progressive-widening-base", type=int, default=16)
    parser.add_argument("--progressive-widening-scale", type=float, default=2.5)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--full-validate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-games", type=int, default=0)
    parser.add_argument("--full-shard-games", type=int, default=128)
    parser.add_argument("--full-burn-in-shards", type=int, default=-1)
    parser.add_argument("--full-measure-shards", type=int, default=2)
    parser.add_argument("--full-candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261020)
    arguments = parser.parse_args()

    arguments.checkpoint = arguments.checkpoint.expanduser().resolve()
    arguments.output_dir = arguments.output_dir.expanduser().resolve()
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(arguments.checkpoint)
    validate_benchmark_output_root(arguments.output_dir)
    contexts = parse_positive_csv(arguments.contexts)
    threads = parse_positive_csv(arguments.threads)
    if min(arguments.profile_turns, arguments.repeats, arguments.simulations) < 1:
        raise ValueError("profile-turns, repeats, and simulations must be positive")
    if (
        arguments.full_shard_games < 1
        or arguments.full_measure_shards < 1
        or arguments.full_candidates < 1
        or arguments.full_burn_in_shards < -1
        or arguments.full_burn_in_shards == 0
    ):
        raise ValueError("invalid full-validation shard settings")
    if arguments.workers_per_gpu > 1 and not arguments.full_validate:
        raise ValueError("multi-worker tuning requires --full-validate")
    if arguments.hierarchical_first * arguments.hierarchical_second != arguments.top_k:
        raise ValueError("hierarchical candidates must multiply to top-k")
    deployment_gpu_ids = (
        sorted({
            int(item)
            for item in arguments.deployment_gpus.split(",")
            if item.strip()
        })
        if arguments.deployment_gpus
        else [arguments.gpu]
    )
    if min(deployment_gpu_ids) < 0 or arguments.gpu not in deployment_gpu_ids:
        raise ValueError("invalid deployment GPU list")
    affinity = detect_cpu_affinity(
        arguments.gpu,
        arguments.cpu_affinity,
        deployment_gpu_ids,
    )
    affinity_text = compress_cpu_list(affinity)
    quick_affinity = partition_cpu_affinity(
        affinity, arguments.workers_per_gpu
    )[0]
    quick_affinity_text = compress_cpu_list(quick_affinity)
    hardware = {
        "gpu": detect_gpu(arguments.gpu),
        "cpu_affinity": affinity,
        "physical_cpu_count": len(affinity),
        "deployment_gpu_ids": deployment_gpu_ids,
        "workers_per_gpu": arguments.workers_per_gpu,
    }
    root = arguments.output_dir
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        repository / "experiments/nebula_v4/profile_selfplay.py",
        repository / "experiments/nebula_v4/model.py",
        repository / "experiments/pair_native/selfplay.py",
        repository / "experiments/pair_native/pair_mcts.py",
        repository / "experiments/pair_native/pair_mcts.cpp",
    ]
    request = {
        "schema_version": 1,
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": sha256_file(arguments.checkpoint),
        "hardware": hardware,
        "source_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in source_paths
        },
        "contexts": contexts,
        "threads": threads,
        "profile_turns": arguments.profile_turns,
        "repeats": arguments.repeats,
        "simulations": arguments.simulations,
        "top_k": arguments.top_k,
        "hierarchical_first": arguments.hierarchical_first,
        "hierarchical_second": arguments.hierarchical_second,
        "progressive_widening_base": arguments.progressive_widening_base,
        "progressive_widening_scale": arguments.progressive_widening_scale,
        "workers_per_gpu": arguments.workers_per_gpu,
        "full_validate": arguments.full_validate,
        "full_games": arguments.full_games,
        "full_shard_games": arguments.full_shard_games,
        "full_burn_in_shards": arguments.full_burn_in_shards,
        "full_measure_shards": arguments.full_measure_shards,
        "full_candidates": arguments.full_candidates,
        "seed": arguments.seed,
    }
    validate_or_write_request(root, request)

    baseline_threads = min(
        threads, key=lambda item: abs(item - min(24, len(quick_affinity)))
    )
    stage_one = []
    for context_count in contexts:
        rows = [
            quick_profile(
                arguments,
                root,
                quick_affinity_text,
                context_count,
                baseline_threads,
                repeat,
            )
            for repeat in range(arguments.repeats)
        ]
        stage_one.append(aggregate_quick(rows))
    finalists = sorted(stage_one, key=lambda item: item["turns_per_second"], reverse=True)[:2]

    stage_two = []
    for finalist in finalists:
        for thread_count in threads:
            rows = [
                quick_profile(
                    arguments,
                    root,
                    quick_affinity_text,
                    finalist["contexts"],
                    thread_count,
                    repeat,
                )
                for repeat in range(arguments.repeats)
            ]
            stage_two.append(aggregate_quick(rows))
    ranked = sorted(stage_two, key=lambda item: item["turns_per_second"], reverse=True)

    full_results = []
    if arguments.full_validate:
        full_candidates = distinct_context_candidates(
            ranked, arguments.full_candidates
        )
        for index, candidate in enumerate(full_candidates):
            full_results.append(
                full_profile(arguments, root, affinity_text, candidate, index)
            )
        if any(not item["stable_shard"] for item in full_results):
            raise RuntimeError("full validation did not produce a stable shard interval")
        ranked_full = sorted(
            full_results,
            key=lambda item: item["stable_shard"].get(
                "turns_per_second", item["turns_per_second"]
            ),
            reverse=True,
        )
        winner = ranked_full[0]
    else:
        winner = ranked[0]

    profile = {
        "schema_version": 2,
        "name": "hardware-autotuned-v4",
        "hardware": hardware,
        "checkpoint_sha256": sha256_file(arguments.checkpoint),
        "concurrent_games": winner["contexts"],
        "batch_buckets": [winner["contexts"]],
        "mcts_cpu_threads_per_gpu": winner["mcts_cpu_threads"],
        "cpu_affinity_policy": "auto",
        "workers_per_gpu": arguments.workers_per_gpu,
        "shard_games": arguments.full_shard_games,
        "top_k": arguments.top_k,
        "hierarchical_first_candidates": arguments.hierarchical_first,
        "hierarchical_seconds_per_first": arguments.hierarchical_second,
        "progressive_widening_base": arguments.progressive_widening_base,
        "progressive_widening_scale": arguments.progressive_widening_scale,
        "candidate_gumbel": 0.25,
        "minimum_simulations": 400,
        "maximum_simulations": 1200,
        "measured_turns_per_second": winner["stable_shard"].get(
            "turns_per_second", winner["turns_per_second"]
        ) if full_results else winner["turns_per_second"],
        "validation": "full-shard" if full_results else "quick-search",
    }
    report = {
        "schema_version": 1,
        "purpose": "hardware self-play autotune; performance-only",
        "hardware": hardware,
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": sha256_file(arguments.checkpoint),
        "stage_one": stage_one,
        "stage_two": ranked,
        "full_results": full_results,
        "selected_profile": profile,
    }
    atomic_json(root / "autotune.json", report)
    atomic_json(root / "search-profile.json", profile)
    print(json.dumps({
        "autotune": str(root / "autotune.json"),
        "search_profile": str(root / "search-profile.json"),
        "gpu": hardware["gpu"]["name"],
        "concurrent_games": profile["concurrent_games"],
        "mcts_cpu_threads_per_gpu": profile["mcts_cpu_threads_per_gpu"],
        "workers_per_gpu": profile["workers_per_gpu"],
        "measured_turns_per_second": profile["measured_turns_per_second"],
        "validation": profile["validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
