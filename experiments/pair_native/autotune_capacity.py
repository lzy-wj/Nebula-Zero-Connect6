"""One-command hardware and capacity autotuning for Connect6 self-play.

The tuner changes execution parameters only.  Search semantics (MCTS budget,
candidate construction, widening, noise, and temperatures) remain fixed so a
faster result cannot be obtained by silently lowering training-data quality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.pair_native.autotune_selfplay import (
    atomic_json,
    compress_cpu_list,
    parse_positive_csv,
    validate_or_write_request,
)
from experiments.pair_native.async_loop import (
    GpuBusyError,
    require_idle_gpus,
    require_owned_gpus,
)
from experiments.pair_native.pair_replay import sha256_file
from experiments.pair_native.selfplay import (
    expand_gpu_workers,
    gpu_topology_cpu_affinity,
    topology_worker_affinities,
)
from experiments.pair_native.selfplay_sweep import validate_benchmark_output_root


def parse_gpu_inventory() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    inventory = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",", 5)]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        index, name, total, used, utilization, driver = fields
        inventory.append({
            "index": int(index),
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "utilization_percent": int(utilization),
            "driver": driver,
        })
    if not inventory:
        raise RuntimeError("no NVIDIA GPUs were discovered")
    return inventory


def select_gpu_ids(
    inventory: list[dict],
    requested: str,
    required: int,
    require_idle: bool,
    maximum_used_memory_mib: int,
    maximum_utilization: int,
) -> list[int]:
    by_id = {item["index"]: item for item in inventory}
    if requested.strip():
        selected = [int(item) for item in requested.split(",") if item.strip()]
        if len(selected) != len(set(selected)) or any(item not in by_id for item in selected):
            raise ValueError("--gpu-ids must contain distinct installed GPU IDs")
    else:
        candidates = list(inventory)
        if require_idle:
            candidates = [
                item for item in candidates
                if item["memory_used_mib"] <= maximum_used_memory_mib
                and item["utilization_percent"] <= maximum_utilization
            ]
        candidates.sort(key=lambda item: (
            item["memory_used_mib"], item["utilization_percent"], item["index"]
        ))
        # Alternate between CPU-locality domains.  This prevents an automatic
        # 2/4-GPU profile from crowding all workers onto one NUMA node merely
        # because those GPU indices happen to be lower.
        domains: dict[tuple[int, ...], list[int]] = {}
        for item in candidates:
            affinity = tuple(gpu_topology_cpu_affinity(item["index"]))
            domains.setdefault(affinity or (item["index"],), []).append(item["index"])
        selected = []
        while any(domains.values()):
            for domain in domains.values():
                if domain:
                    selected.append(domain.pop(0))
    if len(selected) < required:
        raise RuntimeError(
            f"need {required} usable GPUs but found {len(selected)}; "
            "pass --gpu-ids or --no-require-idle if intentional"
        )
    names = {by_id[index]["name"] for index in selected[:required]}
    if len(names) != 1:
        raise ValueError("one capacity profile cannot mix different GPU models")
    return selected[:required]


def ensure_gpus_idle(
    inventory: list[dict],
    gpu_ids: list[int],
    maximum_used_memory_mib: int,
    maximum_utilization: int,
) -> None:
    """Reject a capacity whose GPUs became occupied after initial discovery."""
    by_id = {item["index"]: item for item in inventory}
    busy = [
        gpu_id for gpu_id in gpu_ids
        if gpu_id not in by_id
        or by_id[gpu_id]["memory_used_mib"] > maximum_used_memory_mib
        or by_id[gpu_id]["utilization_percent"] > maximum_utilization
    ]
    if busy:
        raise RuntimeError(
            f"GPUs became busy before capacity launch: {busy}; rerun to select idle GPUs"
        )


def worker_affinities(gpu_ids: list[int], workers_per_gpu: int) -> tuple[list[int], list[list[int]]]:
    expanded = expand_gpu_workers(gpu_ids, workers_per_gpu)
    affinities = topology_worker_affinities(expanded)
    if len(affinities) != len(expanded) or any(not item for item in affinities):
        raise RuntimeError("failed to allocate CPU affinity for every GPU worker")
    flattened = [cpu for group in affinities for cpu in group]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("autotuned worker CPU affinities overlap")
    return expanded, affinities


def default_contexts(workers_per_gpu: int) -> int:
    return {1: 2048, 2: 1024, 3: 768, 4: 512}[workers_per_gpu]


def baseline_threads(cpu_count: int) -> int:
    return min(cpu_count, min(24, max(12, cpu_count // 4)))


def rounded_contexts(value: float) -> int:
    return max(128, int(math.floor(value / 128.0 + 0.5)) * 128)


def candidate_key(candidate: dict) -> tuple[int, int, int]:
    return (
        int(candidate["workers_per_gpu"]),
        int(candidate["contexts"]),
        int(candidate["mcts_cpu_threads"]),
    )


def candidate_name(candidate: dict) -> str:
    workers, contexts, threads = candidate_key(candidate)
    return f"w{workers}-c{contexts}-t{threads}"


def distinct_worker_candidates(rows: list[dict], limit: int) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        workers = int(row["workers_per_gpu"])
        if workers in seen:
            continue
        seen.add(workers)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def full_validation_candidates(
    ranked: list[dict], stage_one: list[dict], limit: int
) -> list[dict]:
    """Keep the best and baseline shape for each of the two best worker counts."""
    worker_winners = distinct_worker_candidates(ranked, 2)
    worker_counts = [int(item["workers_per_gpu"]) for item in worker_winners]
    selected = []
    seen = set()

    def append(row: dict) -> None:
        key = candidate_key(row)
        if key not in seen and len(selected) < limit:
            seen.add(key)
            selected.append(row)

    for row in worker_winners:
        append(row)
    for workers in worker_counts:
        baseline = next(
            (row for row in stage_one if int(row["workers_per_gpu"]) == workers),
            None,
        )
        if baseline:
            append(baseline)
    for row in ranked:
        if int(row["workers_per_gpu"]) in worker_counts:
            append(row)
    return selected


def interpolate_cumulative(points: list[tuple[float, float]], timestamp: float) -> float:
    if not points or timestamp < points[0][0] or timestamp > points[-1][0]:
        raise ValueError("timestamp is outside cumulative measurement points")
    for left, right in zip(points, points[1:]):
        if timestamp <= right[0]:
            span = right[0] - left[0]
            if span <= 0.0:
                return right[1]
            fraction = (timestamp - left[0]) / span
            return left[1] + fraction * (right[1] - left[1])
    return points[-1][1]


def quick_profile_group(arguments, capacity_root: Path, gpu_ids: list[int], candidate: dict) -> dict:
    name = candidate_name(candidate)
    root = capacity_root / "quick" / name
    root.mkdir(parents=True, exist_ok=True)
    expanded, affinities = worker_affinities(gpu_ids, candidate["workers_per_gpu"])
    outputs = [root / f"worker-{worker:02d}.json" for worker in range(len(expanded))]
    ready_paths = [root / f"worker-{worker:02d}.ready" for worker in range(len(expanded))]
    start_path = root / "start.signal"
    if not all(path.is_file() for path in outputs):
        for path in outputs + ready_paths + [start_path, root / "aggregate.json"]:
            path.unlink(missing_ok=True)
    processes = []
    handles = []
    try:
        if arguments.require_idle and not all(path.is_file() for path in outputs):
            require_idle_gpus(gpu_ids, f"quick autotune {name}")
        for worker, (gpu_id, affinity) in enumerate(zip(expanded, affinities)):
            output = outputs[worker]
            if output.is_file():
                continue
            log_path = root / f"worker-{worker:02d}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable,
                "-m",
                "experiments.nebula_v4.profile_selfplay",
                "--checkpoint", str(arguments.checkpoint),
                "--contexts", str(candidate["contexts"]),
                "--turns", str(arguments.profile_turns),
                "--simulations", str(arguments.simulations),
                "--top-k", str(arguments.top_k),
                "--hierarchical-first", str(arguments.hierarchical_first),
                "--hierarchical-second", str(arguments.hierarchical_second),
                "--progressive-widening-base", str(arguments.progressive_widening_base),
                "--progressive-widening-scale", str(arguments.progressive_widening_scale),
                "--mcts-cpu-threads", str(candidate["mcts_cpu_threads"]),
                "--cpu-affinity", compress_cpu_list(affinity),
                "--batch-buckets", str(candidate["contexts"]),
                "--candidate-gumbel", str(arguments.candidate_gumbel),
                "--compile-model",
                "--seed", str(arguments.seed + worker),
                "--output", str(output),
                "--ready-file", str(ready_paths[worker]),
                "--start-file", str(start_path),
                "--barrier-timeout-seconds", str(arguments.quick_timeout_seconds),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker, process, log_path))
        deadline = time.monotonic() + arguments.quick_timeout_seconds
        next_ownership_check = 0.0
        barrier_released = not processes
        while processes and any(process.poll() is None for _, process, _ in processes):
            now = time.monotonic()
            if arguments.require_idle and now >= next_ownership_check:
                require_owned_gpus(gpu_ids, os.getpid(), f"quick autotune {name}")
                next_ownership_check = now + 10.0
            failures = [
                {"worker": worker, "returncode": process.returncode, "log": str(log)}
                for worker, process, log in processes
                if process.poll() not in (None, 0)
            ]
            if failures:
                raise RuntimeError(f"quick candidate failed: {failures}")
            if not barrier_released and all(path.is_file() for path in ready_paths):
                temporary = start_path.with_suffix(".tmp")
                temporary.write_text(f"{time.time()}\n", encoding="utf-8")
                temporary.replace(start_path)
                barrier_released = True
            if time.monotonic() >= deadline:
                waiting = [
                    worker for worker, process, _ in processes if process.poll() is None
                ]
                raise TimeoutError(f"quick candidate timed out; workers={waiting}")
            time.sleep(0.05)
        failures = []
        for worker, process, log in processes:
            returncode = process.wait()
            if returncode:
                failures.append({
                    "worker": worker,
                    "returncode": returncode,
                    "log": str(log),
                })
        if failures:
            raise RuntimeError(f"quick candidate failed: {failures}")
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()

    rows = [
        json.loads((root / f"worker-{worker:02d}.json").read_text(encoding="utf-8"))
        for worker in range(len(expanded))
    ]
    result = dict(candidate) | {
        "gpu_ids": gpu_ids,
        "worker_count": len(rows),
        "turns_per_second": sum(row["turns_per_second"] for row in rows),
        "simulations_per_second": sum(row["simulations_per_second"] for row in rows),
        "per_gpu_turns_per_second": (
            sum(row["turns_per_second"] for row in rows) / len(gpu_ids)
        ),
        "callback_share_percent": sum(
            row["callback_share_of_search_percent"] for row in rows
        ) / len(rows),
        "worker_results": [
            {
                "worker": worker,
                "physical_gpu": expanded[worker],
                "cpu_affinity": affinities[worker],
                "turns_per_second": row["turns_per_second"],
                "average_cpu_cores": row["average_cpu_cores"],
                "result": str(root / f"worker-{worker:02d}.json"),
            }
            for worker, row in enumerate(rows)
        ],
    }
    atomic_json(root / "aggregate.json", result)
    return result


def full_profile_group(
    arguments,
    capacity_root: Path,
    gpu_ids: list[int],
    candidate: dict,
    generation: int,
) -> dict:
    name = candidate_name(candidate)
    root = capacity_root / "full" / name
    root.mkdir(parents=True, exist_ok=True)
    expanded, affinities = worker_affinities(gpu_ids, candidate["workers_per_gpu"])
    shard_games = int(arguments.full_shard_games)
    burn_in_shards = math.ceil(candidate["contexts"] / shard_games)
    stop_after_shards = burn_in_shards + int(arguments.full_measure_shards)
    games = len(expanded) * (
        candidate["contexts"] + stop_after_shards * shard_games
    )
    summary_path = root / f"gen-{generation:04d}-summary.json"
    log_path = root / "run.log"
    if not summary_path.is_file():
        if arguments.require_idle:
            require_idle_gpus(gpu_ids, f"full autotune {name}")
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.pair_native.selfplay",
            "--output-dir", str(root),
            "--checkpoint", str(arguments.checkpoint),
            "--games", str(games),
            "--shard-games", str(shard_games),
            "--stop-after-shards", str(stop_after_shards),
            "--gpus", ",".join(str(item) for item in gpu_ids),
            "--workers-per-gpu", str(candidate["workers_per_gpu"]),
            "--worker-cpu-affinities", ";".join(
                compress_cpu_list(item) for item in affinities
            ),
            "--concurrent-games", str(candidate["contexts"]),
            "--simulations", str(arguments.simulations),
            "--black-simulations", str(arguments.simulations),
            "--white-simulations", str(arguments.simulations),
            "--top-k", str(arguments.top_k),
            "--hierarchical-first-candidates", str(arguments.hierarchical_first),
            "--hierarchical-seconds-per-first", str(arguments.hierarchical_second),
            "--progressive-widening-base", str(arguments.progressive_widening_base),
            "--progressive-widening-scale", str(arguments.progressive_widening_scale),
            "--candidate-gumbel", str(arguments.candidate_gumbel),
            "--mcts-cpu-threads", str(candidate["mcts_cpu_threads"]),
            "--batch-buckets", str(candidate["contexts"]),
            "--value-scale", "0",
            "--forced-opening-radius", "4",
            "--temperature-turns", "6",
            "--early-temperature", "1.0",
            "--opening-temperature", "1.0",
            "--generation", str(generation),
            "--seed", str(arguments.seed),
            "--worker-timeout-seconds", str(arguments.worker_timeout_seconds),
            "--compile-model",
            "--no-async-pipeline",
            "--no-deduplicate",
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    if arguments.require_idle:
                        require_owned_gpus(
                            gpu_ids, os.getpid(), f"full autotune {name}"
                        )
                    time.sleep(10.0)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, 15)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, 9)
                        process.wait()
                raise
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not all(worker["audit"]["valid"] for worker in summary["workers"]):
        raise RuntimeError(f"full replay audit failed: {summary_path}")
    if summary["diversity"]["unique_complete_game_rate"] < arguments.minimum_unique_rate:
        raise RuntimeError(f"full replay diversity failed: {summary_path}")

    events_by_worker = defaultdict(list)
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("event") == "selfplay_shard_committed":
            events_by_worker[int(item["worker"])].append(item)
    if len(events_by_worker) != len(expanded) or any(
        len(events_by_worker[worker]) < stop_after_shards
        for worker in range(len(expanded))
    ):
        raise RuntimeError("full run did not produce every stable shard interval")

    measurements = []
    cumulative_by_worker = {}
    for worker in range(len(expanded)):
        events = events_by_worker[worker]
        seconds = (
            events[stop_after_shards - 1]["elapsed_seconds"]
            - events[burn_in_shards - 1]["elapsed_seconds"]
        )
        manifests = [
            json.loads((
                root
                / f"gen-{generation:04d}-worker-{worker:02d}-part-{part:04d}.manifest.json"
            ).read_text(encoding="utf-8"))
            for part in range(burn_in_shards, stop_after_shards)
        ]
        measured_games = sum(item["games"] for item in manifests)
        measured_turns = sum(item["turns"] for item in manifests)
        if any("unix_seconds" not in event for event in events):
            raise RuntimeError("full run is missing global shard timestamps")
        turn_points = [(float(events[burn_in_shards - 1]["unix_seconds"]), 0.0)]
        game_points = [(float(events[burn_in_shards - 1]["unix_seconds"]), 0.0)]
        cumulative_turns = 0
        cumulative_games = 0
        for offset, manifest in enumerate(manifests, start=burn_in_shards):
            cumulative_turns += int(manifest["turns"])
            cumulative_games += int(manifest["games"])
            timestamp = float(events[offset]["unix_seconds"])
            turn_points.append((timestamp, float(cumulative_turns)))
            game_points.append((timestamp, float(cumulative_games)))
        cumulative_by_worker[worker] = {
            "turns": turn_points,
            "games": game_points,
        }
        measurements.append({
            "worker": worker,
            "physical_gpu": expanded[worker],
            "seconds": seconds,
            "games": measured_games,
            "turns": measured_turns,
            "games_per_second": measured_games / seconds,
            "turns_per_second": measured_turns / seconds,
        })
    common_start = max(
        points["turns"][0][0] for points in cumulative_by_worker.values()
    )
    common_end = min(
        points["turns"][-1][0] for points in cumulative_by_worker.values()
    )
    common_seconds = common_end - common_start
    if common_seconds <= 0.0:
        raise RuntimeError("full workers have no common stable measurement window")
    if common_seconds < arguments.minimum_common_window_seconds:
        raise RuntimeError(
            "full common stable window is too short: "
            f"{common_seconds:.1f}s < {arguments.minimum_common_window_seconds:.1f}s"
        )
    common_turns = sum(
        interpolate_cumulative(points["turns"], common_end)
        - interpolate_cumulative(points["turns"], common_start)
        for points in cumulative_by_worker.values()
    )
    common_games = sum(
        interpolate_cumulative(points["games"], common_end)
        - interpolate_cumulative(points["games"], common_start)
        for points in cumulative_by_worker.values()
    )
    result = dict(candidate) | {
        "gpu_ids": gpu_ids,
        "worker_count": len(expanded),
        "games_per_second": common_games / common_seconds,
        "turns_per_second": common_turns / common_seconds,
        "per_gpu_turns_per_second": (
            common_turns / common_seconds / len(gpu_ids)
        ),
        "measured_games": sum(item["games"] for item in measurements),
        "measured_turns": sum(item["turns"] for item in measurements),
        "burn_in_shards": burn_in_shards,
        "measure_shards": arguments.full_measure_shards,
        "common_window": {
            "start_unix": common_start,
            "end_unix": common_end,
            "seconds": common_seconds,
            "interpolated_games": common_games,
            "interpolated_turns": common_turns,
        },
        "workers": measurements,
        "winner_counts": summary["winner_counts"],
        "length_buckets": summary["length_buckets"],
        "unique_complete_game_rate": summary["diversity"]["unique_complete_game_rate"],
        "summary": str(summary_path),
    }
    atomic_json(root / "stable-result.json", result)
    return result


def add_candidate(
    arguments,
    capacity_root: Path,
    gpu_ids: list[int],
    candidate: dict,
    rows: dict[tuple[int, int, int], dict],
) -> dict | None:
    key = candidate_key(candidate)
    if key in rows:
        return rows[key]
    print(json.dumps({
        "event": "quick_candidate_started",
        "capacity": len(gpu_ids),
        **candidate,
    }), flush=True)
    try:
        row = quick_profile_group(arguments, capacity_root, gpu_ids, candidate)
    except GpuBusyError:
        raise
    except Exception as error:  # Candidate failures are expected during portable sweeps.
        row = dict(candidate) | {"error": repr(error)}
        atomic_json(capacity_root / "quick" / candidate_name(candidate) / "failure.json", row)
    rows[key] = row
    print(json.dumps({
        "event": "quick_candidate_completed",
        "capacity": len(gpu_ids),
        **candidate,
        "turns_per_second": row.get("turns_per_second"),
        "error": row.get("error"),
    }), flush=True)
    return None if "error" in row else row


def tune_capacity(arguments, root: Path, inventory: list[dict], gpu_ids: list[int]) -> dict:
    capacity = len(gpu_ids)
    capacity_root = root / f"{capacity}gpu"
    capacity_root.mkdir(parents=True, exist_ok=True)
    quick_rows: dict[tuple[int, int, int], dict] = {}

    # Stage 1: compare worker counts using one hardware-informed point each.
    stage_one = []
    for workers in arguments.worker_candidates:
        _, affinities = worker_affinities(gpu_ids, workers)
        candidate = {
            "workers_per_gpu": workers,
            "contexts": default_contexts(workers),
            "mcts_cpu_threads": baseline_threads(min(map(len, affinities))),
        }
        row = add_candidate(arguments, capacity_root, gpu_ids, candidate, quick_rows)
        if row:
            stage_one.append(row)
    if not stage_one:
        raise RuntimeError(f"all quick worker candidates failed for {capacity} GPUs")

    # Stage 2: vary contexts around the two best worker-count candidates.
    stage_two = []
    worker_finalists = sorted(
        stage_one, key=lambda item: item["turns_per_second"], reverse=True
    )[:2]
    for finalist in worker_finalists:
        for scale in (0.75, 1.0, 1.25):
            candidate = {
                "workers_per_gpu": finalist["workers_per_gpu"],
                "contexts": rounded_contexts(finalist["contexts"] * scale),
                "mcts_cpu_threads": finalist["mcts_cpu_threads"],
            }
            row = add_candidate(arguments, capacity_root, gpu_ids, candidate, quick_rows)
            if row:
                stage_two.append(row)

    # Stage 3: sweep CPU threads only around the two best shapes.
    shape_finalists = distinct_worker_candidates(
        sorted(stage_two, key=lambda item: item["turns_per_second"], reverse=True),
        2,
    )
    for finalist in shape_finalists:
        _, affinities = worker_affinities(gpu_ids, finalist["workers_per_gpu"])
        maximum_threads = min(map(len, affinities))
        for threads in sorted({min(maximum_threads, item) for item in (8, 12, 16, 24)}):
            candidate = {
                "workers_per_gpu": finalist["workers_per_gpu"],
                "contexts": finalist["contexts"],
                "mcts_cpu_threads": threads,
            }
            add_candidate(arguments, capacity_root, gpu_ids, candidate, quick_rows)

    successful = sorted(
        (row for row in quick_rows.values() if "error" not in row),
        key=lambda item: item["turns_per_second"],
        reverse=True,
    )
    full_results = []
    if arguments.mode == "full":
        full_candidates = full_validation_candidates(
            successful, stage_one, arguments.full_candidates
        )
        for index, candidate in enumerate(full_candidates):
            full_results.append(full_profile_group(
                arguments,
                capacity_root,
                gpu_ids,
                candidate,
                9700 + capacity * 10 + index,
            ))
        ranked = sorted(
            full_results, key=lambda item: item["turns_per_second"], reverse=True
        )
    else:
        ranked = successful
    winner = ranked[0]
    gpu_records = [
        next(item for item in inventory if item["index"] == gpu_id)
        for gpu_id in gpu_ids
    ]
    profile = {
        "schema_version": 3,
        "name": f"hardware-autotuned-{capacity}gpu-v4",
        "hardware": {
            "gpu_count": capacity,
            "gpu_name": gpu_records[0]["name"],
            "gpu_memory_mib": gpu_records[0]["memory_total_mib"],
            "driver": gpu_records[0]["driver"],
        },
        "checkpoint_sha256": sha256_file(arguments.checkpoint),
        "workers_per_gpu": winner["workers_per_gpu"],
        "concurrent_games": winner["contexts"],
        "shard_games": arguments.full_shard_games,
        "batch_buckets": [winner["contexts"]],
        "mcts_cpu_threads_per_gpu": winner["mcts_cpu_threads"],
        "cpu_affinity_policy": "auto",
        "top_k": arguments.top_k,
        "hierarchical_first_candidates": arguments.hierarchical_first,
        "hierarchical_seconds_per_first": arguments.hierarchical_second,
        "progressive_widening_base": arguments.progressive_widening_base,
        "progressive_widening_scale": arguments.progressive_widening_scale,
        "candidate_gumbel": arguments.candidate_gumbel,
        "minimum_simulations": arguments.simulations,
        "maximum_simulations": max(1200, arguments.simulations),
        "measured_aggregate_turns_per_second": winner["turns_per_second"],
        "measured_turns_per_second_per_gpu": winner["per_gpu_turns_per_second"],
        "measured_games_per_second": winner.get("games_per_second"),
        "validation": "full-shard" if full_results else "quick-search",
    }
    report = {
        "schema_version": 1,
        "capacity": capacity,
        "gpu_ids": gpu_ids,
        "stage_one": stage_one,
        "quick_results": successful,
        "full_results": full_results,
        "selected_profile": profile,
    }
    atomic_json(capacity_root / "report.json", report)
    atomic_json(root / "profiles" / f"{capacity}gpu.json", profile)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capacities", default="1,2,4,8")
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--worker-candidates", default="1,2,3,4")
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--profile-turns", type=int, default=4)
    parser.add_argument("--quick-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--full-candidates", type=int, default=4)
    parser.add_argument("--full-shard-games", type=int, default=128)
    parser.add_argument("--full-measure-shards", type=int, default=4)
    parser.add_argument("--minimum-unique-rate", type=float, default=0.95)
    parser.add_argument("--minimum-common-window-seconds", type=float, default=30.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=14_400.0)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=96)
    parser.add_argument("--hierarchical-first", type=int, default=24)
    parser.add_argument("--hierarchical-second", type=int, default=4)
    parser.add_argument("--progressive-widening-base", type=int, default=16)
    parser.add_argument("--progressive-widening-scale", type=float, default=2.5)
    parser.add_argument("--candidate-gumbel", type=float, default=0.25)
    parser.add_argument("--require-idle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=1024)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261210)
    arguments = parser.parse_args()

    arguments.checkpoint = arguments.checkpoint.expanduser().resolve()
    arguments.output_dir = arguments.output_dir.expanduser().resolve()
    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(arguments.checkpoint)
    capacities = parse_positive_csv(arguments.capacities)
    arguments.worker_candidates = parse_positive_csv(arguments.worker_candidates)
    if max(arguments.worker_candidates) > 4:
        raise ValueError("worker candidates must be in [1, 4]")
    if (
        arguments.profile_turns < 1
        or arguments.full_candidates < 1
        or arguments.full_shard_games < 1
        or arguments.full_measure_shards < 1
        or arguments.simulations < 1
        or arguments.quick_timeout_seconds <= 0.0
        or arguments.worker_timeout_seconds <= 0.0
        or arguments.minimum_common_window_seconds <= 0.0
        or not 0.0 < arguments.minimum_unique_rate <= 1.0
    ):
        raise ValueError("invalid autotune settings")
    if arguments.hierarchical_first * arguments.hierarchical_second != arguments.top_k:
        raise ValueError("hierarchical candidate dimensions must multiply to top-k")

    validate_benchmark_output_root(arguments.output_dir)
    inventory = parse_gpu_inventory()
    selected = select_gpu_ids(
        inventory,
        arguments.gpu_ids,
        max(capacities),
        arguments.require_idle,
        arguments.maximum_used_memory_mib,
        arguments.maximum_utilization,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
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
        "source_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in source_paths
        },
        "inventory": [
            {
                key: item[key]
                for key in ("index", "name", "memory_total_mib", "driver")
            }
            for item in inventory
        ],
        "selected_gpu_ids": selected,
        "capacities": capacities,
        "worker_candidates": arguments.worker_candidates,
        "mode": arguments.mode,
        "profile_turns": arguments.profile_turns,
        "quick_timeout_seconds": arguments.quick_timeout_seconds,
        "full_candidates": arguments.full_candidates,
        "full_shard_games": arguments.full_shard_games,
        "full_measure_shards": arguments.full_measure_shards,
        "minimum_unique_rate": arguments.minimum_unique_rate,
        "minimum_common_window_seconds": arguments.minimum_common_window_seconds,
        "worker_timeout_seconds": arguments.worker_timeout_seconds,
        "simulations": arguments.simulations,
        "top_k": arguments.top_k,
        "hierarchical_first": arguments.hierarchical_first,
        "hierarchical_second": arguments.hierarchical_second,
        "progressive_widening_base": arguments.progressive_widening_base,
        "progressive_widening_scale": arguments.progressive_widening_scale,
        "candidate_gumbel": arguments.candidate_gumbel,
        "require_idle": arguments.require_idle,
        "maximum_used_memory_mib": arguments.maximum_used_memory_mib,
        "maximum_utilization": arguments.maximum_utilization,
        "seed": arguments.seed,
    }
    validate_or_write_request(arguments.output_dir, request)

    reports = []
    started = time.perf_counter()
    for capacity in capacities:
        if arguments.require_idle:
            ensure_gpus_idle(
                parse_gpu_inventory(), selected[:capacity],
                arguments.maximum_used_memory_mib,
                arguments.maximum_utilization,
            )
        print(json.dumps({
            "event": "capacity_autotune_started",
            "capacity": capacity,
            "gpu_ids": selected[:capacity],
        }), flush=True)
        report = tune_capacity(
            arguments, arguments.output_dir, inventory, selected[:capacity]
        )
        reports.append(report)
        profile = report["selected_profile"]
        print(json.dumps({
            "event": "capacity_autotune_completed",
            "capacity": capacity,
            "workers_per_gpu": profile["workers_per_gpu"],
            "concurrent_games": profile["concurrent_games"],
            "mcts_cpu_threads": profile["mcts_cpu_threads_per_gpu"],
            "aggregate_turns_per_second": profile["measured_aggregate_turns_per_second"],
            "validation": profile["validation"],
        }), flush=True)
    summary = {
        "schema_version": 1,
        "seconds": time.perf_counter() - started,
        "profiles": [report["selected_profile"] for report in reports],
    }
    atomic_json(arguments.output_dir / "summary.json", summary)
    print(json.dumps({
        "summary": str(arguments.output_dir / "summary.json"),
        "profiles": str(arguments.output_dir / "profiles"),
        "seconds": summary["seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
