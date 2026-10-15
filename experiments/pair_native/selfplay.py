"""Multi-GPU native-Pair self-play generator with strict Parquet output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.game_audit import BOARD_POINTS, BOARD_SIZE, check_win
from experiments.pair_native.model import load_checkpoint, save_random_checkpoint
from experiments.pair_native.pair_mcts import (
    PASS_MOVE,
    PairMCTSEngine,
    TorchPairEvaluator,
    ensure_library,
)
from experiments.pair_native.pair_replay import (
    PairGameRecord,
    PairTurnRecord,
    audit_shard,
    board_payloads,
    sha256_file,
    write_shard,
)


PRODUCTION_MIN_SIMULATIONS = 400
PRODUCTION_MAX_SIMULATIONS = 1200


def stable_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def apply_action(board, player, first, second):
    opening = not np.any(board)
    if not 0 <= first < BOARD_POINTS or board.flat[first] != 0:
        raise ValueError(f"illegal first move {first}")
    if opening and second != PASS_MOVE:
        raise ValueError("opening action must be a singleton")
    board.flat[first] = player
    moves = [first]
    if check_win(board, first, player):
        if second != PASS_MOVE:
            raise ValueError("second stone supplied after a first-stone win")
        return moves, player
    if board.size == np.count_nonzero(board):
        return moves, 0
    if opening:
        return moves, None
    if not first < second < BOARD_POINTS or board.flat[second] != 0:
        raise ValueError(f"illegal canonical second move {(first, second)}")
    board.flat[second] = player
    moves.append(second)
    if check_win(board, second, player):
        return moves, player
    if board.size == np.count_nonzero(board):
        return moves, 0
    return moves, None


def game_length_bucket(stones: int) -> str:
    """Operational game-length bins used by replay balancing and monitoring."""
    if stones <= 40:
        return "short"
    if stones <= 80:
        return "medium"
    if stones <= 140:
        return "long"
    return "very_long"


def opening_distribution(counts: Counter, games: int) -> dict:
    """Compact, mergeable opening-diversity metrics for monitoring/control."""
    total = int(games)
    if total <= 0 or not counts:
        return {
            "counts": {},
            "unique": 0,
            "entropy_bits": 0.0,
            "effective_openings": 0.0,
            "top_move": None,
            "top_count": 0,
            "top_share": 0.0,
        }
    top_move, top_count = counts.most_common(1)[0]
    probabilities = [count / total for count in counts.values() if count > 0]
    entropy = -sum(value * math.log2(value) for value in probabilities)
    return {
        "counts": {str(int(move)): int(count) for move, count in sorted(counts.items())},
        "unique": len(counts),
        "entropy_bits": entropy,
        "effective_openings": 2.0**entropy,
        "top_move": int(top_move),
        "top_count": int(top_count),
        "top_share": top_count / total,
    }


def game_fingerprint(moves: list[int]) -> str:
    payload = np.asarray(moves, dtype="<i2").tobytes()
    return hashlib.sha256(payload).hexdigest()


def validate_simulation_budgets(
    simulations: int,
    black_simulations: int,
    white_simulations: int,
    allow_nonproduction: bool = False,
) -> None:
    budgets = (int(simulations), int(black_simulations), int(white_simulations))
    if any(budget < 1 for budget in budgets):
        raise ValueError("self-play simulation budgets must be positive")
    if not allow_nonproduction and any(
        not PRODUCTION_MIN_SIMULATIONS <= budget <= PRODUCTION_MAX_SIMULATIONS
        for budget in budgets
    ):
        raise ValueError(
            "production self-play budgets must all be in [400, 1200]; "
            "use --allow-nonproduction-simulations only for isolated tests"
        )


def parse_linux_cpu_list(specification: str) -> list[int]:
    """Parse Linux sysfs CPU lists such as ``0-3,192-195``."""
    cpus: set[int] = set()
    for item in specification.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return sorted(cpus)


def cpu_thread_siblings(cpu: int) -> list[int]:
    path = Path(f"/sys/devices/system/cpu/cpu{int(cpu)}/topology/thread_siblings_list")
    try:
        return parse_linux_cpu_list(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [int(cpu)]


def gpu_topology_cpu_affinity(gpu_id: int, topology_text: str | None = None) -> list[int]:
    """Return the CPUs local to a physical GPU according to nvidia-smi."""
    if topology_text is None:
        try:
            topology_text = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return []
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", topology_text)
    target = f"GPU{int(gpu_id)}"
    pattern = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")
    for line in clean.splitlines():
        fields = line.split()
        if not fields or fields[0] != target:
            continue
        candidates = [
            parse_linux_cpu_list(field)
            for field in fields[1:]
            if pattern.fullmatch(field)
        ]
        if candidates:
            return max(candidates, key=len)
    return []


def topology_worker_affinities(
    gpu_ids: list[int],
    include_smt: bool = False,
    topology_text: str | None = None,
    allowed_cpus: set[int] | None = None,
) -> list[list[int]]:
    """Partition physical cores among workers sharing each GPU NUMA domain."""
    if not gpu_ids:
        return []
    available = (
        set(allowed_cpus)
        if allowed_cpus is not None
        else (
            set(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else set(range(os.cpu_count() or 1))
        )
    )
    local_sets = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        local = [
            cpu
            for cpu in gpu_topology_cpu_affinity(gpu_id, topology_text)
            if cpu in available
        ]
        if not local:
            local = [
                cpu
                for cpu in planned_cpu_affinity(worker_id, gpu_id, include_smt)
                if cpu in available
            ]
        local_sets.append(local or sorted(available))

    results: list[list[int]] = [[] for _ in gpu_ids]
    domains: dict[tuple[int, ...], list[int]] = {}
    for worker_id, local in enumerate(local_sets):
        domains.setdefault(tuple(local), []).append(worker_id)
    for local_tuple, workers in domains.items():
        local = set(local_tuple)
        core_groups = []
        seen: set[tuple[int, ...]] = set()
        for cpu in sorted(local):
            siblings = tuple(
                sibling
                for sibling in cpu_thread_siblings(cpu)
                if sibling in local
            ) or (cpu,)
            if siblings in seen:
                continue
            seen.add(siblings)
            core_groups.append(siblings)
        for position, worker_id in enumerate(workers):
            start = len(core_groups) * position // len(workers)
            end = len(core_groups) * (position + 1) // len(workers)
            assigned_groups = core_groups[start:end]
            if include_smt:
                assigned = sorted({cpu for group in assigned_groups for cpu in group})
            else:
                assigned = [group[0] for group in assigned_groups]
            results[worker_id] = assigned
    return results


def expand_gpu_workers(gpu_ids: list[int], workers_per_gpu: int) -> list[int]:
    if (
        not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
        or any(int(gpu) < 0 for gpu in gpu_ids)
    ):
        raise ValueError("distinct non-negative physical GPU IDs are required")
    if not 1 <= workers_per_gpu <= 4:
        raise ValueError("workers_per_gpu must be in [1, 4]")
    return [gpu for gpu in gpu_ids for _ in range(workers_per_gpu)]


def should_auto_cpu_affinity(specification: str, workers_per_gpu: int) -> bool:
    value = str(specification).strip().lower()
    return value == "auto" or (not value and workers_per_gpu > 1)


def parse_worker_cpu_affinities(
    specification: str, expected_workers: int
) -> list[list[int]]:
    if not specification.strip():
        return []
    groups = [
        parse_linux_cpu_list(item.strip())
        for item in specification.split(";")
        if item.strip()
    ]
    if len(groups) != expected_workers or any(not group for group in groups):
        raise ValueError(
            "--worker-cpu-affinities must provide one non-empty CPU list per worker"
        )
    flattened = [cpu for group in groups for cpu in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("worker CPU affinities must be disjoint")
    return groups


def planned_cpu_affinity(
    worker_id: int,
    gpu_id: int,
    include_smt: bool = False,
    sibling_resolver=None,
    available_cpus: list[int] | None = None,
) -> list[int]:
    """Return a topology-neutral fallback based on the process CPU set.

    ``topology_worker_affinities`` partitions this common set when several
    workers share a host.  No GPU index or host-specific core range is assumed.
    """
    del worker_id, gpu_id
    available = sorted(set(
        available_cpus
        if available_cpus is not None
        else (
            os.sched_getaffinity(0)
            if hasattr(os, "sched_getaffinity")
            else range(os.cpu_count() or 1)
        )
    ))
    resolver = sibling_resolver or cpu_thread_siblings
    groups: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for cpu in available:
        group = tuple(sorted(set(int(item) for item in resolver(cpu)))) or (cpu,)
        if group not in seen:
            seen.add(group)
            groups.append(group)
    if include_smt:
        return sorted({cpu for group in groups for cpu in group})
    return [next((cpu for cpu in group if cpu in available), group[0]) for group in groups]


def assign_cpu_affinity(
    worker_id: int,
    gpu_id: int,
    include_smt: bool = False,
    explicit_cpus: list[int] | None = None,
):
    if not hasattr(os, "sched_setaffinity"):
        return []
    cpus = (
        list(explicit_cpus)
        if explicit_cpus
        else planned_cpu_affinity(worker_id, gpu_id, include_smt)
    )
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        return []
    return cpus


def worker_main(config: dict, worker_id: int, gpu_id: int, game_indices: list[int], result_queue):
    affinity = assign_cpu_affinity(
        worker_id,
        gpu_id,
        include_smt=config["include_smt_affinity"],
        explicit_cpus=config["worker_cpu_affinities"][worker_id],
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    model, checkpoint = load_checkpoint(config["checkpoint"])
    evaluator = TorchPairEvaluator(
        model,
        device,
        top_k=config["top_k"],
        candidate_gumbel=config["candidate_gumbel"],
        seed=config["seed"] + worker_id * 10_000,
        compile_model=config["compile_model"],
        deduplicate=config["deduplicate"],
        pad_batch_to=config["pad_batch_to"],
        batch_buckets=config["batch_buckets"],
        value_scale=config["value_scale"],
        hierarchical_first_candidates=config["hierarchical_first_candidates"],
        hierarchical_seconds_per_first=config["hierarchical_seconds_per_first"],
    )
    engine = PairMCTSEngine(
        evaluator,
        library_path=config["library"],
        top_k=config["top_k"],
        cpuct=config["cpuct"],
        root_noise_epsilon=config["root_noise_epsilon"],
        root_dirichlet_alpha=config["root_dirichlet_alpha"],
        random_candidates=config["random_candidates"],
        cpu_threads=config["mcts_cpu_threads"],
        progressive_widening_base=config["progressive_widening_base"],
        progressive_widening_scale=config["progressive_widening_scale"],
        async_pipeline=config["async_pipeline"],
    )
    search_hash = stable_hash({
        key: config[key]
        for key in (
            "simulations",
            "black_simulations",
            "white_simulations",
            "adaptive_simulations",
            "adaptive_min_simulations",
            "adaptive_mid_simulations",
            "adaptive_confidence_threshold",
            "top_k",
            "hierarchical_first_candidates",
            "hierarchical_seconds_per_first",
            "progressive_widening_base",
            "progressive_widening_scale",
            "cpuct",
            "root_noise_epsilon",
            "root_dirichlet_alpha",
            "candidate_gumbel",
            "temperature_turns",
            "early_temperature",
            "opening_temperature",
            "value_scale",
            "forced_opening_radius",
            "random_candidates",
            "mcts_cpu_threads",
            "batch_buckets",
            "async_pipeline",
        )
    })
    pending_indices = iter(game_indices)
    games = []
    turns = []
    shard_manifests = []
    shard_audits = []
    flushed_games = 0

    def shard_metadata():
        return {
            "worker": worker_id,
            "gpu": gpu_id,
            "affinity": affinity,
            "checkpoint": config["checkpoint"],
            "model_config": checkpoint["model_config"],
            "execution": {
                "concurrent_games": config["concurrent_games"],
                "workers_per_gpu": config["workers_per_gpu"],
                "include_smt_affinity": config["include_smt_affinity"],
                "pad_batch_to": config["pad_batch_to"],
                "shard_games": config["shard_games"],
                "compile_model": config["compile_model"],
                "stop_after_shards": config["stop_after_shards"],
            },
            "search": {
                key: config[key]
                for key in (
                    "simulations",
                    "black_simulations",
                    "white_simulations",
                    "adaptive_simulations",
                    "adaptive_min_simulations",
                    "adaptive_mid_simulations",
                    "adaptive_confidence_threshold",
                    "top_k",
                    "hierarchical_first_candidates",
                    "hierarchical_seconds_per_first",
                    "progressive_widening_base",
                    "progressive_widening_scale",
                    "cpuct",
                    "root_noise_epsilon",
                    "root_dirichlet_alpha",
                    "candidate_gumbel",
                    "temperature_turns",
                    "early_temperature",
                    "opening_temperature",
                    "value_scale",
                    "forced_opening_radius",
                    "random_candidates",
                    "mcts_cpu_threads",
                    "batch_buckets",
                    "async_pipeline",
                )
            },
        }

    def flush_completed(force: bool = False):
        nonlocal turns, flushed_games
        available = len(games) - flushed_games
        target = config["shard_games"]
        if available <= 0 or (not force and (target <= 0 or available < target)):
            return
        while available > 0 and (force or available >= target):
            count = available if target <= 0 else min(target, available)
            chunk_games = games[flushed_games:flushed_games + count]
            game_ids = {game.game_id for game in chunk_games}
            chunk_turns = [turn for turn in turns if turn.game_id in game_ids]
            if target <= 0:
                shard_name = f"gen-{config['generation']:04d}-worker-{worker_id:02d}"
            else:
                part = len(shard_manifests)
                shard_name = (
                    f"gen-{config['generation']:04d}-worker-{worker_id:02d}"
                    f"-part-{part:04d}"
                )
            manifest = write_shard(
                config["output_dir"],
                shard_name,
                chunk_games,
                chunk_turns,
                metadata=shard_metadata(),
            )
            audit = audit_shard(manifest)
            if not audit["valid"]:
                raise RuntimeError(f"generated shard failed audit: {audit['errors']}")
            shard_manifests.append(manifest)
            shard_audits.append(audit)
            turns = [turn for turn in turns if turn.game_id not in game_ids]
            flushed_games += count
            print(json.dumps({
                "event": "selfplay_shard_committed",
                "worker": worker_id,
                "shard": shard_name,
                "shard_games": count,
                "total_committed_games": flushed_games,
                "elapsed_seconds": time.perf_counter() - started,
                "unix_seconds": time.time(),
            }), flush=True)
            available = len(games) - flushed_games
            if (
                config["stop_after_shards"] > 0
                and len(shard_manifests) >= config["stop_after_shards"]
            ):
                break
            if not force and available < target:
                break

    def create_slot(game_index):
        game_seed = config["seed"] + game_index * 1009
        game_id = f"g{config['generation']:04d}-w{worker_id}-{game_index:08d}"
        context = engine.create_context(game_seed)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        moves = []
        player = 1
        turn_count = 0
        if config["forced_opening_radius"] >= 0:
            radius = config["forced_opening_radius"]
            generator = np.random.default_rng(game_seed)
            row = int(generator.integers(9 - radius, 10 + radius))
            column = int(generator.integers(9 - radius, 10 + radius))
            opening = row * BOARD_SIZE + column
            candidates = [
                candidate_row * BOARD_SIZE + candidate_column
                for candidate_row in range(9 - radius, 10 + radius)
                for candidate_column in range(9 - radius, 10 + radius)
            ]
            black, white, state_digest = board_payloads(board)
            turns.append(PairTurnRecord(
                game_id=game_id,
                turn_index=0,
                player=player,
                stones_before=0,
                board_black=black,
                board_white=white,
                state_hash=state_digest,
                chosen_a=opening,
                chosen_b=PASS_MOVE,
                policy_a=candidates,
                policy_b=[PASS_MOVE] * len(candidates),
                policy_probability=[1.0 / len(candidates)] * len(candidates),
                root_value=0.0,
                simulations=0,
                unique_nodes=0,
                neural_evaluations=0,
                search_ms=0.0,
            ))
            placed, winner = apply_action(board, player, opening, PASS_MOVE)
            if winner is not None:
                raise RuntimeError("forced opening unexpectedly terminated the game")
            context.play(opening, PASS_MOVE)
            moves.extend(placed)
            player = -player
            turn_count = 1
        return {
            "game_index": game_index,
            "game_id": game_id,
            "seed": game_seed,
            "context": context,
            "board": board,
            "player": player,
            "moves": moves,
            "turn_count": turn_count,
        }

    slots = []
    for _ in range(min(config["concurrent_games"], len(game_indices))):
        try:
            slots.append(create_slot(next(pending_indices)))
        except StopIteration:
            break

    started = time.perf_counter()
    total_search_seconds = 0.0
    try:
        while slots:
            contexts = [slot["context"] for slot in slots]
            maximum_budgets = [
                config["black_simulations"]
                if slot["player"] == 1
                else config["white_simulations"]
                for slot in slots
            ]
            search_started = time.perf_counter()
            if config["adaptive_simulations"]:
                spent = [0] * len(contexts)
                accumulated = [
                    {"simulations": 0, "unique_nodes": 0, "neural_evaluations": 0}
                    for _ in contexts
                ]

                def run_to(targets):
                    increments = [
                        max(0, int(target) - spent[index])
                        for index, target in enumerate(targets)
                    ]
                    engine.run(contexts, increments)
                    for index, context in enumerate(contexts):
                        stats = context.stats()
                        spent[index] += stats["simulations"]
                        for key in accumulated[index]:
                            accumulated[index][key] += stats[key]

                minimum_targets = [
                    min(config["adaptive_min_simulations"], maximum)
                    for maximum in maximum_budgets
                ]
                run_to(minimum_targets)
                confident = [
                    max(context.policy()[2], default=0.0)
                    >= config["adaptive_confidence_threshold"]
                    for context in contexts
                ]
                middle_targets = [
                    spent[index]
                    if confident[index]
                    else min(config["adaptive_mid_simulations"], maximum)
                    for index, maximum in enumerate(maximum_budgets)
                ]
                run_to(middle_targets)
                confident = [
                    confident[index]
                    or max(context.policy()[2], default=0.0)
                    >= config["adaptive_confidence_threshold"]
                    for index, context in enumerate(contexts)
                ]
                final_targets = [
                    spent[index] if confident[index] else maximum
                    for index, maximum in enumerate(maximum_budgets)
                ]
                run_to(final_targets)
                search_statistics = accumulated
            else:
                engine.run(contexts, maximum_budgets)
                search_statistics = [context.stats() for context in contexts]
            torch.cuda.synchronize(device)
            search_seconds = time.perf_counter() - search_started
            total_search_seconds += search_seconds
            next_slots = []
            for slot_index, slot in enumerate(slots):
                context = slot["context"]
                temperature = (
                    config["opening_temperature"]
                    if slot["turn_count"] == 0
                    else config["early_temperature"]
                    if slot["turn_count"] < config["temperature_turns"]
                    else 0.0
                )
                first, second = context.choose_action(temperature)
                policy_a, policy_b, policy_probability = context.policy()
                search_stats = search_statistics[slot_index]
                black, white, state_digest = board_payloads(slot["board"])
                turns.append(PairTurnRecord(
                    game_id=slot["game_id"],
                    turn_index=slot["turn_count"],
                    player=slot["player"],
                    stones_before=len(slot["moves"]),
                    board_black=black,
                    board_white=white,
                    state_hash=state_digest,
                    chosen_a=first,
                    chosen_b=second,
                    policy_a=policy_a.astype(int).tolist(),
                    policy_b=policy_b.astype(int).tolist(),
                    policy_probability=policy_probability.astype(float).tolist(),
                    root_value=context.root_value(),
                    simulations=search_stats["simulations"],
                    unique_nodes=search_stats["unique_nodes"],
                    neural_evaluations=search_stats["neural_evaluations"],
                    search_ms=search_seconds * 1000.0,
                ))
                placed, winner = apply_action(
                    slot["board"],
                    slot["player"],
                    first,
                    second,
                )
                slot["moves"].extend(placed)
                slot["turn_count"] += 1
                context.play(first, second)
                if context.winner != (winner or 0) or context.terminal != (winner is not None):
                    raise RuntimeError("Python/C++ game-state mismatch")
                if winner is not None:
                    games.append(PairGameRecord(
                        game_id=slot["game_id"],
                        generation=config["generation"],
                        seed=slot["seed"],
                        model_sha256=config["model_sha256"],
                        engine_sha256=config["engine_sha256"],
                        search_config_sha256=search_hash,
                        winner=winner,
                        termination="draw" if winner == 0 else "six",
                        moves=slot["moves"],
                    ))
                    context.close()
                    try:
                        next_slots.append(create_slot(next(pending_indices)))
                    except StopIteration:
                        pass
                else:
                    slot["player"] = -slot["player"]
                    next_slots.append(slot)
            slots = next_slots
            flush_completed()
            if (
                config["stop_after_shards"] > 0
                and len(shard_manifests) >= config["stop_after_shards"]
            ):
                break
    finally:
        for slot in slots:
            slot["context"].close()

    elapsed = time.perf_counter() - started
    stopped_after_shards = (
        config["stop_after_shards"] > 0
        and len(shard_manifests) >= config["stop_after_shards"]
    )
    if not stopped_after_shards:
        flush_completed(force=True)
    total_turns = sum(int(audit["turns"]) for audit in shard_audits)
    audit = (
        shard_audits[0]
        if len(shard_audits) == 1
        else {
            "valid": all(item["valid"] for item in shard_audits),
            "shards": len(shard_audits),
            "games": sum(int(item["games"]) for item in shard_audits),
            "turns": total_turns,
        }
    )
    published_games = games[:flushed_games]
    winner_counts = Counter(game.winner for game in published_games)
    length_buckets = Counter(
        game_length_bucket(len(game.moves)) for game in published_games
    )
    opening_counts = Counter(int(game.moves[0]) for game in published_games)
    evaluator_stats = evaluator.stats()
    mcts_stats = engine.global_stats()
    result_queue.put({
        "worker": worker_id,
        "gpu": gpu_id,
        "games": len(published_games),
        "turns": total_turns,
        "stones": sum(len(game.moves) for game in published_games),
        "winner_counts": {
            "black": winner_counts[1],
            "white": winner_counts[-1],
            "draw": winner_counts[0],
        },
        "length_buckets": {
            name: length_buckets[name]
            for name in ("short", "medium", "long", "very_long")
        },
        "opening": opening_distribution(opening_counts, len(published_games)),
        "_game_fingerprints": [
            game_fingerprint(game.moves) for game in published_games
        ],
        "seconds": elapsed,
        "search_seconds": total_search_seconds,
        "games_per_second": len(published_games) / elapsed,
        "turns_per_second": total_turns / elapsed,
        "manifest": str(shard_manifests[0]),
        "manifests": [str(path) for path in shard_manifests],
        "evaluator": evaluator_stats,
        "mcts": mcts_stats,
        "audit": audit,
    })


def worker_entry(config, worker_id, gpu_id, game_indices, result_queue):
    try:
        worker_main(config, worker_id, gpu_id, game_indices, result_queue)
    except Exception as error:
        result_queue.put({
            "worker": worker_id,
            "gpu": gpu_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument(
        "--shard-games",
        type=int,
        default=0,
        help="commit this many completed games per shard without draining active slots",
    )
    parser.add_argument(
        "--stop-after-shards",
        type=int,
        default=0,
        help="performance-only: stop after this many committed shards and discard active games",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="independent self-play workers sharing each physical GPU (1 to 4)",
    )
    parser.add_argument("--concurrent-games", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument(
        "--black-simulations",
        type=int,
        default=0,
        help="per-turn black budget; zero inherits --simulations",
    )
    parser.add_argument(
        "--white-simulations",
        type=int,
        default=0,
        help="per-turn white budget; zero inherits --simulations",
    )
    parser.add_argument(
        "--allow-nonproduction-simulations",
        action="store_true",
        help="permit budgets outside 400-1200 for isolated tests only",
    )
    parser.add_argument(
        "--adaptive-simulations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stop confident positions early while respecting each color's maximum budget",
    )
    parser.add_argument("--adaptive-min-simulations", type=int, default=400)
    parser.add_argument("--adaptive-mid-simulations", type=int, default=800)
    parser.add_argument("--adaptive-confidence-threshold", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--search-profile",
        default="",
        help="JSON profile whose search values override the matching CLI defaults",
    )
    parser.add_argument("--hierarchical-first-candidates", type=int, default=0)
    parser.add_argument("--hierarchical-seconds-per-first", type=int, default=0)
    parser.add_argument("--progressive-widening-base", type=int, default=0)
    parser.add_argument("--progressive-widening-scale", type=float, default=0.0)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--root-noise-epsilon", type=float, default=0.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.15)
    parser.add_argument("--candidate-gumbel", type=float, default=0.25)
    parser.add_argument("--random-candidates", type=int, default=0)
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=14_400.0,
        help="maximum wall time for all GPU workers in one self-play invocation",
    )
    parser.add_argument(
        "--include-smt-affinity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="research-only: add each physical CPU's SMT sibling to worker affinity",
    )
    parser.add_argument(
        "--cpu-affinity",
        default="",
        help="explicit CPU list for a single-GPU worker, for example 62-85",
    )
    parser.add_argument(
        "--worker-cpu-affinities",
        default="",
        help="semicolon-separated explicit CPU list for every expanded GPU worker",
    )
    parser.add_argument("--temperature-turns", type=int, default=6)
    parser.add_argument("--early-temperature", type=float, default=1.0)
    parser.add_argument(
        "--opening-temperature",
        type=float,
        help="temperature for the searched black singleton; defaults to the historical early-turn behavior",
    )
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--async-pipeline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlap native callback with CPU preparation",
    )
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pad-batch-to", type=int, default=0)
    parser.add_argument("--batch-buckets", default="16,32,64,128")
    parser.add_argument("--value-scale", type=float, default=0.0)
    parser.add_argument("--forced-opening-radius", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.async_pipeline and arguments.compile_model:
        raise ValueError(
            "--async-pipeline requires --no-compile-model; torch.compile is not thread-safe"
        )
    if arguments.search_profile:
        profile_path = Path(arguments.search_profile).expanduser().resolve()
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        for key in (
            "top_k",
            "hierarchical_first_candidates",
            "hierarchical_seconds_per_first",
            "progressive_widening_base",
            "progressive_widening_scale",
            "candidate_gumbel",
            "mcts_cpu_threads_per_gpu",
            "concurrent_games",
            "batch_buckets",
            "cpu_affinity",
            "cpu_affinity_policy",
            "workers_per_gpu",
            "shard_games",
        ):
            if key not in profile:
                continue
            target = {
                "mcts_cpu_threads_per_gpu": "mcts_cpu_threads",
                "cpu_affinity_policy": "cpu_affinity",
            }.get(key, key)
            if target == "cpu_affinity" and arguments.worker_cpu_affinities:
                continue
            value = profile[key]
            if key == "batch_buckets" and isinstance(value, list):
                value = ",".join(str(item) for item in value)
            setattr(arguments, target, value)
    if (
        arguments.games < 1
        or arguments.shard_games < 0
        or arguments.stop_after_shards < 0
        or arguments.concurrent_games < 1
        or not 1 <= arguments.workers_per_gpu <= 4
        or arguments.simulations < 1
        or arguments.black_simulations < 0
        or arguments.white_simulations < 0
        or arguments.top_k < 1
        or arguments.mcts_cpu_threads < 1
        or not math.isfinite(arguments.worker_timeout_seconds)
        or arguments.worker_timeout_seconds <= 0.0
    ):
        raise ValueError(
            "games, concurrent_games, simulations, top_k, mcts_cpu_threads, "
            "and worker_timeout_seconds must be positive"
        )
    if arguments.stop_after_shards > 0 and arguments.shard_games <= 0:
        raise ValueError("--stop-after-shards requires positive --shard-games")
    if arguments.early_temperature < 0.0 or arguments.temperature_turns < 0:
        raise ValueError("early_temperature and temperature_turns must be non-negative")
    hierarchical = bool(
        arguments.hierarchical_first_candidates
        or arguments.hierarchical_seconds_per_first
    )
    if hierarchical and (
        min(
            arguments.hierarchical_first_candidates,
            arguments.hierarchical_seconds_per_first,
        ) <= 0
        or arguments.hierarchical_first_candidates
        * arguments.hierarchical_seconds_per_first
        != arguments.top_k
    ):
        raise ValueError(
            "hierarchical candidates must both be positive and multiply to top_k"
        )
    if (
        arguments.progressive_widening_base < 0
        or not math.isfinite(arguments.progressive_widening_scale)
        or arguments.progressive_widening_scale < 0
        or bool(arguments.progressive_widening_base)
        != bool(arguments.progressive_widening_scale)
    ):
        raise ValueError(
            "progressive widening base and scale must both be zero or positive"
        )
    if arguments.progressive_widening_base > arguments.top_k:
        raise ValueError("progressive widening base cannot exceed top_k")
    if arguments.opening_temperature is None:
        arguments.opening_temperature = (
            arguments.early_temperature if arguments.temperature_turns > 0 else 0.0
        )
    if arguments.opening_temperature < 0.0:
        raise ValueError("opening_temperature must be non-negative")
    arguments.black_simulations = arguments.black_simulations or arguments.simulations
    arguments.white_simulations = arguments.white_simulations or arguments.simulations
    validate_simulation_budgets(
        arguments.simulations,
        arguments.black_simulations,
        arguments.white_simulations,
        allow_nonproduction=arguments.allow_nonproduction_simulations,
    )
    if arguments.adaptive_simulations:
        if not (
            1 <= arguments.adaptive_min_simulations <= arguments.adaptive_mid_simulations
            and arguments.adaptive_min_simulations
            <= min(arguments.black_simulations, arguments.white_simulations)
        ):
            raise ValueError(
                "adaptive simulation stages must satisfy 1 <= min <= mid and min <= caps"
            )
        if not 0.0 < arguments.adaptive_confidence_threshold <= 1.0:
            raise ValueError("adaptive confidence threshold must be in (0, 1]")
    if not -1 <= arguments.forced_opening_radius <= 9:
        raise ValueError("forced_opening_radius must be -1 (disabled) or in [0, 9]")
    arguments.batch_buckets = [
        int(value)
        for value in arguments.batch_buckets.split(",")
        if value.strip()
    ]
    if arguments.pad_batch_to > 0:
        arguments.batch_buckets = []
    elif arguments.batch_buckets:
        arguments.pad_batch_to = 0
    else:
        arguments.pad_batch_to = arguments.concurrent_games

    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(arguments.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        save_random_checkpoint(checkpoint, seed=arguments.seed)
    library = ensure_library()
    physical_gpu_ids = [
        int(item) for item in arguments.gpus.split(",") if item.strip()
    ]
    gpu_ids = expand_gpu_workers(physical_gpu_ids, arguments.workers_per_gpu)
    affinity_spec = str(arguments.cpu_affinity).strip()
    worker_affinity_spec = str(arguments.worker_cpu_affinities).strip()
    explicit_worker_affinities = parse_worker_cpu_affinities(
        worker_affinity_spec, len(gpu_ids)
    )
    if explicit_worker_affinities and affinity_spec:
        raise ValueError(
            "--cpu-affinity and --worker-cpu-affinities are mutually exclusive"
        )
    auto_affinity = should_auto_cpu_affinity(
        affinity_spec, arguments.workers_per_gpu
    )
    explicit_affinity = (
        [] if not affinity_spec or auto_affinity
        else parse_linux_cpu_list(affinity_spec)
    )
    if explicit_affinity and len(gpu_ids) != 1:
        raise ValueError("--cpu-affinity is supported only for one GPU worker")
    if explicit_affinity and hasattr(os, "sched_getaffinity"):
        unavailable = sorted(set(explicit_affinity) - os.sched_getaffinity(0))
        if unavailable:
            raise ValueError(f"CPU affinity contains unavailable CPUs: {unavailable}")
    if explicit_worker_affinities and hasattr(os, "sched_getaffinity"):
        requested_cpus = {
            cpu for group in explicit_worker_affinities for cpu in group
        }
        unavailable = sorted(requested_cpus - os.sched_getaffinity(0))
        if unavailable:
            raise ValueError(f"CPU affinity contains unavailable CPUs: {unavailable}")
    if explicit_worker_affinities:
        worker_cpu_affinities = explicit_worker_affinities
    elif auto_affinity:
        worker_cpu_affinities = topology_worker_affinities(
            gpu_ids,
            include_smt=arguments.include_smt_affinity,
        )
    elif explicit_affinity:
        worker_cpu_affinities = [explicit_affinity]
    else:
        worker_cpu_affinities = [None] * len(gpu_ids)
    model_sha256 = sha256_file(checkpoint)
    config = vars(arguments) | {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "library": str(library),
        "model_sha256": model_sha256,
        "engine_sha256": sha256_file(library),
        "worker_cpu_affinities": worker_cpu_affinities,
    }
    assignments = [list(range(worker, arguments.games, len(gpu_ids))) for worker in range(len(gpu_ids))]
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    started = time.perf_counter()
    for worker_id, (gpu_id, indices) in enumerate(zip(gpu_ids, assignments)):
        if not indices:
            continue
        process = context.Process(
            target=worker_entry,
            args=(config, worker_id, gpu_id, indices, result_queue),
        )
        process.start()
        processes.append(process)
    results = []
    deadline = time.monotonic() + arguments.worker_timeout_seconds
    try:
        while len(results) < len(processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("self-play worker timed out")
            try:
                results.append(result_queue.get(timeout=min(1.0, remaining)))
                continue
            except queue.Empty:
                pass
            failed = [
                process for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                exitcodes = [process.exitcode for process in failed]
                raise RuntimeError(f"self-play workers exited early: {exitcodes}")
            if all(process.exitcode == 0 for process in processes):
                raise RuntimeError("self-play workers exited without publishing all results")
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join()
        raise
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"self-play worker exited with {process.exitcode}")
    failures = [item for item in results if "error" in item]
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))
    elapsed = time.perf_counter() - started
    opening_counts = Counter()
    fingerprints = Counter()
    for item in results:
        opening_counts.update(
            {int(move): int(count) for move, count in item["opening"]["counts"].items()}
        )
        fingerprints.update(item.pop("_game_fingerprints"))
    total_games = sum(item["games"] for item in results)
    summary = {
        "games": total_games,
        "turns": sum(item["turns"] for item in results),
        "stones": sum(item["stones"] for item in results),
        "winner_counts": {
            name: sum(item["winner_counts"][name] for item in results)
            for name in ("black", "white", "draw")
        },
        "length_buckets": {
            name: sum(item["length_buckets"][name] for item in results)
            for name in ("short", "medium", "long", "very_long")
        },
        "opening": opening_distribution(opening_counts, total_games),
        "diversity": {
            "unique_complete_games": len(fingerprints),
            "duplicate_complete_games": total_games - len(fingerprints),
            "unique_complete_game_rate": len(fingerprints) / total_games,
            "maximum_complete_game_repetitions": max(fingerprints.values(), default=0),
        },
        "seconds": elapsed,
        "games_per_second": sum(item["games"] for item in results) / elapsed,
        "turns_per_second": sum(item["turns"] for item in results) / elapsed,
        "workers": sorted(results, key=lambda item: item["worker"]),
    }
    summary_path = output_dir / f"gen-{arguments.generation:04d}-summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
