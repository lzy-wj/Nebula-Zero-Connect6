"""Asynchronous native-Pair actors with a serialized learner/arena gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from experiments.pair_native.pair_replay import (
    audit_shard,
    load_shard,
    load_verified_games,
    sha256_file,
)


PRODUCTION_MIN_SIMULATIONS = 400
PRODUCTION_MAX_SIMULATIONS = 1200


def parse_simulation_schedule(specification: str) -> list[tuple[int, int]]:
    """Parse ``budget:weight`` entries without changing the legacy default."""
    if not specification.strip():
        return []
    result = []
    for item in specification.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError("simulation schedule entries must be budget:weight")
        budget, weight = (int(value) for value in fields)
        if budget < 1 or weight < 1:
            raise ValueError("simulation schedule budgets and weights must be positive")
        result.append((budget, weight))
    if len({budget for budget, _ in result}) != len(result):
        raise ValueError("simulation schedule budgets must be unique")
    return result


def scheduled_simulations(
    round_index: int,
    schedule: list[tuple[int, int]],
    seed: int,
    fallback: int,
) -> int:
    """Choose an exactly weighted, deterministically shuffled cyclic budget."""
    if not schedule:
        return int(fallback)
    cycle = [budget for budget, weight in schedule for _ in range(weight)]
    random.Random(int(seed) ^ 0x53494D53).shuffle(cycle)
    return int(cycle[int(round_index) % len(cycle)])


def select_replay_generations(
    available: list[dict],
    selected_window: int,
    pool_window: int = 0,
    recent_window: int = 0,
) -> list[dict]:
    """Select recent replay plus evenly spaced history from a bounded pool."""
    ordered = sorted(available, key=lambda item: int(item["round_index"]))
    effective_pool = int(pool_window) if pool_window else int(selected_window)
    pool = ordered[-effective_pool:]
    if len(pool) <= selected_window or not pool_window:
        return pool[-selected_window:]
    recent_count = min(int(recent_window), int(selected_window), len(pool))
    history_count = int(selected_window) - recent_count
    recent = pool[-recent_count:] if recent_count else []
    older = pool[:-recent_count] if recent_count else pool
    if history_count <= 0:
        return recent
    if history_count == 1:
        history = [older[len(older) // 2]]
    else:
        indices = [
            round(index * (len(older) - 1) / (history_count - 1))
            for index in range(history_count)
        ]
        history = [older[index] for index in indices]
    return sorted(history + recent, key=lambda item: int(item["round_index"]))


@dataclass
class RunningJob:
    kind: str
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    metadata: dict

    def close_log(self):
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class ReplayQualityError(RuntimeError):
    """An audited actor replay failed a configured quality threshold."""

    def __init__(self, message: str, summary: dict):
        super().__init__(message)
        self.summary = summary


def atomic_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def next_attempt(directory: Path, prefix: str) -> Path:
    for attempt in range(10_000):
        candidate = directory / f"{prefix}-attempt-{attempt:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many attempts for {prefix}")


def launch_job(kind: str, command: list[str], log_path: Path, metadata: dict, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        start_new_session=True,
    )
    metadata = dict(metadata)
    metadata[f"{kind}_started_unix"] = time.time()
    metadata[f"{kind}_pid"] = process.pid
    # Persist the concrete log filename so a rejected/retried actor remains
    # inspectable after the orchestrator has moved on.  In particular, retry
    # logs are intentionally distinct (see launch_actor below).
    metadata[f"{kind}_log_path"] = str(log_path)
    return RunningJob(kind, process, log_handle, log_path, metadata)


def log_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def terminate_job(job: RunningJob | None):
    if job is None or job.process.poll() is not None:
        if job is not None:
            job.close_log()
        return
    try:
        os.killpg(job.process.pid, 15)
        job.process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(job.process.pid, 9)
        except ProcessLookupError:
            pass
        job.process.wait()
    finally:
        job.close_log()


def validate_replay(
    directory: Path,
    generation: int,
    expected_games: int,
    expected_model_sha256: str | None = None,
    expected_black_simulations: int | None = None,
    expected_white_simulations: int | None = None,
    expected_opening_temperature: float | None = None,
    maximum_opening_top_share: float = 1.0,
    minimum_unique_game_rate: float = 0.0,
    audit_workers: int = 1,
) -> dict:
    summary_path = directory / f"gen-{generation:04d}-summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing self-play summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["games"]) != expected_games:
        raise RuntimeError(
            f"self-play game count mismatch: {summary['games']} != {expected_games}"
        )
    manifests = sorted(directory.glob("*.manifest.json"))
    if not manifests:
        raise RuntimeError(f"no committed replay manifests under {directory}")
    if audit_workers < 1:
        raise ValueError("audit_workers must be positive")
    if audit_workers == 1 or len(manifests) == 1:
        audits = [audit_shard(manifest) for manifest in manifests]
    else:
        worker_count = min(audit_workers, len(manifests))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            audits = list(executor.map(audit_shard, manifests))
    invalid = [audit for audit in audits if not audit["valid"]]
    if invalid:
        raise RuntimeError(f"async actor produced invalid replay: {invalid}")
    engine_hashes = set()
    search_hashes = set()
    game_ids = set()
    winner_counts = Counter()
    length_buckets = Counter()
    opening_counts = Counter()
    game_signatures = Counter()
    derived_stones = 0
    for audit in audits:
        provenance = audit["provenance"]
        if provenance["generations"] != [generation]:
            raise RuntimeError("replay generation does not match actor assignment")
        if (
            expected_model_sha256 is not None
            and provenance["model_sha256"] != [expected_model_sha256]
        ):
            raise RuntimeError("replay model hash does not match actor checkpoint")
        engine_hashes.update(provenance["engine_sha256"])
        search_hashes.update(provenance["search_config_sha256"])
        search = audit.get("metadata", {}).get("search", {})
        for color, expected in (
            ("black", expected_black_simulations),
            ("white", expected_white_simulations),
        ):
            observed = int(search.get(f"{color}_simulations", -1))
            if expected is not None and observed != int(expected):
                raise RuntimeError(
                    f"actor {color} simulation budget does not match controller assignment"
                )
        observed_opening_temperature = search.get("opening_temperature")
        if (
            expected_opening_temperature is not None
            and (
                observed_opening_temperature is None
                or not math.isclose(
                    float(observed_opening_temperature),
                    float(expected_opening_temperature),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            )
        ):
            raise RuntimeError(
                "actor opening temperature does not match controller assignment"
            )
        _, games = load_verified_games(audit["manifest"])
        for game in games:
            game_id = game["game_id"]
            if game_id in game_ids:
                raise RuntimeError(f"duplicate game id across actor shards: {game_id}")
            game_ids.add(game_id)
            winner = int(game["winner"])
            winner_counts[{1: "black", -1: "white", 0: "draw"}[winner]] += 1
            moves = tuple(int(move) for move in game["moves"])
            stones = len(moves)
            derived_stones += stones
            if not moves:
                raise RuntimeError("replay contains an empty game")
            opening_counts[moves[0]] += 1
            game_signatures[moves] += 1
            bucket = (
                "short" if stones <= 40
                else "medium" if stones <= 80
                else "long" if stones <= 140
                else "very_long"
            )
            length_buckets[bucket] += 1
    if len(engine_hashes) != 1 or len(search_hashes) != 1:
        raise RuntimeError("actor shards disagree on engine or search provenance")
    if sum(int(audit["games"]) for audit in audits) != expected_games:
        raise RuntimeError("manifest game total does not match self-play summary")
    derived_winners = {
        name: winner_counts[name]
        for name in ("black", "white", "draw")
    }
    derived_lengths = {
        name: length_buckets[name]
        for name in ("short", "medium", "long", "very_long")
    }
    for name, derived in (
        ("winner_counts", derived_winners),
        ("length_buckets", derived_lengths),
    ):
        if name in summary and summary[name] != derived:
            raise RuntimeError(f"self-play {name} does not match audited games")
        summary[name] = derived
    if "stones" in summary and int(summary["stones"]) != derived_stones:
        raise RuntimeError("self-play stone count does not match audited games")
    summary["stones"] = derived_stones
    derived_turns = sum(int(audit["turns"]) for audit in audits)
    if "turns" in summary and int(summary["turns"]) != derived_turns:
        raise RuntimeError("self-play turn count does not match audited shards")
    summary["turns"] = derived_turns
    top_move, top_count = opening_counts.most_common(1)[0]
    probabilities = [count / expected_games for count in opening_counts.values()]
    entropy = -sum(value * math.log2(value) for value in probabilities if value > 0)
    derived_opening = {
        "counts": {
            str(int(move)): int(count)
            for move, count in sorted(opening_counts.items())
        },
        "unique": len(opening_counts),
        "entropy_bits": entropy,
        "effective_openings": 2.0**entropy,
        "top_move": int(top_move),
        "top_count": int(top_count),
        "top_share": top_count / expected_games,
    }
    derived_diversity = {
        "unique_complete_games": len(game_signatures),
        "duplicate_complete_games": expected_games - len(game_signatures),
        "unique_complete_game_rate": len(game_signatures) / expected_games,
        "maximum_complete_game_repetitions": max(game_signatures.values()),
    }
    if "opening" in summary and summary["opening"].get("counts") != derived_opening["counts"]:
        raise RuntimeError("self-play opening counts do not match audited games")
    if (
        "diversity" in summary
        and int(summary["diversity"].get("duplicate_complete_games", -1))
        != derived_diversity["duplicate_complete_games"]
    ):
        raise RuntimeError("self-play duplicate count does not match audited games")
    summary["opening"] = derived_opening
    summary["diversity"] = derived_diversity
    if derived_opening["top_share"] > maximum_opening_top_share:
        raise ReplayQualityError(
            "opening collapse: top share "
            f"{derived_opening['top_share']:.4f} exceeds {maximum_opening_top_share:.4f}",
            summary,
        )
    if derived_diversity["unique_complete_game_rate"] < minimum_unique_game_rate:
        raise ReplayQualityError(
            "game diversity collapse: unique rate "
            f"{derived_diversity['unique_complete_game_rate']:.4f} below "
            f"{minimum_unique_game_rate:.4f}",
            summary,
        )
    return summary


def promotion_decision(
    arena: dict,
    z_score: float,
    margin: float,
    policy: str = "confidence",
) -> dict:
    score = float(arena["trained_score"])
    standard_error = float(arena["trained_score_standard_error"])
    lower_bound = score - z_score * standard_error
    confidence_pass = lower_bound > 0.5 + margin
    advantage_pass = score > 0.5 + margin
    if policy == "confidence":
        promote = confidence_pass
    elif policy == "advantage":
        promote = advantage_pass
    else:
        raise ValueError(f"unknown promotion policy: {policy}")
    return {
        "promote": promote,
        "promotion_reason": policy if promote else "rejected",
        "promotion_policy": policy,
        "confidence_pass": confidence_pass,
        "advantage_pass": advantage_pass,
        "score": score,
        "standard_error": standard_error,
        "lower_bound": lower_bound,
        "required_lower_bound": 0.5 + margin,
    }


def stalled_promotion_candidate(
    promotion_history: list[dict],
    current: dict,
    maximum_stall: int,
    minimum_score: float,
) -> dict | None:
    """Return the strongest recent rejected candidate after a bounded stall.

    Only candidates evaluated against the current accepted checkpoint belong
    to the window.  A prior promotion therefore resets the streak naturally.
    """
    if maximum_stall <= 0:
        return None
    streak = []
    for item in reversed(promotion_history):
        if (
            item.get("promote")
            or item.get("baseline_sha256") != current.get("baseline_sha256")
            or item.get("gate_config") != current.get("gate_config")
        ):
            break
        streak.append(item)
    streak.reverse()
    streak.append(current)
    if len(streak) < maximum_stall:
        return None
    window = streak[-maximum_stall:]
    best = max(
        window,
        key=lambda item: (
            float(item["score"]),
            float(item.get("lower_bound", float("-inf"))),
            int(item["through_round"]),
        ),
    )
    return best if float(best["score"]) >= minimum_score else None


def gpu_cpu_affinity(gpu_id: int) -> str:
    if 0 <= gpu_id < 4:
        start = gpu_id * 24
    elif 4 <= gpu_id < 8:
        start = 96 + (gpu_id - 4) * 24
    else:
        raise ValueError("GPU IDs must be in [0, 7]")
    return f"{start}-{start + 23}"


def validate_gpu_layout(
    actor_gpus: list[int],
    learner_gpu: int,
    gate_gpu: int,
    single_gpu_serial: bool = False,
    phase_serial: bool = False,
) -> None:
    """Validate an async GPU layout, including a shared service GPU.

    Learner and gate jobs are serialized by the state machine: learner
    completion launches its gate, and a new learner cannot launch while either
    job exists.  They may therefore safely share one GPU.  Actor GPUs must stay
    disjoint because self-play intentionally overlaps both service phases.
    """
    all_gpus = list(actor_gpus) + [int(learner_gpu), int(gate_gpu)]
    if len(actor_gpus) < 1 or len(actor_gpus) > 7:
        raise ValueError("async loop requires one to seven actor GPUs")
    if len(set(actor_gpus)) != len(actor_gpus):
        raise ValueError("actor GPU IDs must be unique")
    if any(not 0 <= gpu < 8 for gpu in all_gpus):
        raise ValueError("GPU IDs must be in [0, 7]")
    if single_gpu_serial and phase_serial:
        raise ValueError("choose only one serial scheduling mode")
    if single_gpu_serial:
        if not (
            len(actor_gpus) == 1
            and actor_gpus[0] == learner_gpu == gate_gpu
        ):
            raise ValueError(
                "single-GPU serial mode requires one shared actor/learner/gate GPU"
            )
    elif phase_serial:
        if learner_gpu != gate_gpu or learner_gpu not in actor_gpus:
            raise ValueError(
                "phase-serial mode requires learner/gate to share an actor GPU"
            )
    elif learner_gpu in actor_gpus or gate_gpu in actor_gpus:
        raise ValueError("learner and gate GPUs cannot also be actor GPUs")


def next_training_round(completed_rounds: set[int], trained_through: int) -> int | None:
    target = trained_through + 1
    return target if target in completed_rounds else None


def pause_drain_complete(
    pause_requested: bool,
    actor_active: bool,
    learner_active: bool,
    gate_active: bool,
    training_round: int | None,
) -> bool:
    """Return true only at a provenance-safe profile-switch boundary."""
    return (
        pause_requested
        and not actor_active
        and not learner_active
        and not gate_active
        and training_round is None
    )


def dynamic_color_budgets(
    base_simulations: int,
    summaries: list[dict],
    lower_rate: float = 0.45,
    upper_rate: float = 0.55,
    gain: float = 4.0,
    maximum_multiplier: float = 3.0,
    maximum_simulations: int | None = 1200,
) -> dict:
    """Increase only the under-performing color's search budget.

    Draws are excluded from the feedback rate. Missing legacy summaries safely
    fall back to equal budgets, so older replay directories remain usable.
    """
    if base_simulations < 1:
        raise ValueError("base_simulations must be positive")
    if (
        not math.isfinite(gain)
        or gain < 0.0
        or not math.isfinite(maximum_multiplier)
        or maximum_multiplier < 1.0
    ):
        raise ValueError("invalid color-budget gain or multiplier")
    black = sum(int(item.get("winner_counts", {}).get("black", 0)) for item in summaries)
    white = sum(int(item.get("winner_counts", {}).get("white", 0)) for item in summaries)
    draws = sum(int(item.get("winner_counts", {}).get("draw", 0)) for item in summaries)
    decisive = black + white
    black_rate = black / decisive if decisive else None
    black_budget = int(base_simulations)
    white_budget = int(base_simulations)
    boosted = "none"
    multiplier = 1.0
    if black_rate is not None and not lower_rate <= black_rate <= upper_rate:
        exponent = min(
            math.log(float(maximum_multiplier)),
            float(gain) * abs(black_rate - 0.5),
        )
        multiplier = math.exp(exponent)
        boosted = "white" if black_rate > upper_rate else "black"
        adjusted = max(base_simulations, int(round(base_simulations * multiplier)))
        if maximum_simulations is not None:
            adjusted = min(int(maximum_simulations), adjusted)
        multiplier = adjusted / base_simulations
        if boosted == "white":
            white_budget = adjusted
        else:
            black_budget = adjusted
    return {
        "black_simulations": black_budget,
        "white_simulations": white_budget,
        "boosted_color": boosted,
        "budget_multiplier": multiplier,
        "observed_black_wins": black,
        "observed_white_wins": white,
        "observed_draws": draws,
        "observed_black_decisive_rate": black_rate,
        "target_black_decisive_rate_band": [lower_rate, upper_rate],
    }


def dynamic_opening_temperature(
    base_temperature: float,
    summaries: list[dict],
    lower_rate: float = 0.45,
    upper_rate: float = 0.55,
    color_gain: float = 32.0,
    minimum_temperature: float = 0.1,
    maximum_temperature: float = 1.5,
    maximum_top_share: float = 0.40,
    diversity_gain: float = 2.0,
) -> dict:
    """Use searched-opening temperature as the first color/diversity actuator.

    The controller preserves model-selected openings: it only changes how the
    MCTS visit distribution is sampled.  Color feedback moves temperature up
    when black is too strong and down when black is too weak.  A separate
    opening-concentration term can raise it even while aggregate color remains
    inside the deadband.
    """
    values = (
        base_temperature,
        color_gain,
        minimum_temperature,
        maximum_temperature,
        maximum_top_share,
        diversity_gain,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("opening temperature controller values must be finite")
    if not 0.0 <= minimum_temperature <= base_temperature <= maximum_temperature:
        raise ValueError("opening temperatures must satisfy 0 <= minimum <= base <= maximum")
    if color_gain < 0.0 or diversity_gain < 0.0:
        raise ValueError("opening temperature gains must be non-negative")
    if not 0.0 < maximum_top_share <= 1.0:
        raise ValueError("maximum opening top share must be in (0, 1]")

    black = sum(int(item.get("winner_counts", {}).get("black", 0)) for item in summaries)
    white = sum(int(item.get("winner_counts", {}).get("white", 0)) for item in summaries)
    decisive = black + white
    black_rate = black / decisive if decisive else None
    opening_counts = Counter()
    round_top_shares = []
    for summary in summaries:
        counts = {
            int(move): int(count)
            for move, count in summary.get("opening", {}).get("counts", {}).items()
            if int(count) > 0
        }
        opening_counts.update(counts)
        round_games = sum(counts.values())
        if round_games:
            round_top_shares.append(max(counts.values()) / round_games)
    opening_games = sum(opening_counts.values())
    top_move, top_count = opening_counts.most_common(1)[0] if opening_counts else (None, 0)
    aggregate_top_share = top_count / opening_games if opening_games else None
    # A merged window can hide a per-generation collapse when each generation
    # collapses onto a different move.  Keep the aggregate metric for reporting,
    # but let the strictest recent generation drive the diversity actuator.
    maximum_round_top_share = max(round_top_shares, default=None)
    top_share = (
        max(aggregate_top_share, maximum_round_top_share)
        if aggregate_top_share is not None and maximum_round_top_share is not None
        else aggregate_top_share if aggregate_top_share is not None
        else maximum_round_top_share
    )

    temperature = float(base_temperature)
    reasons = []
    if black_rate is not None and black_rate > upper_rate:
        temperature = max(
            temperature,
            base_temperature + color_gain * (black_rate - upper_rate),
        )
        reasons.append("black_overperforming")
    elif black_rate is not None and black_rate < lower_rate:
        temperature = min(
            temperature,
            base_temperature - color_gain * (lower_rate - black_rate),
        )
        reasons.append("black_underperforming")
    if top_share is not None and top_share > maximum_top_share:
        temperature = max(
            temperature,
            base_temperature + diversity_gain * (top_share - maximum_top_share),
        )
        reasons.append("opening_concentrated")
    temperature = min(maximum_temperature, max(minimum_temperature, temperature))
    return {
        "opening_temperature": temperature,
        "base_opening_temperature": float(base_temperature),
        "observed_black_decisive_rate": black_rate,
        "observed_opening_top_move": top_move,
        "observed_opening_top_share": top_share,
        "observed_aggregate_opening_top_share": aggregate_top_share,
        "observed_max_round_opening_top_share": maximum_round_top_share,
        "opening_games": opening_games,
        "reasons": reasons or ["base"],
        "target_black_decisive_rate_band": [lower_rate, upper_rate],
        "maximum_opening_top_share": maximum_top_share,
        "temperature_range": [minimum_temperature, maximum_temperature],
    }


def quality_retry_opening_control(
    control: dict,
    retry_index: int,
    minimum_temperature: float,
    maximum_temperature: float,
    diversity_gain: float,
) -> dict:
    """Raise opening temperature for an audited-but-rejected actor retry.

    The rejected summary is already fed through ``dynamic_opening_temperature``;
    this deterministic floor guarantees that a unique-game failure with a flat
    opening histogram still gets a hotter retry.  The cap makes retries safe at
    the configured production ceiling.
    """
    if retry_index < 0:
        raise ValueError("retry_index must be non-negative")
    if not all(
        math.isfinite(float(value))
        for value in (minimum_temperature, maximum_temperature, diversity_gain)
    ):
        raise ValueError("quality retry temperature values must be finite")
    if not 0.0 <= minimum_temperature <= maximum_temperature:
        raise ValueError("quality retry temperature bounds are invalid")
    result = dict(control)
    current = float(result["opening_temperature"])
    if retry_index:
        bump = max(0.1, float(diversity_gain) * 0.25)
        base = float(result.get("base_opening_temperature", current))
        retry_floor = base + retry_index * bump
        result["opening_temperature"] = min(
            maximum_temperature,
            max(minimum_temperature, current, retry_floor),
        )
        result["reasons"] = list(result.get("reasons", [])) + ["quality_retry"]
    result["quality_retry_index"] = int(retry_index)
    return result


def quality_retry_sampling_control(
    base_gumbel: float,
    retry_index: int,
    gumbel_increment: float,
    maximum_gumbel: float,
    base_seed: int,
    round_index: int,
    seed_stride: int,
) -> dict:
    """Return reproducible but distinct sampling settings for each retry."""
    if retry_index < 0 or round_index < 0:
        raise ValueError("retry and round indices must be non-negative")
    if seed_stride < 1:
        raise ValueError("quality retry seed stride must be positive")
    if not all(
        math.isfinite(float(value))
        for value in (base_gumbel, gumbel_increment, maximum_gumbel)
    ):
        raise ValueError("quality retry gumbel values must be finite")
    if gumbel_increment < 0.0 or maximum_gumbel < base_gumbel:
        raise ValueError("quality retry gumbel bounds are invalid")
    return {
        "candidate_gumbel": min(
            float(maximum_gumbel),
            float(base_gumbel) + retry_index * float(gumbel_increment),
        ),
        "seed": int(base_seed + round_index * 1_000_003 + retry_index * seed_stride),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--start-generation", type=int, default=2)
    parser.add_argument("--games-per-round", type=int, default=1536)
    parser.add_argument("--actor-gpus", default="0,1,2,3,4,5")
    parser.add_argument("--learner-gpu", type=int, default=6)
    parser.add_argument("--gate-gpu", type=int, default=7)
    parser.add_argument(
        "--single-gpu-serial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="serialize actor, learner, and gate on one shared GPU",
    )
    parser.add_argument(
        "--phase-serial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="serialize multi-GPU actor and shared learner/gate phases",
    )
    parser.add_argument(
        "--pause-request-file",
        default="",
        help="drain in-flight work and pause cleanly when this file exists",
    )
    parser.add_argument("--concurrent-games", type=int, default=128)
    parser.add_argument("--batch-buckets", default="16,32,64,128")
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--minimum-selfplay-simulations", type=int, default=400)
    parser.add_argument("--maximum-selfplay-simulations", type=int, default=1200)
    parser.add_argument(
        "--simulation-schedule",
        default="",
        help="deterministic weighted per-generation budgets, e.g. 400:75,600:15,800:8,1200:2",
    )
    parser.add_argument(
        "--adaptive-simulations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="treat scheduled/color budgets as caps and stop confident positions early",
    )
    parser.add_argument("--adaptive-min-simulations", type=int, default=400)
    parser.add_argument("--adaptive-mid-simulations", type=int, default=800)
    parser.add_argument("--adaptive-confidence-threshold", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--candidate-gumbel", type=float, default=0.25)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--forced-opening-radius", type=int, default=-1)
    parser.add_argument("--temperature-turns", type=int, default=6)
    parser.add_argument("--early-temperature", type=float, default=0.35)
    parser.add_argument(
        "--opening-temperature",
        type=float,
        default=1.25,
        help="base temperature for the searched black singleton",
    )
    parser.add_argument(
        "--dynamic-opening-temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--opening-temperature-minimum", type=float, default=1.0)
    parser.add_argument("--opening-temperature-maximum", type=float, default=1.5)
    parser.add_argument("--opening-temperature-color-gain", type=float, default=32.0)
    parser.add_argument("--opening-top-share-target", type=float, default=0.40)
    parser.add_argument("--opening-temperature-diversity-gain", type=float, default=2.0)
    parser.add_argument(
        "--maximum-opening-top-share",
        type=float,
        default=0.50,
        help="hard replay rejection threshold for the most concentrated opening",
    )
    parser.add_argument(
        "--minimum-unique-game-rate",
        type=float,
        default=0.90,
        help="hard replay rejection threshold for complete-game fingerprints",
    )
    parser.add_argument(
        "--max-quality-retries",
        type=int,
        default=2,
        help="re-run a quality-rejected actor round this many times before failing",
    )
    parser.add_argument("--quality-retry-gumbel-increment", type=float, default=0.0)
    parser.add_argument("--quality-retry-gumbel-maximum", type=float, default=1.0)
    parser.add_argument("--quality-retry-seed-stride", type=int, default=10_000_019)
    parser.add_argument(
        "--dynamic-color-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--color-balance-window", type=int, default=3)
    parser.add_argument("--color-balance-lower", type=float, default=0.45)
    parser.add_argument("--color-balance-upper", type=float, default=0.55)
    parser.add_argument("--color-balance-gain", type=float, default=4.0)
    parser.add_argument("--color-balance-max-multiplier", type=float, default=3.0)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--decisive-value-weight", type=float, default=10.0)
    parser.add_argument(
        "--train-sampling-mode",
        choices=("phase", "joint"),
        default="joint",
    )
    parser.add_argument("--train-length-targets", default="0.20,0.35,0.30,0.15")
    parser.add_argument("--train-winner-targets", default="0.475,0.475,0.05")
    parser.add_argument("--train-sampler-max-weight-ratio", type=float, default=4.0)
    parser.add_argument("--train-sampler-raking-iterations", type=int, default=30)
    parser.add_argument("--replay-window", type=int, default=3)
    parser.add_argument(
        "--replay-pool-window",
        type=int,
        default=0,
        help="bounded generation pool; 0 preserves a purely recent replay window",
    )
    parser.add_argument(
        "--replay-recent-window",
        type=int,
        default=0,
        help="recent generations kept when sampling history from replay-pool-window",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="maximum weighted turn draws per learner epoch; 0 uses all loaded turns",
    )
    parser.add_argument("--gate-pairs", type=int, default=64)
    parser.add_argument("--gate-simulations", type=int, default=400)
    parser.add_argument(
        "--max-stale-rounds",
        type=int,
        default=1,
        help="allow this many completed actor rounds beyond the next training target",
    )
    parser.add_argument("--gate-opening-radius", type=int, default=4)
    parser.add_argument(
        "--gate-opening-mode",
        choices=("forced", "free"),
        default="forced",
    )
    parser.add_argument(
        "--gate-opening-temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--gate-seed-mode",
        choices=("per_round", "champion"),
        default="per_round",
        help="keep a common random panel during each champion's tenure",
    )
    parser.add_argument("--promotion-z", type=float, default=1.96)
    parser.add_argument("--promotion-margin", type=float, default=0.0)
    parser.add_argument(
        "--promotion-policy",
        choices=("confidence", "advantage"),
        default="confidence",
    )
    parser.add_argument(
        "--max-promotion-stall",
        type=int,
        default=0,
        help="after this many rejections, promote the best non-regressing candidate; 0 disables",
    )
    parser.add_argument(
        "--fallback-promotion-floor",
        type=float,
        default=0.48,
        help="minimum raw arena score eligible for bounded-stall promotion",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume-adopt-checkpoint",
        default="",
        help="one-shot paused-resume adoption of a previously evaluated candidate",
    )
    parser.add_argument(
        "--adoption-evidence",
        default="",
        help="optional JSON/directory provenance for resume-adopt-checkpoint",
    )
    parser.add_argument(
        "--resume-audit-from-round",
        type=int,
        default=0,
        help="first completed round to re-audit on resume; earlier rounds must have a recorded prior audit",
    )
    parser.add_argument(
        "--replay-audit-workers",
        type=int,
        default=1,
        help="strict replay-audit processes; one preserves historical scheduling",
    )
    parser.add_argument(
        "--train-single-pass-audit-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="make the learner reuse rows decoded by its strict replay audit",
    )
    parser.add_argument(
        "--train-replay-audit-workers",
        type=int,
        default=1,
        help="strict replay-audit processes used by each learner",
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    actor_gpus = [int(value) for value in arguments.actor_gpus.split(",") if value.strip()]
    validate_gpu_layout(
        actor_gpus,
        arguments.learner_gpu,
        arguments.gate_gpu,
        arguments.single_gpu_serial,
        arguments.phase_serial,
    )
    if arguments.rounds < 1 or arguments.games_per_round < len(actor_gpus):
        raise ValueError("rounds and games_per_round are too small")
    if arguments.replay_window < 1:
        raise ValueError("replay_window must be positive")
    if arguments.replay_pool_window < 0 or arguments.replay_recent_window < 0:
        raise ValueError("replay pool and recent windows must be non-negative")
    if arguments.replay_pool_window:
        if not (
            1 <= arguments.replay_recent_window
            <= arguments.replay_window
            <= arguments.replay_pool_window
        ):
            raise ValueError(
                "replay windows must satisfy 1 <= recent <= selected <= pool"
            )
    elif arguments.replay_recent_window:
        raise ValueError("replay_recent_window requires replay_pool_window")
    if arguments.train_max_samples < 0:
        raise ValueError("train_max_samples must be non-negative")
    if arguments.replay_audit_workers < 1:
        raise ValueError("replay_audit_workers must be positive")
    if arguments.train_replay_audit_workers < 1:
        raise ValueError("train_replay_audit_workers must be positive")
    if arguments.max_stale_rounds < 0:
        raise ValueError("max_stale_rounds must be nonnegative")
    if arguments.color_balance_window < 1:
        raise ValueError("color_balance_window must be positive")
    if not 0.0 <= arguments.color_balance_lower < 0.5:
        raise ValueError("color_balance_lower must be in [0, 0.5)")
    if not 0.5 < arguments.color_balance_upper <= 1.0:
        raise ValueError("color_balance_upper must be in (0.5, 1]")
    if (
        not math.isfinite(arguments.color_balance_gain)
        or arguments.color_balance_gain < 0.0
        or not math.isfinite(arguments.color_balance_max_multiplier)
        or arguments.color_balance_max_multiplier < 1.0
    ):
        raise ValueError("invalid color-balance gain or maximum multiplier")
    if not (
        PRODUCTION_MIN_SIMULATIONS <= arguments.minimum_selfplay_simulations
        <= arguments.simulations
        <= arguments.maximum_selfplay_simulations
        <= PRODUCTION_MAX_SIMULATIONS
    ):
        raise ValueError(
            "production self-play simulations must satisfy "
            "400 <= minimum <= base <= maximum <= 1200"
        )
    simulation_schedule = parse_simulation_schedule(arguments.simulation_schedule)
    if any(
        not arguments.minimum_selfplay_simulations
        <= budget
        <= arguments.maximum_selfplay_simulations
        for budget, _ in simulation_schedule
    ):
        raise ValueError(
            "simulation schedule budgets must lie within the configured self-play range"
        )
    if not (
        arguments.minimum_selfplay_simulations
        <= arguments.adaptive_min_simulations
        <= arguments.adaptive_mid_simulations
        <= arguments.maximum_selfplay_simulations
    ):
        raise ValueError("adaptive stages must lie within the self-play simulation range")
    if not 0.0 < arguments.adaptive_confidence_threshold <= 1.0:
        raise ValueError("adaptive confidence threshold must be in (0, 1]")
    if arguments.early_temperature < 0.0 or arguments.temperature_turns < 0:
        raise ValueError("early_temperature and temperature_turns must be non-negative")
    dynamic_opening_temperature(
        arguments.opening_temperature,
        [],
        lower_rate=arguments.color_balance_lower,
        upper_rate=arguments.color_balance_upper,
        color_gain=arguments.opening_temperature_color_gain,
        minimum_temperature=arguments.opening_temperature_minimum,
        maximum_temperature=arguments.opening_temperature_maximum,
        maximum_top_share=arguments.opening_top_share_target,
        diversity_gain=arguments.opening_temperature_diversity_gain,
    )
    if not 0.0 < arguments.maximum_opening_top_share <= 1.0:
        raise ValueError("maximum_opening_top_share must be in (0, 1]")
    if not 0.0 <= arguments.minimum_unique_game_rate <= 1.0:
        raise ValueError("minimum_unique_game_rate must be in [0, 1]")
    if arguments.max_quality_retries < 0:
        raise ValueError("max_quality_retries must be non-negative")
    if arguments.quality_retry_gumbel_increment < 0.0:
        raise ValueError("quality_retry_gumbel_increment must be non-negative")
    if arguments.quality_retry_gumbel_maximum < arguments.candidate_gumbel:
        raise ValueError("quality_retry_gumbel_maximum must cover candidate_gumbel")
    if arguments.quality_retry_seed_stride < 1:
        raise ValueError("quality_retry_seed_stride must be positive")
    if not -1 <= arguments.forced_opening_radius <= 9:
        raise ValueError("forced_opening_radius must be -1 (disabled) or in [0, 9]")
    if arguments.train_epochs < 1 or arguments.train_batch_size < 1:
        raise ValueError("training epochs and batch size must be positive")
    if arguments.train_sampler_max_weight_ratio < 1.0:
        raise ValueError("train_sampler_max_weight_ratio must be at least 1")
    if arguments.train_sampler_raking_iterations < 1:
        raise ValueError("train_sampler_raking_iterations must be positive")
    if arguments.simulations < 1 or arguments.top_k < 1 or arguments.gate_simulations < 1:
        raise ValueError("simulations, gate_simulations, and top_k must be positive")
    if arguments.gate_pairs < 2:
        raise ValueError("gate_pairs must be at least two for a paired confidence bound")
    if not 0 <= arguments.gate_opening_radius <= 9:
        raise ValueError("gate_opening_radius must be in [0, 9]")
    if (
        arguments.gate_opening_mode == "forced"
        and arguments.gate_pairs > (2 * arguments.gate_opening_radius + 1) ** 2
    ):
        raise ValueError("gate_pairs exceeds the unique opening capacity")
    if arguments.gate_opening_temperature < 0.0:
        raise ValueError("gate_opening_temperature must be non-negative")
    if arguments.max_promotion_stall < 0:
        raise ValueError("max_promotion_stall must be non-negative")
    if not 0.0 <= arguments.fallback_promotion_floor <= 1.0:
        raise ValueError("fallback_promotion_floor must be in [0, 1]")

    root = Path(arguments.output_dir).expanduser().resolve()
    pause_request_path = (
        Path(arguments.pause_request_file).expanduser().resolve()
        if arguments.pause_request_file
        else None
    )
    serial_service_phases = arguments.single_gpu_serial or arguments.phase_serial
    replay_root = root / "replay"
    candidate_root = root / "candidates"
    gate_root = root / "gates"
    log_root = root / "logs"
    for directory in (replay_root, candidate_root, gate_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / ".orchestrator.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"another async loop owns {root}") from error
    state_path = root / "state.json"
    initial_checkpoint = Path(arguments.initial_checkpoint).expanduser().resolve()
    if not initial_checkpoint.exists():
        raise FileNotFoundError(initial_checkpoint)

    if arguments.resume and not state_path.exists():
        raise FileNotFoundError(f"cannot resume without {state_path}")
    if not arguments.resume and state_path.exists():
        raise FileExistsError(
            f"{state_path} already exists; use --resume or a new output directory"
        )

    if arguments.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_config = state.get("config", {})
        # Promotion confidence only affects gates launched after a resume.  It
        # is safe to tune without rewriting any committed promotion decision;
        # record the transition below so historical provenance stays explicit.
        changeable = {
            "actor_gpus",
            "fallback_promotion_floor",
            "gate_gpu",
            "gate_opening_mode",
            "gate_opening_temperature",
            "gate_seed_mode",
            "learner_gpu",
            "max_quality_retries",
            "max_stale_rounds",
            "max_promotion_stall",
            "pause_request_file",
            "phase_serial",
            "poll_seconds",
            "promotion_policy",
            "promotion_z",
            "resume_adopt_checkpoint",
            "adoption_evidence",
            "resume_audit_from_round",
            "resume",
            "rounds",
            "single_gpu_serial",
            "quality_retry_gumbel_increment",
            "quality_retry_gumbel_maximum",
            "quality_retry_seed_stride",
        }
        backward_compatible_defaults = {
            "replay_audit_workers": 1,
            "train_single_pass_audit_load": False,
            "train_replay_audit_workers": 1,
            "simulation_schedule": "",
            "adaptive_simulations": False,
            "adaptive_min_simulations": 400,
            "adaptive_mid_simulations": 800,
            "adaptive_confidence_threshold": 0.50,
            "replay_pool_window": 0,
            "replay_recent_window": 0,
            "train_max_samples": 0,
            "gate_opening_mode": "forced",
            "gate_opening_temperature": 0.0,
            "gate_seed_mode": "per_round",
            "promotion_policy": "confidence",
            "max_promotion_stall": 0,
            "fallback_promotion_floor": 0.48,
            "resume_adopt_checkpoint": "",
            "adoption_evidence": "",
            "resume_audit_from_round": 0,
            "single_gpu_serial": False,
            "phase_serial": False,
            "pause_request_file": "",
            "quality_retry_gumbel_increment": 0.0,
            "quality_retry_gumbel_maximum": 1.0,
            "quality_retry_seed_stride": 10_000_019,
        }
        incompatible = {
            key: (
                previous_config.get(key, backward_compatible_defaults.get(key)),
                value,
            )
            for key, value in vars(arguments).items()
            if key not in changeable and previous_config.get(key) != value
            if previous_config.get(key, backward_compatible_defaults.get(key)) != value
        }
        if incompatible:
            raise ValueError(f"resume configuration mismatch: {incompatible}")
        config_changes = {
            key: {"from": previous_config.get(key), "to": value}
            for key, value in vars(arguments).items()
            if key in changeable
            and key != "resume"
            and previous_config.get(key) != value
        }
        if config_changes:
            state.setdefault("config_changes", []).append({
                "changed_unix": time.time(),
                "changes": config_changes,
            })
        accepted_checkpoint = Path(state["accepted_checkpoint"])
        if not accepted_checkpoint.exists():
            raise FileNotFoundError(f"accepted checkpoint is missing: {accepted_checkpoint}")
        accepted_hash = sha256_file(accepted_checkpoint)
        if accepted_hash != state.get("accepted_sha256"):
            raise RuntimeError("accepted checkpoint hash does not match state.json")
        completed_rounds = list(state.get("completed_rounds", []))
        rejected_attempts = list(state.get("rejected_attempts", []))
        trained_through = int(state.get("trained_through", -1))
        promotions = list(state.get("promotions", []))
        completed_ids = sorted(int(item["round_index"]) for item in completed_rounds)
        if completed_ids != list(range(len(completed_ids))):
            raise RuntimeError(f"completed actor rounds are not contiguous: {completed_ids}")
        if not 0 <= arguments.resume_audit_from_round <= len(completed_ids):
            raise ValueError("resume_audit_from_round is outside completed history")
        if arguments.resume_audit_from_round:
            prior_audits = state.get("resume_audits", [])
            prior_coverage = max(
                [int(item.get("through_round", -1)) + 1 for item in prior_audits]
                + [
                    len(completed_ids)
                    if state.get("checkpoint_adoptions")
                    else 0
                ]
            )
            if prior_coverage < arguments.resume_audit_from_round:
                raise RuntimeError(
                    "cannot skip replay audit without recorded prior coverage"
                )
        previous_rounds = int(previous_config.get("rounds", arguments.rounds))
        if arguments.rounds < previous_rounds:
            raise ValueError(
                f"--rounds cannot shrink a resume plan ({arguments.rounds} < {previous_rounds})"
            )
        if arguments.rounds < len(completed_ids) or arguments.rounds <= trained_through:
            raise ValueError("--rounds cannot truncate completed resume state")
        if not -1 <= trained_through <= max(completed_ids, default=-1):
            raise RuntimeError("trained_through is outside the completed actor range")
        promotion_ids = [int(item["through_round"]) for item in promotions]
        if promotion_ids != list(range(trained_through + 1)):
            raise RuntimeError("promotion history is not contiguous with trained_through")
        expected_checkpoint = initial_checkpoint
        expected_hash = sha256_file(expected_checkpoint)
        for promotion in promotions:
            if promotion["baseline_sha256"] != expected_hash:
                raise RuntimeError("promotion baseline hash breaks the accepted chain")
            candidate = Path(promotion["candidate_checkpoint"])
            if not candidate.exists() or sha256_file(candidate) != promotion["candidate_sha256"]:
                raise RuntimeError("promotion candidate checkpoint is missing or corrupt")
            if promotion["promote"]:
                expected_checkpoint = Path(
                    promotion.get("promoted_candidate_checkpoint", candidate)
                )
                expected_hash = promotion.get(
                    "promoted_candidate_sha256",
                    promotion["candidate_sha256"],
                )
                if (
                    not expected_checkpoint.exists()
                    or sha256_file(expected_checkpoint) != expected_hash
                ):
                    raise RuntimeError("promoted checkpoint is missing or corrupt")
        if accepted_hash != expected_hash or accepted_checkpoint != expected_checkpoint:
            raise RuntimeError("accepted checkpoint does not match promotion history")
        if arguments.resume_adopt_checkpoint:
            if state.get("status") != "paused":
                raise RuntimeError("resume checkpoint adoption requires a paused state")
            adopted = Path(arguments.resume_adopt_checkpoint).expanduser().resolve()
            if not adopted.exists():
                raise FileNotFoundError(f"adopted checkpoint is missing: {adopted}")
            adopted_hash = sha256_file(adopted)
            source = next(
                (
                    item
                    for item in promotions
                    if Path(item["candidate_checkpoint"]).resolve() == adopted
                    and item["candidate_sha256"] == adopted_hash
                ),
                None,
            )
            if source is None:
                raise RuntimeError(
                    "resume adoption must select an already evaluated candidate"
                )
            if not promotions:
                raise RuntimeError("resume adoption requires promotion history")
            event = promotions[-1]
            if event.get("promote"):
                raise RuntimeError("cannot overwrite an existing final promotion event")
            evidence = (
                str(Path(arguments.adoption_evidence).expanduser().resolve())
                if arguments.adoption_evidence
                else ""
            )
            event.update({
                "promote": True,
                "promotion_reason": "free_opening_tournament",
                "selected_through_round": int(source["through_round"]),
                "promoted_candidate_checkpoint": str(adopted),
                "promoted_candidate_sha256": adopted_hash,
                "adoption_evidence": evidence,
                "original_gate_decision": {
                    "promote": False,
                    "score": event.get("score"),
                    "lower_bound": event.get("lower_bound"),
                },
            })
            accepted_checkpoint = adopted
            accepted_hash = adopted_hash
            state.setdefault("checkpoint_adoptions", []).append({
                "adopted_unix": time.time(),
                "event_through_round": int(event["through_round"]),
                "selected_through_round": int(source["through_round"]),
                "checkpoint": str(adopted),
                "sha256": adopted_hash,
                "evidence": evidence,
            })
        for item in completed_rounds:
            round_index = int(item["round_index"])
            expected_generation = arguments.start_generation + round_index
            if int(item["generation"]) != expected_generation:
                raise RuntimeError("completed generation does not match resume configuration")
            actor_checkpoint = Path(item["checkpoint"])
            if (
                not actor_checkpoint.exists()
                or sha256_file(actor_checkpoint) != item["checkpoint_sha256"]
            ):
                raise RuntimeError("actor checkpoint provenance is missing or corrupt")
            if round_index < arguments.resume_audit_from_round:
                continue
            validate_replay(
                Path(item["directory"]),
                expected_generation,
                arguments.games_per_round,
                item["checkpoint_sha256"],
                item.get("color_control", {}).get("black_simulations"),
                item.get("color_control", {}).get("white_simulations"),
                item.get("opening_control", {}).get("opening_temperature"),
                arguments.maximum_opening_top_share,
                arguments.minimum_unique_game_rate,
                arguments.replay_audit_workers,
            )
        state.setdefault("resume_audits", []).append({
            "audited_unix": time.time(),
            "from_round": int(arguments.resume_audit_from_round),
            "through_round": len(completed_ids) - 1,
        })
    else:
        accepted_checkpoint = initial_checkpoint
        completed_rounds = []
        rejected_attempts = []
        trained_through = -1
        promotions = []
        state = {}

    def save_state(status: str):
        state.update({
            "schema_version": 1,
            "status": status,
            "config": vars(arguments),
            "accepted_checkpoint": str(accepted_checkpoint),
            "accepted_sha256": sha256_file(accepted_checkpoint),
            "completed_rounds": completed_rounds,
            "rejected_attempts": rejected_attempts,
            "trained_through": trained_through,
            "promotions": promotions,
            "updated_unix": time.time(),
        })
        atomic_json(state, state_path)

    completed_indices = {int(item["round_index"]) for item in completed_rounds}
    next_actor_round = 0
    while next_actor_round in completed_indices:
        next_actor_round += 1
    actor_job = None
    learner_job = None
    gate_job = None

    def launch_actor(round_index: int):
        generation = arguments.start_generation + round_index
        round_simulations = scheduled_simulations(
            round_index,
            simulation_schedule,
            arguments.seed,
            arguments.simulations,
        )
        directory = next_attempt(replay_root, f"gen-{generation:04d}")
        retry_attempts = [
            item
            for item in rejected_attempts
            if int(item.get("round_index", -1)) == round_index
        ]
        retry_index = len(retry_attempts)
        recent_summaries = [
            item.get("summary", {})
            for item in sorted(
                completed_rounds,
                key=lambda item: int(item["round_index"]),
            )[-arguments.color_balance_window :]
        ]
        opening_feedback = recent_summaries + [
            item.get("summary", {}) for item in retry_attempts
        ]
        color_control = dynamic_color_budgets(
            round_simulations,
            recent_summaries if arguments.dynamic_color_balance else [],
            lower_rate=arguments.color_balance_lower,
            upper_rate=arguments.color_balance_upper,
            gain=arguments.color_balance_gain,
            maximum_multiplier=arguments.color_balance_max_multiplier,
            maximum_simulations=arguments.maximum_selfplay_simulations,
        )
        opening_control = dynamic_opening_temperature(
            arguments.opening_temperature,
            opening_feedback if arguments.dynamic_opening_temperature else [],
            lower_rate=arguments.color_balance_lower,
            upper_rate=arguments.color_balance_upper,
            color_gain=arguments.opening_temperature_color_gain,
            minimum_temperature=arguments.opening_temperature_minimum,
            maximum_temperature=arguments.opening_temperature_maximum,
            maximum_top_share=arguments.opening_top_share_target,
            diversity_gain=arguments.opening_temperature_diversity_gain,
        )
        opening_control = quality_retry_opening_control(
            opening_control,
            retry_index,
            arguments.opening_temperature_minimum,
            arguments.opening_temperature_maximum,
            arguments.opening_temperature_diversity_gain,
        )
        retry_sampling = quality_retry_sampling_control(
            arguments.candidate_gumbel,
            retry_index,
            arguments.quality_retry_gumbel_increment,
            arguments.quality_retry_gumbel_maximum,
            arguments.seed,
            round_index,
            arguments.quality_retry_seed_stride,
        )
        retry_gumbel = retry_sampling["candidate_gumbel"]
        actor_seed = retry_sampling["seed"]
        command = [
            sys.executable,
            "-m",
            "experiments.pair_native.selfplay",
            "--output-dir",
            str(directory),
            "--checkpoint",
            str(accepted_checkpoint),
            "--games",
            str(arguments.games_per_round),
            "--gpus",
            ",".join(str(value) for value in actor_gpus),
            "--concurrent-games",
            str(arguments.concurrent_games),
            "--batch-buckets",
            arguments.batch_buckets,
            "--mcts-cpu-threads",
            str(arguments.mcts_cpu_threads),
            "--simulations",
            str(round_simulations),
            "--black-simulations",
            str(color_control["black_simulations"]),
            "--white-simulations",
            str(color_control["white_simulations"]),
            "--adaptive-min-simulations",
            str(arguments.adaptive_min_simulations),
            "--adaptive-mid-simulations",
            str(arguments.adaptive_mid_simulations),
            "--adaptive-confidence-threshold",
            str(arguments.adaptive_confidence_threshold),
            "--top-k",
            str(arguments.top_k),
            "--candidate-gumbel",
            str(retry_gumbel),
            "--random-candidates",
            "0",
            "--value-scale",
            str(arguments.value_scale),
            "--forced-opening-radius",
            str(arguments.forced_opening_radius),
            "--temperature-turns",
            str(arguments.temperature_turns),
            "--early-temperature",
            str(arguments.early_temperature),
            "--opening-temperature",
            str(opening_control["opening_temperature"]),
            "--generation",
            str(generation),
            "--seed",
            str(actor_seed),
        ]
        command.append(
            "--adaptive-simulations"
            if arguments.adaptive_simulations
            else "--no-adaptive-simulations"
        )
        metadata = {
            "round_index": round_index,
            "generation": generation,
            "directory": str(directory),
            "checkpoint": str(accepted_checkpoint),
            "checkpoint_sha256": sha256_file(accepted_checkpoint),
            "simulation_control": {
                "base_simulations": round_simulations,
                "schedule": arguments.simulation_schedule,
            },
            "color_control": color_control,
            "opening_control": opening_control,
            "quality_retry_index": retry_index,
            "quality_retry_control": {
                "seed": actor_seed,
                "candidate_gumbel": retry_gumbel,
            },
        }
        # Keep every quality attempt's log.  Reusing one round-level filename
        # used to truncate the failed attempt as soon as retry 1 started,
        # making a post-mortem (and resume audit) needlessly ambiguous.
        actor_log_name = (
            f"actor-round-{round_index:04d}-retry-{retry_index:02d}.log"
        )
        return launch_job(
            "actor",
            command,
            log_root / actor_log_name,
            metadata,
        )

    def launch_learner(through_round: int):
        available = sorted(
            (
                item for item in completed_rounds
                if int(item["round_index"]) <= through_round
            ),
            key=lambda item: int(item["round_index"]),
        )
        selected = select_replay_generations(
            available,
            arguments.replay_window,
            arguments.replay_pool_window,
            arguments.replay_recent_window,
        )
        directory = next_attempt(candidate_root, f"through-{through_round:04d}")
        command = [
            sys.executable,
            "-m",
            "experiments.pair_native.train",
        ]
        for replay in selected:
            command.extend(("--data-dir", replay["directory"]))
        command.extend([
            "--init-checkpoint",
            str(accepted_checkpoint),
            "--output-dir",
            str(directory),
            "--epochs",
            str(arguments.train_epochs),
            "--batch-size",
            str(arguments.train_batch_size),
            "--workers",
            str(arguments.train_workers),
            "--learning-rate",
            str(arguments.learning_rate),
            "--value-weight",
            str(arguments.value_weight),
            "--decisive-value-weight",
            str(arguments.decisive_value_weight),
            "--value-target",
            "search",
            "--sampling-mode",
            arguments.train_sampling_mode,
            "--length-targets",
            arguments.train_length_targets,
            "--winner-targets",
            arguments.train_winner_targets,
            "--sampler-max-weight-ratio",
            str(arguments.train_sampler_max_weight_ratio),
            "--sampler-raking-iterations",
            str(arguments.train_sampler_raking_iterations),
            "--seed",
            str(arguments.seed + through_round * 7_919),
            "--cpu-affinity",
            gpu_cpu_affinity(arguments.learner_gpu),
        ])
        command.append(
            "--single-pass-audit-load"
            if arguments.train_single_pass_audit_load
            else "--no-single-pass-audit-load"
        )
        command.extend((
            "--replay-audit-workers",
            str(arguments.train_replay_audit_workers),
        ))
        if arguments.train_max_samples:
            command.extend(("--max-train-samples", str(arguments.train_max_samples)))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(arguments.learner_gpu)
        metadata = {
            "through_round": through_round,
            "directory": str(directory),
            "baseline_checkpoint": str(accepted_checkpoint),
            "baseline_sha256": sha256_file(accepted_checkpoint),
            "replay_directories": [item["directory"] for item in selected],
            "replay_rounds": [int(item["round_index"]) for item in selected],
            "replay_pool_window": int(arguments.replay_pool_window),
            "replay_recent_window": int(arguments.replay_recent_window),
        }
        return launch_job(
            "learner",
            command,
            log_root / f"learner-through-{through_round:04d}.log",
            metadata,
            env=environment,
        )

    def launch_gate(learner_metadata: dict):
        candidate = Path(learner_metadata["directory"]) / "best.pth"
        if not candidate.exists():
            raise RuntimeError(f"learner did not publish {candidate}")
        through_round = int(learner_metadata["through_round"])
        output = next_attempt(gate_root, f"through-{through_round:04d}")
        output.mkdir(parents=True)
        result_path = output / "arena.json"
        champion_index = sum(bool(item.get("promote")) for item in promotions)
        gate_seed = arguments.seed + (
            champion_index * 104_729
            if arguments.gate_seed_mode == "champion"
            else through_round * 104_729
        )
        command = [
            sys.executable,
            "-m",
            "experiments.pair_native.arena",
            "--initial-checkpoint",
            learner_metadata["baseline_checkpoint"],
            "--trained-checkpoint",
            str(candidate),
            "--pairs",
            str(arguments.gate_pairs),
            "--simulations",
            str(arguments.gate_simulations),
            "--top-k",
            str(arguments.top_k),
            "--opening-radius",
            str(arguments.gate_opening_radius),
            "--opening-temperature",
            str(arguments.gate_opening_temperature),
            "--value-scale",
            str(arguments.value_scale),
            "--mcts-cpu-threads",
            str(arguments.mcts_cpu_threads),
            "--batch-buckets",
            arguments.batch_buckets,
            "--cpu-affinity",
            gpu_cpu_affinity(arguments.gate_gpu),
            "--seed",
            str(gate_seed),
            "--output",
            str(result_path),
        ]
        command.append(
            "--free-opening"
            if arguments.gate_opening_mode == "free"
            else "--no-free-opening"
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(arguments.gate_gpu)
        metadata = dict(learner_metadata) | {
            "candidate_checkpoint": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "result_path": str(result_path),
            "gate_config": {
                "opening_mode": arguments.gate_opening_mode,
                "opening_temperature": arguments.gate_opening_temperature,
                "pairs": arguments.gate_pairs,
                "simulations": arguments.gate_simulations,
                "top_k": arguments.top_k,
                "seed_mode": arguments.gate_seed_mode,
                "seed": gate_seed,
            },
        }
        return launch_job(
            "gate",
            command,
            log_root / f"gate-through-{through_round:04d}.log",
            metadata,
            env=environment,
        )

    save_state("running")
    try:
        while True:
            if actor_job is not None and actor_job.process.poll() is not None:
                return_code = actor_job.process.returncode
                actor_job.close_log()
                if return_code:
                    raise RuntimeError(
                        f"actor failed ({return_code})\n{log_tail(actor_job.log_path)}"
                    )
                metadata = actor_job.metadata
                metadata["actor_finished_unix"] = time.time()
                try:
                    summary = validate_replay(
                        Path(metadata["directory"]),
                        int(metadata["generation"]),
                        arguments.games_per_round,
                        metadata["checkpoint_sha256"],
                        metadata["color_control"]["black_simulations"],
                        metadata["color_control"]["white_simulations"],
                        metadata["opening_control"]["opening_temperature"],
                        arguments.maximum_opening_top_share,
                        arguments.minimum_unique_game_rate,
                        arguments.replay_audit_workers,
                    )
                except ReplayQualityError as error:
                    retry_index = int(metadata.get("quality_retry_index", 0))
                    rejected = dict(metadata)
                    rejected["summary"] = error.summary
                    rejected["rejection_reason"] = str(error)
                    rejected_attempts.append(rejected)
                    rejected_attempts.sort(
                        key=lambda item: (
                            int(item.get("round_index", -1)),
                            int(item.get("quality_retry_index", 0)),
                        )
                    )
                    actor_round = int(metadata["round_index"])
                    actor_job = None
                    if retry_index >= arguments.max_quality_retries:
                        save_state("failed")
                        raise RuntimeError(
                            f"actor quality retry limit exceeded for round {actor_round}: {error}"
                        ) from error
                    next_actor_round = actor_round
                    save_state("running")
                    continue
                metadata["summary"] = summary
                completed_rounds.append(metadata)
                completed_rounds.sort(key=lambda item: int(item["round_index"]))
                actor_job = None
                save_state("running")

            if learner_job is not None and learner_job.process.poll() is not None:
                return_code = learner_job.process.returncode
                learner_job.close_log()
                if return_code:
                    raise RuntimeError(
                        f"learner failed ({return_code})\n{log_tail(learner_job.log_path)}"
                    )
                metadata = learner_job.metadata
                metadata["learner_finished_unix"] = time.time()
                learner_job = None
                gate_job = launch_gate(metadata)
                save_state("running")

            if gate_job is not None and gate_job.process.poll() is not None:
                return_code = gate_job.process.returncode
                gate_job.close_log()
                if return_code:
                    raise RuntimeError(
                        f"gate failed ({return_code})\n{log_tail(gate_job.log_path)}"
                    )
                metadata = gate_job.metadata
                metadata["gate_finished_unix"] = time.time()
                arena = json.loads(Path(metadata["result_path"]).read_text(encoding="utf-8"))
                decision = promotion_decision(
                    arena,
                    arguments.promotion_z,
                    arguments.promotion_margin,
                    arguments.promotion_policy,
                )
                entry = metadata | decision
                if not decision["promote"]:
                    fallback = stalled_promotion_candidate(
                        promotions,
                        entry,
                        arguments.max_promotion_stall,
                        arguments.fallback_promotion_floor,
                    )
                    if fallback is not None:
                        entry.update({
                            "promote": True,
                            "promotion_reason": "stall_best",
                            "selected_through_round": int(fallback["through_round"]),
                            "promoted_candidate_checkpoint": fallback["candidate_checkpoint"],
                            "promoted_candidate_sha256": fallback["candidate_sha256"],
                            "stall_window": int(arguments.max_promotion_stall),
                        })
                if entry["promote"]:
                    accepted_checkpoint = Path(
                        entry.get(
                            "promoted_candidate_checkpoint",
                            entry["candidate_checkpoint"],
                        )
                    )
                trained_through = int(metadata["through_round"])
                promotions.append(entry)
                gate_job = None
                save_state("running")

            completed_indices = {int(item["round_index"]) for item in completed_rounds}
            while next_actor_round in completed_indices:
                next_actor_round += 1
            training_round = next_training_round(completed_indices, trained_through)
            finished = (
                len(completed_indices) >= arguments.rounds
                and trained_through >= arguments.rounds - 1
                and actor_job is None
                and learner_job is None
                and gate_job is None
            )
            if finished:
                save_state("complete")
                print(json.dumps(state, indent=2))
                return 0
            pause_requested = (
                pause_request_path is not None and pause_request_path.exists()
            )
            if pause_drain_complete(
                pause_requested,
                actor_job is not None,
                learner_job is not None,
                gate_job is not None,
                training_round,
            ):
                save_state("paused")
                return 0
            if (
                actor_job is None
                and not pause_requested
                and (
                    not serial_service_phases
                    or (
                        learner_job is None
                        and gate_job is None
                        and training_round is None
                    )
                )
                and next_actor_round < arguments.rounds
                and next_actor_round <= trained_through + 1 + arguments.max_stale_rounds
            ):
                actor_job = launch_actor(next_actor_round)
                next_actor_round += 1
                save_state("running")

            if (
                learner_job is None
                and gate_job is None
                and (not serial_service_phases or actor_job is None)
                and training_round is not None
            ):
                learner_job = launch_learner(training_round)
                save_state("running")
            time.sleep(arguments.poll_seconds)
    except KeyboardInterrupt:
        terminate_job(actor_job)
        terminate_job(learner_job)
        terminate_job(gate_job)
        save_state("paused")
        return 130
    except BaseException:
        terminate_job(actor_job)
        terminate_job(learner_job)
        terminate_job(gate_job)
        save_state("failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
