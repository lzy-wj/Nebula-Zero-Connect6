"""Paired color-swapped arena for two native-Pair checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_mcts import PASS_MOVE, PairMCTSEngine, TorchPairEvaluator
from experiments.pair_native.train import parse_cpu_affinity


def paired_standard_error(pair_scores: list[float]) -> float:
    """Standard error for color-swapped pair means, the gate's independent units."""
    if len(pair_scores) < 2:
        return 0.5
    return float(np.std(pair_scores, ddof=1) / math.sqrt(len(pair_scores)))


def sample_unique_openings(
    generator: np.random.Generator,
    pairs: int,
    radius: int,
) -> list[int]:
    if pairs < 2:
        raise ValueError("pairs must be at least two")
    if not 0 <= radius <= 9:
        raise ValueError("opening radius must be in [0, 9]")
    candidates = [
        row * 19 + column
        for row in range(9 - radius, 10 + radius)
        for column in range(9 - radius, 10 + radius)
    ]
    if pairs > len(candidates):
        raise ValueError(
            f"{pairs} pairs exceed the {len(candidates)} unique openings "
            f"available at radius {radius}"
        )
    return generator.choice(candidates, size=pairs, replace=False).astype(int).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--opening-radius", type=int, default=4)
    parser.add_argument(
        "--free-opening",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let the black model search and choose the singleton opening",
    )
    parser.add_argument(
        "--opening-temperature",
        type=float,
        default=0.0,
        help="temperature used only for a model-selected singleton opening",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--value-scale", type=float, default=0.0)
    parser.add_argument("--initial-value-scale", type=float)
    parser.add_argument("--trained-value-scale", type=float)
    parser.add_argument("--output", default="")
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument("--batch-buckets", default="16,32,64,128")
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pad-batch-to",
        type=int,
        default=-1,
        help="explicit evaluator padding; -1 preserves the historical bucket/fixed rule",
    )
    parser.add_argument("--cpu-affinity", default="")
    arguments = parser.parse_args()
    if arguments.simulations < 1 or arguments.top_k < 1:
        raise ValueError("simulations and top_k must be positive")
    if arguments.opening_temperature < 0.0:
        raise ValueError("opening-temperature must be non-negative")
    affinity = parse_cpu_affinity(arguments.cpu_affinity)
    if affinity:
        os.sched_setaffinity(0, affinity)
    batch_buckets = [
        int(value)
        for value in arguments.batch_buckets.split(",")
        if value.strip()
    ]
    if arguments.pad_batch_to < -1:
        raise ValueError("pad-batch-to must be -1 or non-negative")
    pad_batch_to = arguments.pad_batch_to
    if pad_batch_to == -1:
        pad_batch_to = 0 if batch_buckets else arguments.pairs
    initial_value_scale = (
        arguments.value_scale
        if arguments.initial_value_scale is None
        else arguments.initial_value_scale
    )
    trained_value_scale = (
        arguments.value_scale
        if arguments.trained_value_scale is None
        else arguments.trained_value_scale
    )

    device = torch.device("cuda:0")
    initial_model, _ = load_checkpoint(arguments.initial_checkpoint)
    trained_model, _ = load_checkpoint(arguments.trained_checkpoint)
    evaluators = {
        "initial": TorchPairEvaluator(
            initial_model,
            device,
            top_k=arguments.top_k,
            candidate_gumbel=0.0,
            compile_model=arguments.compile_model,
            deduplicate=False,
            pad_batch_to=pad_batch_to,
            batch_buckets=batch_buckets,
            value_scale=initial_value_scale,
            seed=arguments.seed,
        ),
        "trained": TorchPairEvaluator(
            trained_model,
            device,
            top_k=arguments.top_k,
            candidate_gumbel=0.0,
            compile_model=arguments.compile_model,
            deduplicate=False,
            pad_batch_to=pad_batch_to,
            batch_buckets=batch_buckets,
            value_scale=trained_value_scale,
            seed=arguments.seed + 1,
        ),
    }
    engine = PairMCTSEngine(
        evaluators["initial"],
        top_k=arguments.top_k,
        root_noise_epsilon=0.0,
        cpu_threads=arguments.mcts_cpu_threads,
    )
    rng = np.random.default_rng(arguments.seed)
    openings = (
        [None] * arguments.pairs
        if arguments.free_opening
        else sample_unique_openings(rng, arguments.pairs, arguments.opening_radius)
    )
    games = []
    for pair_index, opening in enumerate(openings):
        for trained_black in (True, False):
            # A color-swapped pair shares its random stream.  In free-opening
            # mode each black model still chooses its own first move, which is
            # the real deployment behavior rather than an injected opening.
            context = engine.create_context(arguments.seed + pair_index)
            if opening is not None:
                context.play(opening, PASS_MOVE)
                context.reset_tree()
            games.append({
                "pair": pair_index,
                "trained_black": trained_black,
                "context": context,
                "turns": 0,
                "winner": None,
                "actions": [],
            })

    active = list(range(len(games)))
    started = time.perf_counter()
    try:
        while active:
            grouped = {"initial": [], "trained": []}
            for game_index in active:
                game = games[game_index]
                player = game["context"].current_player
                trained_to_move = (
                    (player == 1 and game["trained_black"])
                    or (player == -1 and not game["trained_black"])
                )
                grouped["trained" if trained_to_move else "initial"].append(game_index)
            for model_name in ("initial", "trained"):
                indices = grouped[model_name]
                if not indices:
                    continue
                engine.set_evaluator(evaluators[model_name])
                contexts = [games[index]["context"] for index in indices]
                engine.run(contexts, arguments.simulations)
                for index in indices:
                    game = games[index]
                    context = game["context"]
                    opening_temperature = (
                        arguments.opening_temperature
                        if arguments.free_opening and game["turns"] == 0
                        else 0.0
                    )
                    action = context.choose_action(temperature=opening_temperature)
                    context.play(*action)
                    game["actions"].append(action)
                    game["turns"] += 1
                    if context.terminal:
                        game["winner"] = context.winner
                    else:
                        context.reset_tree()
            active = [index for index in active if games[index]["winner"] is None]
            if active and max(games[index]["turns"] for index in active) > 180:
                raise RuntimeError("arena exceeded the legal maximum number of turns")
        torch.cuda.synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        for game in games:
            game["context"].close()

    trained_wins = 0
    initial_wins = 0
    draws = 0
    black_wins = 0
    white_wins = 0
    pair_scores = []
    for pair_index in range(arguments.pairs):
        score = 0.0
        for game in (item for item in games if item["pair"] == pair_index):
            winner = game["winner"]
            if winner == 0:
                draws += 1
                score += 0.5
            else:
                black_wins += int(winner == 1)
                white_wins += int(winner == -1)
                trained_color = 1 if game["trained_black"] else -1
                if winner == trained_color:
                    trained_wins += 1
                    score += 1.0
                else:
                    initial_wins += 1
        pair_scores.append(score / 2.0)
    total = len(games)
    trained_score = (trained_wins + 0.5 * draws) / total
    binomial_standard_error = math.sqrt(
        max(trained_score * (1.0 - trained_score), 0.0) / total
    )
    standard_error = paired_standard_error(pair_scores)
    game_fingerprints = []
    game_details = []
    for game in games:
        payload = json.dumps(
            {
                "pair": game["pair"],
                "trained_black": game["trained_black"],
                "actions": game["actions"],
                "winner": game["winner"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        game_fingerprints.append(fingerprint)
        game_details.append({
            "pair": game["pair"],
            "trained_black": game["trained_black"],
            "winner": game["winner"],
            "turns": game["turns"],
            "opening": (
                game["actions"][0][0]
                if arguments.free_opening and game["actions"]
                else openings[game["pair"]]
            ),
            "fingerprint": fingerprint,
        })
    result = {
        "pairs": arguments.pairs,
        "games": total,
        "simulations": arguments.simulations,
        "top_k": arguments.top_k,
        "opening_radius": arguments.opening_radius,
        "opening_mode": "free" if arguments.free_opening else "forced",
        "opening_temperature": arguments.opening_temperature,
        "openings": openings,
        "initial_value_scale": initial_value_scale,
        "trained_value_scale": trained_value_scale,
        "mcts_cpu_threads": arguments.mcts_cpu_threads,
        "batch_buckets": batch_buckets,
        "compile_model": arguments.compile_model,
        "pad_batch_to": pad_batch_to,
        "cpu_affinity": affinity,
        "trained_wins": trained_wins,
        "initial_wins": initial_wins,
        "draws": draws,
        "trained_score": trained_score,
        "trained_score_standard_error": standard_error,
        "trained_score_binomial_standard_error": binomial_standard_error,
        "paired_score_mean": float(np.mean(pair_scores)),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "mean_turns": float(np.mean([game["turns"] for game in games])),
        "seconds": elapsed,
        "games_per_second": total / elapsed,
        "turns_per_second": sum(game["turns"] for game in games) / elapsed,
        "game_fingerprints": game_fingerprints,
        "unique_game_fingerprints": len(set(game_fingerprints)),
        "game_details": game_details,
        "evaluators": {
            name: evaluator.stats() for name, evaluator in evaluators.items()
        },
    }
    payload = json.dumps(result, indent=2)
    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
