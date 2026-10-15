"""Color-swapped arena between native-Pair and legacy sequential checkpoints.

The native model searches one unordered two-stone action per Connect6 turn.
The legacy C6TransNet searches each stone separately, matching its production
training and inference semantics.  Consequently ``--legacy-simulations`` is a
per-stone budget while ``--native-simulations`` is a per-turn budget; reporting
keeps that asymmetry explicit instead of pretending the compute is identical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.arena import paired_standard_error, sample_unique_openings
from experiments.pair_native.legacy_provenance import (
    default_manifest_path,
    validate_manifest,
)
from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_mcts import PASS_MOVE, PairMCTSEngine, TorchPairEvaluator
from experiments.pair_native.pair_replay import sha256_file
from experiments.pair_native.train import parse_cpu_affinity
from reinforcement_learning.core.connect6_game import Connect6Game
from reinforcement_learning.core.mcts import MCTSEngine


def normalized_winner(game: Connect6Game) -> int:
    return 0 if game.winner == 2 else int(game.winner)


def verify_synchronized(game: Connect6Game, native_context) -> None:
    if native_context.total_stones != len(game.moves):
        raise RuntimeError("legacy and native contexts disagree on stone count")
    if native_context.current_player != game.current_player and not game.winner:
        raise RuntimeError("legacy and native contexts disagree on player to move")
    if bool(native_context.terminal) != bool(game.winner):
        raise RuntimeError("legacy and native contexts disagree on terminal state")
    if game.winner and native_context.winner != normalized_winner(game):
        raise RuntimeError("legacy and native contexts disagree on winner")


def verify_terminal_result(game: Connect6Game, native_context) -> None:
    """Verify a terminal result when canonical Pair order may omit the last stone.

    Legacy search plays its two stones in order.  Pair actions are canonicalized
    by point index, so a winning legacy second stone can be applied first by the
    Pair context and terminate before it records the other stone.  The Python
    game remains the authoritative ordered record; only terminal/winner must
    agree because this context is closed without another search.
    """
    if not game.winner or not native_context.terminal:
        raise RuntimeError("legacy and native contexts disagree on terminal state")
    if native_context.winner != normalized_winner(game):
        raise RuntimeError("legacy and native contexts disagree on winner")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--legacy-engine", required=True)
    parser.add_argument("--legacy-engine-manifest", default="")
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--native-simulations", type=int, default=400)
    parser.add_argument("--legacy-simulations", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--hierarchical-first-candidates", type=int, default=0)
    parser.add_argument("--hierarchical-seconds-per-first", type=int, default=0)
    parser.add_argument("--progressive-widening-base", type=int, default=0)
    parser.add_argument("--progressive-widening-scale", type=float, default=0.0)
    parser.add_argument(
        "--opening-mode",
        choices=("forced", "natural"),
        default="forced",
        help="force a shared random first stone or let the black model search it",
    )
    parser.add_argument("--opening-radius", type=int, default=4)
    parser.add_argument("--early-temperature", type=float, default=0.0)
    parser.add_argument("--temperature-turns", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--mcts-cpu-threads", type=int, default=24)
    parser.add_argument("--legacy-batch-size", type=int, default=64)
    parser.add_argument("--batch-buckets", default="16,32,64,128")
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()

    if arguments.native_simulations < 1 or arguments.legacy_simulations < 1:
        raise ValueError("simulation budgets must be positive")
    if arguments.top_k < 1 or arguments.mcts_cpu_threads < 1:
        raise ValueError("top_k and mcts_cpu_threads must be positive")
    if bool(arguments.hierarchical_first_candidates or arguments.hierarchical_seconds_per_first) and (
        min(arguments.hierarchical_first_candidates, arguments.hierarchical_seconds_per_first) <= 0
        or arguments.hierarchical_first_candidates * arguments.hierarchical_seconds_per_first != arguments.top_k
    ):
        raise ValueError("hierarchical candidates must multiply to top_k")
    if (
        arguments.progressive_widening_base < 0
        or arguments.progressive_widening_base > arguments.top_k
        or not math.isfinite(arguments.progressive_widening_scale)
        or arguments.progressive_widening_scale < 0
        or bool(arguments.progressive_widening_base) != bool(arguments.progressive_widening_scale)
    ):
        raise ValueError("invalid progressive widening parameters")
    if arguments.early_temperature < 0.0 or arguments.temperature_turns < 0:
        raise ValueError("early temperature and temperature turns must be non-negative")
    affinity = parse_cpu_affinity(arguments.cpu_affinity)
    if affinity:
        os.sched_setaffinity(0, affinity)
    batch_buckets = [
        int(value)
        for value in arguments.batch_buckets.split(",")
        if value.strip()
    ]

    native_checkpoint = Path(arguments.native_checkpoint).expanduser().resolve()
    legacy_checkpoint = Path(arguments.legacy_checkpoint).expanduser().resolve()
    legacy_engine_path = Path(arguments.legacy_engine).expanduser().resolve()
    for path in (native_checkpoint, legacy_checkpoint, legacy_engine_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    legacy_manifest_path = (
        Path(arguments.legacy_engine_manifest).expanduser().resolve()
        if arguments.legacy_engine_manifest
        else default_manifest_path(legacy_engine_path)
    )
    legacy_manifest = validate_manifest(
        legacy_manifest_path,
        legacy_checkpoint,
        legacy_engine_path,
    )

    device = torch.device("cuda:0")
    native_model, _ = load_checkpoint(native_checkpoint)
    native_evaluator = TorchPairEvaluator(
        native_model,
        device,
        top_k=arguments.top_k,
        candidate_gumbel=0.0,
        compile_model=True,
        deduplicate=False,
        batch_buckets=batch_buckets,
        value_scale=arguments.value_scale,
        seed=arguments.seed,
        hierarchical_first_candidates=arguments.hierarchical_first_candidates,
        hierarchical_seconds_per_first=arguments.hierarchical_seconds_per_first,
    )
    native_engine = PairMCTSEngine(
        native_evaluator,
        top_k=arguments.top_k,
        root_noise_epsilon=0.0,
        cpu_threads=arguments.mcts_cpu_threads,
        progressive_widening_base=arguments.progressive_widening_base,
        progressive_widening_scale=arguments.progressive_widening_scale,
    )

    legacy_engine = MCTSEngine(str(legacy_engine_path), device=device)
    legacy_engine.set_params(
        batch_size=arguments.legacy_batch_size,
        num_threads=arguments.mcts_cpu_threads,
    )
    legacy_engine.set_tree_batch_size(0)
    legacy_engine.set_unique_leaf_batching(False)
    legacy_engine.set_search_params(cpuct=1.5, widening_base=20, widening_scale=0.25)
    legacy_engine.set_deterministic_selection(True)
    legacy_engine.set_eval_cache_capacity(32768)
    legacy_engine.reset_search_stats()

    rng = np.random.default_rng(arguments.seed)
    openings = (
        sample_unique_openings(rng, arguments.pairs, arguments.opening_radius)
        if arguments.opening_mode == "forced"
        else [None] * arguments.pairs
    )
    games = []
    for pair_index, opening in enumerate(openings):
        for native_black in (True, False):
            seed = arguments.seed + pair_index * 2 + int(native_black)
            native_context = native_engine.create_context(seed)
            legacy_context = legacy_engine.create_game_context(seed)
            game = Connect6Game()
            if opening is not None:
                game.play(opening)
                native_context.play(opening, PASS_MOVE)
                native_context.reset_tree()
                legacy_context.update_state(opening)
            verify_synchronized(game, native_context)
            games.append({
                "pair": pair_index,
                "native_black": native_black,
                "native_context": native_context,
                "legacy_context": legacy_context,
                "game": game,
                "turns": 0,
            })

    active = list(range(len(games)))
    started = time.perf_counter()
    try:
        while active:
            native_indices = []
            legacy_indices = []
            for index in active:
                item = games[index]
                native_to_move = (
                    (item["game"].current_player == 1 and item["native_black"])
                    or (item["game"].current_player == -1 and not item["native_black"])
                )
                (native_indices if native_to_move else legacy_indices).append(index)

            if native_indices:
                contexts = [games[index]["native_context"] for index in native_indices]
                native_engine.run(contexts, arguments.native_simulations)
                for index in native_indices:
                    item = games[index]
                    game = item["game"]
                    opening_turn = not game.moves
                    temperature = (
                        arguments.early_temperature
                        if item["turns"] < arguments.temperature_turns
                        else 0.0
                    )
                    first, second = item["native_context"].choose_action(
                        temperature=temperature,
                    )
                    if opening_turn and second != PASS_MOVE:
                        raise RuntimeError("native opening search returned two stones")
                    item["native_context"].play(first, second)
                    game.play(first)
                    item["legacy_context"].update_state(first)
                    if opening_turn:
                        if game.winner:
                            raise RuntimeError("opening stone unexpectedly ended the game")
                    elif game.winner:
                        if second != PASS_MOVE:
                            raise RuntimeError("native search played after a winning first stone")
                    else:
                        if second == PASS_MOVE:
                            raise RuntimeError("native search returned a non-terminal singleton")
                        game.play(second)
                        item["legacy_context"].update_state(second)
                    item["turns"] += 1
                    verify_synchronized(game, item["native_context"])
                    if not game.winner:
                        item["native_context"].reset_tree()

            if legacy_indices:
                legacy_contexts = [games[index]["legacy_context"] for index in legacy_indices]
                legacy_engine.run_simulations_multi(
                    legacy_contexts,
                    arguments.legacy_simulations,
                )
                pending_second = []
                first_moves = {}
                for index in legacy_indices:
                    item = games[index]
                    opening_turn = not item["game"].moves
                    temperature = (
                        arguments.early_temperature
                        if item["turns"] < arguments.temperature_turns
                        else 0.0
                    )
                    first = item["legacy_context"].get_mcts_move(
                        temperature=temperature,
                    )
                    first_moves[index] = first
                    item["game"].play(first)
                    item["legacy_context"].update_state(first)
                    if opening_turn:
                        if item["game"].winner:
                            raise RuntimeError("opening stone unexpectedly ended the game")
                        item["native_context"].play(first, PASS_MOVE)
                        item["turns"] += 1
                        verify_synchronized(item["game"], item["native_context"])
                        item["native_context"].reset_tree()
                    elif item["game"].winner:
                        item["native_context"].play(first, PASS_MOVE)
                        item["turns"] += 1
                        verify_synchronized(item["game"], item["native_context"])
                    else:
                        pending_second.append(index)

                if pending_second:
                    contexts = [games[index]["legacy_context"] for index in pending_second]
                    legacy_engine.run_simulations_multi(
                        contexts,
                        arguments.legacy_simulations,
                    )
                    for index in pending_second:
                        item = games[index]
                        temperature = (
                            arguments.early_temperature
                            if item["turns"] < arguments.temperature_turns
                            else 0.0
                        )
                        second = item["legacy_context"].get_mcts_move(
                            temperature=temperature,
                        )
                        item["game"].play(second)
                        item["legacy_context"].update_state(second)
                        first = first_moves[index]
                        item["native_context"].play(*sorted((first, second)))
                        item["turns"] += 1
                        if item["game"].winner:
                            verify_terminal_result(item["game"], item["native_context"])
                        else:
                            verify_synchronized(item["game"], item["native_context"])
                            item["native_context"].reset_tree()

            active = [index for index in active if not games[index]["game"].winner]
            if active and max(games[index]["turns"] for index in active) > 180:
                raise RuntimeError("arena exceeded the legal maximum number of turns")
        torch.cuda.synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        for item in games:
            item["native_context"].close()
            item["legacy_context"].close()

    native_wins = 0
    legacy_wins = 0
    draws = 0
    black_wins = 0
    white_wins = 0
    pair_scores = []
    for pair_index in range(arguments.pairs):
        score = 0.0
        for item in (game for game in games if game["pair"] == pair_index):
            winner = normalized_winner(item["game"])
            if winner == 0:
                draws += 1
                score += 0.5
            else:
                black_wins += int(winner == 1)
                white_wins += int(winner == -1)
                native_color = 1 if item["native_black"] else -1
                if winner == native_color:
                    native_wins += 1
                    score += 1.0
                else:
                    legacy_wins += 1
        pair_scores.append(score / 2.0)

    total = len(games)
    native_score = (native_wins + 0.5 * draws) / total
    paired_score_standard_error = paired_standard_error(pair_scores)
    binomial_score_standard_error = math.sqrt(
        max(native_score * (1.0 - native_score), 0.0) / total
    )
    common_opening_pairs = 0
    for pair_index in range(arguments.pairs):
        pair_openings = {
            str(item["game"].moves[0])
            for item in games
            if item["pair"] == pair_index
        }
        common_opening_pairs += int(len(pair_openings) == 1)
    natural_opening = arguments.opening_mode == "natural"
    result = {
        "schema_version": 3,
        "pairs": arguments.pairs,
        "games": total,
        "opening_mode": arguments.opening_mode,
        "openings": [int(opening) for opening in openings if opening is not None],
        "actual_openings": [str(item["game"].moves[0]) for item in games],
        "unique_actual_openings": len({str(item["game"].moves[0]) for item in games}),
        "game_openings": [
            {
                "pair": item["pair"],
                "black_model": "native" if item["native_black"] else "legacy",
                "move": str(item["game"].moves[0]),
            }
            for item in games
        ],
        "opening_radius": arguments.opening_radius,
        "early_temperature": arguments.early_temperature,
        "temperature_turns": arguments.temperature_turns,
        "native_checkpoint": str(native_checkpoint),
        "native_sha256": sha256_file(native_checkpoint),
        "legacy_checkpoint": str(legacy_checkpoint),
        "legacy_sha256": sha256_file(legacy_checkpoint),
        "legacy_engine": str(legacy_engine_path),
        "legacy_engine_sha256": sha256_file(legacy_engine_path),
        "legacy_engine_manifest": str(legacy_manifest_path),
        "legacy_engine_manifest_payload": legacy_manifest,
        "native_simulations_per_turn": arguments.native_simulations,
        "legacy_simulations_per_stone": arguments.legacy_simulations,
        "legacy_nominal_simulations_per_full_turn": 2 * arguments.legacy_simulations,
        "top_k": arguments.top_k,
        "native_wins": native_wins,
        "legacy_wins": legacy_wins,
        "draws": draws,
        "native_score": native_score,
        "native_score_standard_error": (
            binomial_score_standard_error
            if natural_opening
            else paired_score_standard_error
        ),
        "native_score_standard_error_method": (
            "binomial_independent_games"
            if natural_opening
            else "paired_common_opening"
        ),
        "native_score_paired_standard_error": (
            None if natural_opening else paired_score_standard_error
        ),
        "native_score_binomial_standard_error": binomial_score_standard_error,
        "pair_opening_match_rate": common_opening_pairs / arguments.pairs,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "mean_turns": float(np.mean([
            item["turns"] + int(not natural_opening)
            for item in games
        ])),
        "mean_searched_turns": float(np.mean([item["turns"] for item in games])),
        "opening_searches_per_game": int(natural_opening),
        "mean_stones": float(np.mean([len(item["game"].moves) for item in games])),
        "seconds": elapsed,
        "games_per_second": total / elapsed,
        "native_evaluator": native_evaluator.stats(),
        "legacy_search": legacy_engine.get_search_stats(),
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
