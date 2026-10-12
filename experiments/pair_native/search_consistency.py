"""Compare low/high-budget MCTS consistency on held-out cold-start states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_mcts import PASS_MOVE, PairMCTSEngine, TorchPairEvaluator
from experiments.pair_native.pair_replay import audit_shard, load_shard
from experiments.pair_native.train import validation_game


def load_states(data_dir: Path, validation_fraction: float, per_phase: int):
    games = {}
    rows = []
    for manifest in sorted(data_dir.glob("*.manifest.json")):
        audit = audit_shard(manifest)
        if not audit["valid"]:
            raise ValueError(f"invalid shard {manifest}")
        _, shard_games, shard_turns = load_shard(manifest)
        games.update({game["game_id"]: game for game in shard_games})
        rows.extend(shard_turns)
    candidates = [
        row for row in rows if validation_game(row["game_id"], validation_fraction)
    ]
    selected = []
    for phase in range(3):
        phase_rows = [
            row
            for row in candidates
            if (
                (phase == 0 and row["stones_before"] < 32)
                or (phase == 1 and 32 <= row["stones_before"] <= 64)
                or (phase == 2 and row["stones_before"] > 64)
            )
        ]
        phase_rows.sort(key=lambda row: (row["game_id"], row["turn_index"]))
        if len(phase_rows) > per_phase:
            indices = np.linspace(0, len(phase_rows) - 1, per_phase, dtype=int)
            phase_rows = [phase_rows[index] for index in indices]
        selected.extend(phase_rows)
    return games, selected


def replay_to_state(context, moves, stones_before):
    if stones_before <= 0:
        return
    context.play(int(moves[0]), PASS_MOVE)
    index = 1
    while index < stones_before:
        first, second = sorted((int(moves[index]), int(moves[index + 1])))
        context.play(first, second)
        index += 2
    context.reset_tree()


def policy_dict(context):
    first, second, probability = context.policy()
    return {
        (int(left), int(right)): float(value)
        for left, right, value in zip(first, second, probability)
        if value > 0
    }


def js_divergence(left, right):
    actions = set(left) | set(right)
    if not actions:
        return 0.0
    left_values = np.array([left.get(action, 0.0) for action in actions], dtype=np.float64)
    right_values = np.array([right.get(action, 0.0) for action in actions], dtype=np.float64)
    left_values /= max(left_values.sum(), 1.0e-12)
    right_values /= max(right_values.sum(), 1.0e-12)
    middle = 0.5 * (left_values + right_values)

    def kl(values):
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / middle[mask])))

    return 0.5 * (kl(left_values) + kl(right_values))


def evaluate_checkpoint(
    checkpoint_path,
    games,
    rows,
    low_budget,
    high_budget,
    top_k,
    value_scale,
):
    model, _ = load_checkpoint(checkpoint_path)
    evaluator = TorchPairEvaluator(
        model,
        torch.device("cuda:0"),
        top_k=top_k,
        candidate_gumbel=0.0,
        compile_model=True,
        deduplicate=False,
        pad_batch_to=len(rows),
        value_scale=value_scale,
    )
    engine = PairMCTSEngine(
        evaluator,
        top_k=top_k,
        root_noise_epsilon=0.0,
    )
    low_contexts = [engine.create_context(10_000 + index) for index in range(len(rows))]
    high_contexts = [engine.create_context(20_000 + index) for index in range(len(rows))]
    try:
        for context, row in zip(low_contexts, rows):
            replay_to_state(context, games[row["game_id"]]["moves"], row["stones_before"])
        for context, row in zip(high_contexts, rows):
            replay_to_state(context, games[row["game_id"]]["moves"], row["stones_before"])
        engine.run(low_contexts, low_budget)
        low_policies = [policy_dict(context) for context in low_contexts]
        engine.run(high_contexts, high_budget)
        high_policies = [policy_dict(context) for context in high_contexts]
        top1_matches = []
        divergences = []
        phase_results = {str(phase): [] for phase in range(3)}
        for row, low, high in zip(rows, low_policies, high_policies):
            low_top = max(low, key=low.get)
            high_top = max(high, key=high.get)
            match = float(low_top == high_top)
            divergence = js_divergence(low, high)
            top1_matches.append(match)
            divergences.append(divergence)
            phase = 0 if row["stones_before"] < 32 else 1 if row["stones_before"] <= 64 else 2
            phase_results[str(phase)].append((match, divergence))
        return {
            "states": len(rows),
            "low_budget": low_budget,
            "high_budget": high_budget,
            "top1_agreement": float(np.mean(top1_matches)),
            "js_divergence": float(np.mean(divergences)),
            "by_phase": {
                phase: {
                    "states": len(values),
                    "top1_agreement": float(np.mean([item[0] for item in values])),
                    "js_divergence": float(np.mean([item[1] for item in values])),
                }
                for phase, values in phase_results.items()
                if values
            },
            "evaluator": evaluator.stats(),
        }
    finally:
        for context in low_contexts + high_contexts:
            context.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--per-phase", type=int, default=16)
    parser.add_argument("--low-budget", type=int, default=64)
    parser.add_argument("--high-budget", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--value-scale", type=float, default=0.0)
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()
    games, rows = load_states(
        Path(arguments.data_dir).expanduser().resolve(),
        arguments.validation_fraction,
        arguments.per_phase,
    )
    result = {
        "initial": evaluate_checkpoint(
            arguments.initial_checkpoint,
            games,
            rows,
            arguments.low_budget,
            arguments.high_budget,
            arguments.top_k,
            arguments.value_scale,
        ),
        "trained": evaluate_checkpoint(
            arguments.trained_checkpoint,
            games,
            rows,
            arguments.low_budget,
            arguments.high_budget,
            arguments.top_k,
            arguments.value_scale,
        ),
    }
    payload = json.dumps(result, indent=2)
    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
