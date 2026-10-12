"""Measure MCTS budget sufficiency on audited held-out self-play positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.pair_mcts import PairMCTSEngine, TorchPairEvaluator
from experiments.pair_native.search_consistency import (
    js_divergence,
    load_states,
    policy_dict,
    replay_to_state,
)


def phase_of(row) -> int:
    stones = int(row["stones_before"])
    return 0 if stones < 32 else 1 if stones <= 64 else 2


def top_action(policy: dict) -> tuple[int, int]:
    return max(policy, key=policy.get)


def summarize(indices, policies, values, reference_policies, reference_values):
    agreements = []
    divergences = []
    value_errors = []
    for index in indices:
        agreements.append(
            float(top_action(policies[index]) == top_action(reference_policies[index]))
        )
        divergences.append(js_divergence(policies[index], reference_policies[index]))
        value_errors.append(abs(float(values[index]) - float(reference_values[index])))
    return {
        "states": len(indices),
        "top1_agreement": float(np.mean(agreements)) if indices else None,
        "js_divergence": float(np.mean(divergences)) if indices else None,
        "root_value_mae": float(np.mean(value_errors)) if indices else None,
    }


def run_budget(engine, games, rows, budget: int, seed: int):
    contexts = [engine.create_context(seed + index) for index in range(len(rows))]
    try:
        for context, row in zip(contexts, rows):
            replay_to_state(context, games[row["game_id"]]["moves"], row["stones_before"])
        engine.reset_global_stats()
        engine.run(contexts, int(budget))
        return {
            "policies": [policy_dict(context) for context in contexts],
            "values": [context.root_value() for context in contexts],
            "context_stats": [context.stats() for context in contexts],
            "engine_stats": engine.global_stats(),
        }
    finally:
        for context in contexts:
            context.close()


def run_budget_series(engine, games, rows, budgets: list[int], seed: int):
    """Reuse each tree so the sweep costs only the largest requested budget."""
    contexts = [engine.create_context(seed + index) for index in range(len(rows))]
    results = {}
    previous = 0
    cumulative = {
        "simulations": 0,
        "neural_evaluations": 0,
        "eval_batches": 0,
        "tactical_expansions": 0,
    }
    try:
        for context, row in zip(contexts, rows):
            replay_to_state(context, games[row["game_id"]]["moves"], row["stones_before"])
        for budget in sorted(budgets):
            engine.reset_global_stats()
            engine.run(contexts, int(budget) - previous)
            incremental = engine.global_stats()
            for key in cumulative:
                cumulative[key] += int(incremental[key])
            cumulative_stats = dict(cumulative)
            cumulative_stats["average_batch"] = (
                cumulative["neural_evaluations"] / cumulative["eval_batches"]
                if cumulative["eval_batches"]
                else 0.0
            )
            results[budget] = {
                "policies": [policy_dict(context) for context in contexts],
                "values": [context.root_value() for context in contexts],
                "engine_stats": cumulative_stats,
                "incremental_engine_stats": incremental,
            }
            previous = int(budget)
        return results
    finally:
        for context in contexts:
            context.close()


def reference_confidence(policy: dict) -> float:
    values = sorted(policy.values(), reverse=True)
    return float(values[0]) if values else 0.0


def confidence_groups(policies: list[dict]) -> dict[str, list[int]]:
    groups = {
        "uncertain_lt_0.50": [],
        "clear_0.50_to_0.75": [],
        "easy_ge_0.75": [],
    }
    for index, policy in enumerate(policies):
        confidence = reference_confidence(policy)
        key = (
            "uncertain_lt_0.50"
            if confidence < 0.50
            else "clear_0.50_to_0.75"
            if confidence < 0.75
            else "easy_ge_0.75"
        )
        groups[key].append(index)
    return groups


def adaptive_summary(name, choices, runs, reference):
    policies = [runs[budget]["policies"][index] for index, budget in enumerate(choices)]
    values = [runs[budget]["values"][index] for index, budget in enumerate(choices)]
    counts = {str(budget): choices.count(budget) for budget in sorted(set(choices))}
    return {
        "name": name,
        "average_budget": float(np.mean(choices)),
        "budget_counts": counts,
        **summarize(
            list(range(len(choices))),
            policies,
            values,
            reference["policies"],
            reference["values"],
        ),
    }


def choose_adaptive_budgets(runs, minimum: int) -> list[int]:
    choices = []
    states = len(next(iter(runs.values()))["policies"])
    for index in range(states):
        action = {
            budget: top_action(run["policies"][index])
            for budget, run in runs.items()
        }
        confidence = {
            budget: reference_confidence(run["policies"][index])
            for budget, run in runs.items()
        }
        if (
            minimum <= 200
            and 128 in runs
            and 200 in runs
            and action[128] == action[200]
            and confidence[200] >= 0.75
        ):
            choices.append(200)
        elif (
            minimum <= 300
            and 200 in runs
            and 300 in runs
            and action[200] == action[300]
            and confidence[300] >= 0.75
        ):
            choices.append(300)
        elif (
            300 in runs
            and 400 in runs
            and action[300] == action[400]
            and confidence[400] >= 0.70
        ):
            choices.append(400)
        elif (
            600 in runs
            and 800 in runs
            and action[600] == action[800]
            and confidence[800] >= 0.60
        ):
            choices.append(800)
        else:
            choices.append(1200)
    return choices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--budgets", default="128,200,300,400,600,800,1200")
    parser.add_argument("--reference-budget", type=int, default=4000)
    parser.add_argument("--deep-reference-budget", type=int, default=10000)
    parser.add_argument("--deep-states", type=int, default=24)
    parser.add_argument("--per-phase", type=int, default=24)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026092723)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    budgets = sorted({int(value) for value in arguments.budgets.split(",")})
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("budgets must be positive")
    if arguments.reference_budget <= max(budgets):
        raise ValueError("reference budget must exceed all candidate budgets")
    if arguments.deep_reference_budget < arguments.reference_budget:
        raise ValueError("deep reference budget must not be smaller than reference")

    games, rows = load_states(
        Path(arguments.data_dir).expanduser().resolve(),
        arguments.validation_fraction,
        arguments.per_phase,
    )
    model, _ = load_checkpoint(arguments.checkpoint)
    evaluator = TorchPairEvaluator(
        model,
        torch.device("cuda:0"),
        top_k=arguments.top_k,
        candidate_gumbel=0.0,
        compile_model=True,
        deduplicate=False,
        pad_batch_to=len(rows),
        value_scale=arguments.value_scale,
    )
    engine = PairMCTSEngine(
        evaluator,
        top_k=arguments.top_k,
        root_noise_epsilon=0.0,
        cpu_threads=arguments.cpu_threads,
    )

    runs = run_budget_series(
        engine,
        games,
        rows,
        [*budgets, arguments.reference_budget],
        arguments.seed,
    )
    reference = runs[arguments.reference_budget]
    all_indices = list(range(len(rows)))
    phase_indices = {
        str(phase): [index for index, row in enumerate(rows) if phase_of(row) == phase]
        for phase in range(3)
    }
    confidence_indices = confidence_groups(reference["policies"])

    results = []
    for budget in budgets:
        run = runs[budget]
        results.append({
            "budget": budget,
            **summarize(
                all_indices,
                run["policies"],
                run["values"],
                reference["policies"],
                reference["values"],
            ),
            "by_phase": {
                phase: summarize(
                    indices,
                    run["policies"],
                    run["values"],
                    reference["policies"],
                    reference["values"],
                )
                for phase, indices in phase_indices.items()
            },
            "by_reference_confidence": {
                name: summarize(
                    indices,
                    run["policies"],
                    run["values"],
                    reference["policies"],
                    reference["values"],
                )
                for name, indices in confidence_indices.items()
            },
            "by_search_confidence": {
                name: summarize(
                    indices,
                    run["policies"],
                    run["values"],
                    reference["policies"],
                    reference["values"],
                )
                for name, indices in confidence_groups(run["policies"]).items()
            },
            "engine_stats": run["engine_stats"],
            "incremental_engine_stats": run["incremental_engine_stats"],
        })

    adaptive = []
    if all(budget in runs for budget in (128, 200, 300, 400, 600, 800, 1200)):
        for minimum in (200, 300, 400):
            choices = choose_adaptive_budgets(runs, minimum)
            adaptive.append(
                adaptive_summary(f"adaptive_min_{minimum}", choices, runs, reference)
            )

    comparison_budget = 400 if 400 in runs else budgets[-1]
    comparison = runs[comparison_budget]
    difficulty = [
        (
            js_divergence(policy, reference["policies"][index]),
            float(top_action(policy) != top_action(reference["policies"][index])),
            index,
        )
        for index, policy in enumerate(comparison["policies"])
    ]
    difficulty.sort(reverse=True)
    hard_indices = [item[2] for item in difficulty[: min(arguments.deep_states, len(rows))]]
    hard_rows = [rows[index] for index in hard_indices]
    deep = run_budget(
        engine,
        games,
        hard_rows,
        arguments.deep_reference_budget,
        arguments.seed,
    )
    hard_confirmation = {}
    for budget in [comparison_budget, 1200, arguments.reference_budget]:
        if budget not in runs:
            continue
        source = runs[budget]
        selected_policies = [source["policies"][index] for index in hard_indices]
        selected_values = [source["values"][index] for index in hard_indices]
        hard_confirmation[str(budget)] = summarize(
            list(range(len(hard_indices))),
            selected_policies,
            selected_values,
            deep["policies"],
            deep["values"],
        )

    payload = {
        "schema_version": 1,
        "checkpoint": str(Path(arguments.checkpoint).expanduser().resolve()),
        "data_dir": str(Path(arguments.data_dir).expanduser().resolve()),
        "states": len(rows),
        "states_by_phase": {key: len(value) for key, value in phase_indices.items()},
        "states_by_reference_confidence": {
            key: len(value) for key, value in confidence_indices.items()
        },
        "reference_budget": arguments.reference_budget,
        "deep_reference_budget": arguments.deep_reference_budget,
        "results": results,
        "adaptive_results": adaptive,
        "hard_confirmation": {
            "selection_budget": comparison_budget,
            "states": len(hard_indices),
            "results": hard_confirmation,
            "engine_stats": deep["engine_stats"],
        },
        "evaluator": evaluator.stats(),
    }
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
