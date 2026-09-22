"""Screen adaptive MCTS budgets against a fixed full-search teacher."""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RL_DIR = os.path.join(PROJECT_ROOT, "reinforcement_learning")
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)


def coordinate_to_index(value):
    value = value.strip().lower()
    if len(value) < 2 or not "a" <= value[0] <= "s":
        return None
    try:
        row = int(value[1:]) - 1
    except ValueError:
        return None
    column = ord(value[0]) - ord("a")
    if not 0 <= row < 19:
        return None
    return row * 19 + column


def player_at_move(move_index):
    if move_index == 0:
        return 1
    return -1 if ((move_index + 1) // 2) % 2 else 1


def sample_prefixes(paths, count, seed, player, minimum_stones, maximum_stones):
    rng = random.Random(seed)
    samples = []
    seen = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as source:
            for row in csv.DictReader(source):
                moves = [
                    coordinate_to_index(value)
                    for value in row.get("moves", "").split(",")
                    if value.strip()
                ]
                if not moves or any(move is None for move in moves):
                    continue
                candidates = [
                    index
                    for index in range(minimum_stones, min(len(moves), maximum_stones + 1))
                    if player == "all"
                    or player_at_move(index) == (1 if player == "black" else -1)
                ]
                if not candidates:
                    continue
                prefix_length = rng.choice(candidates)
                prefix = tuple(moves[:prefix_length])
                seen += 1
                if len(samples) < count:
                    samples.append(prefix)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < count:
                        samples[replacement] = prefix
    if len(samples) < count:
        raise RuntimeError(f"only found {len(samples)} eligible game prefixes")
    return samples


def policy_summary(policy):
    order = np.argsort(policy)[::-1]
    first = int(order[0])
    second_probability = float(policy[order[1]]) if order.size > 1 else 0.0
    ratio = float("inf") if second_probability <= 0 else float(policy[first]) / second_probability
    return first, set(int(value) for value in order[:5]), ratio


def evaluate_threshold(checkpoints, threshold, full_budget):
    total_used = 0
    top1_matches = 0
    top5_overlaps = 0.0
    l1_total = 0.0
    stopped_early = 0
    for policies in checkpoints:
        final_policy = policies[-1][1]
        final_top1, final_top5, _ = policy_summary(final_policy)
        chosen_budget, chosen_policy = policies[-1]
        for budget, policy in policies[:-1]:
            _, _, ratio = policy_summary(policy)
            if ratio >= threshold:
                chosen_budget, chosen_policy = budget, policy
                stopped_early += 1
                break
        chosen_top1, chosen_top5, _ = policy_summary(chosen_policy)
        total_used += chosen_budget
        top1_matches += int(chosen_top1 == final_top1)
        top5_overlaps += len(chosen_top5 & final_top5) / 5.0
        l1_total += float(np.abs(chosen_policy - final_policy).sum())

    count = len(checkpoints)
    mean_budget = total_used / count
    return {
        "threshold": threshold,
        "mean_simulations": mean_budget,
        "simulation_saving": 1.0 - mean_budget / full_budget,
        "early_stop_rate": stopped_early / count,
        "top1_agreement": top1_matches / count,
        "mean_top5_overlap": top5_overlaps / count,
        "mean_policy_l1": l1_total / count,
    }


def evaluate_stable_threshold(checkpoints, threshold, full_budget):
    """Require the leading move to survive two consecutive checkpoints."""

    total_used = 0
    top1_matches = 0
    top5_overlaps = 0.0
    l1_total = 0.0
    stopped_early = 0
    for policies in checkpoints:
        final_policy = policies[-1][1]
        final_top1, final_top5, _ = policy_summary(final_policy)
        chosen_budget, chosen_policy = policies[-1]
        previous_top1, _, _ = policy_summary(policies[0][1])
        for budget, policy in policies[1:-1]:
            current_top1, _, ratio = policy_summary(policy)
            if current_top1 == previous_top1 and ratio >= threshold:
                chosen_budget, chosen_policy = budget, policy
                stopped_early += 1
                break
            previous_top1 = current_top1
        chosen_top1, chosen_top5, _ = policy_summary(chosen_policy)
        total_used += chosen_budget
        top1_matches += int(chosen_top1 == final_top1)
        top5_overlaps += len(chosen_top5 & final_top5) / 5.0
        l1_total += float(np.abs(chosen_policy - final_policy).sum())

    count = len(checkpoints)
    mean_budget = total_used / count
    return {
        "threshold": threshold,
        "mean_simulations": mean_budget,
        "simulation_saving": 1.0 - mean_budget / full_budget,
        "early_stop_rate": stopped_early / count,
        "top1_agreement": top1_matches / count,
        "mean_top5_overlap": top5_overlaps / count,
        "mean_policy_l1": l1_total / count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate adaptive-search savings from replay positions",
    )
    parser.add_argument("--engine", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--mcts-library", default=None)
    parser.add_argument("--pair-heads", default=None)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--checkpoints", default="400,800,1200")
    parser.add_argument("--thresholds", default="2,4,6,10")
    parser.add_argument("--player", choices=("black", "white", "all"), default="white")
    parser.add_argument("--minimum-stones", type=int, default=5)
    parser.add_argument("--maximum-stones", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    budgets = [int(value) for value in args.checkpoints.split(",") if value.strip()]
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] <= 0:
        raise ValueError("checkpoints must be unique increasing positive integers")
    if any(threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be greater than one")
    if args.mcts_library:
        os.environ["NEBULA_MCTS_LIBRARY"] = os.path.abspath(args.mcts_library)
    if args.pair_heads:
        os.environ["NEBULA_PAIR_HEADS"] = os.path.abspath(args.pair_heads)

    import torch
    from core.mcts import MCTSEngine

    prefixes = sample_prefixes(
        args.data,
        args.positions,
        args.seed,
        args.player,
        args.minimum_stones,
        args.maximum_stones,
    )
    engine = MCTSEngine(args.engine, device=torch.device("cuda:0"))
    if not engine.supports_multi_context:
        raise RuntimeError("MCTS library does not support multi-context search")
    engine.set_params(batch_size=args.batch_size, num_threads=args.threads)
    engine.set_deterministic_selection(True)

    all_checkpoints = []
    for group_start in range(0, len(prefixes), args.group_size):
        group = prefixes[group_start:group_start + args.group_size]
        contexts = []
        try:
            for offset, prefix in enumerate(group):
                context = engine.create_game_context(
                    seed=args.seed + group_start + offset,
                )
                for move in prefix:
                    context.update_state(move)
                contexts.append(context)
            group_policies = [[] for _ in contexts]
            completed = 0
            for budget in budgets:
                increment = budget - completed
                engine.run_simulations_multi(contexts, [increment] * len(contexts))
                for index, context in enumerate(contexts):
                    group_policies[index].append((budget, context.get_policy().copy()))
                completed = budget
            all_checkpoints.extend(group_policies)
        finally:
            for context in contexts:
                context.close()

    print(json.dumps({
        "engine": os.path.abspath(args.engine),
        "positions": len(all_checkpoints),
        "player": args.player,
        "stone_range": [args.minimum_stones, args.maximum_stones],
        "checkpoints": budgets,
        "results": [
            evaluate_threshold(all_checkpoints, threshold, budgets[-1])
            for threshold in thresholds
        ],
        "stable_top1_results": [
            evaluate_stable_threshold(all_checkpoints, threshold, budgets[-1])
            for threshold in thresholds
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
