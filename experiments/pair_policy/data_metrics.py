"""Streaming quality metrics for pair-policy game CSV files."""

import csv
import math
import statistics


WINNER_VALUE = {"black": 1, "white": -1, "draw": 0}


def player_at(move_index):
    if move_index == 0:
        return 1
    return -1 if ((move_index + 1) // 2) % 2 == 1 else 1


def parse_policy_values(text):
    values = []
    for item in text.split(";"):
        if not item:
            continue
        try:
            _, probability = item.split(":", 1)
            probability = float(probability)
        except (TypeError, ValueError):
            return []
        if probability > 0:
            values.append(probability)
    total = sum(values)
    return [value / total for value in values] if total > 0 else []


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def analyze_games(paths, opening_stones=5):
    """Return scalar, cross-generation-comparable data quality metrics."""

    if isinstance(paths, str):
        paths = [paths]
    winners = {name: 0 for name in WINNER_VALUE}
    game_lengths = []
    openings = set()
    policy_entropies = []
    first_entropies = []
    second_entropies = []
    policy_supports = []
    top1_masses = []
    invalid_games = 0
    missing_policies = 0
    injected_opening_games = 0
    standard_games = 0
    standard_black_wins = 0
    injected_black_wins = 0
    joint_samples = 0
    white_win_joint_samples = 0
    side_targets = {
        "black": {-1: 0, 0: 0, 1: 0},
        "white": {-1: 0, 0: 0, 1: 0},
    }

    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as source:
            for row in csv.DictReader(source):
                winner_name = row.get("winner", "").strip().lower()
                if winner_name not in WINNER_VALUE:
                    invalid_games += 1
                    continue
                moves = [value for value in row.get("moves", "").split(",") if value]
                policies = row.get("policies", "").split("|")
                winners[winner_name] += 1
                game_lengths.append(len(moves))
                openings.add(tuple(moves[:opening_stones]))
                if len(moves) != len(set(moves)) or len(moves) != len(policies):
                    invalid_games += 1

                valid_policy = []
                for move_index, policy_text in enumerate(policies):
                    values = parse_policy_values(policy_text)
                    valid_policy.append(bool(values))
                    if not values:
                        missing_policies += 1
                        continue
                    entropy = -sum(value * math.log(value) for value in values)
                    policy_entropies.append(entropy)
                    policy_supports.append(len(values))
                    top1_masses.append(max(values))
                    if move_index > 0 and move_index % 2 == 0:
                        second_entropies.append(entropy)
                    else:
                        first_entropies.append(entropy)

                if valid_policy and not valid_policy[0]:
                    injected_opening_games += 1
                    if winner_name == "black":
                        injected_black_wins += 1
                else:
                    standard_games += 1
                    if winner_name == "black":
                        standard_black_wins += 1

                winner = WINNER_VALUE[winner_name]
                for move_index in range(1, len(moves) - 1, 2):
                    if move_index + 1 >= len(valid_policy):
                        break
                    if not valid_policy[move_index] or not valid_policy[move_index + 1]:
                        continue
                    player = player_at(move_index)
                    side = "black" if player == 1 else "white"
                    side_targets[side][winner * player] += 1
                    joint_samples += 1
                    if winner_name == "white":
                        white_win_joint_samples += 1

    games = sum(winners.values())

    def mean(values):
        return float(statistics.fmean(values)) if values else 0.0

    def rate(value, total):
        return float(value / total) if total else 0.0

    black_samples = sum(side_targets["black"].values())
    white_samples = sum(side_targets["white"].values())
    policy_entropy = mean(policy_entropies)
    return {
        "games": games,
        "black_wins": winners["black"],
        "white_wins": winners["white"],
        "draws": winners["draw"],
        "black_win_rate": rate(winners["black"], games),
        "white_win_rate": rate(winners["white"], games),
        "draw_rate": rate(winners["draw"], games),
        "mean_stones": mean(game_lengths),
        "median_stones": percentile(game_lengths, 0.5),
        "p90_stones": percentile(game_lengths, 0.9),
        "opening_unique_ratio": rate(len(openings), games),
        "policy_entropy": policy_entropy,
        "effective_actions": math.exp(policy_entropy) if policy_entropies else 0.0,
        "first_policy_entropy": mean(first_entropies),
        "second_policy_entropy": mean(second_entropies),
        "policy_support": mean(policy_supports),
        "policy_top1_mass": mean(top1_masses),
        "joint_samples": joint_samples,
        "white_win_joint_samples": white_win_joint_samples,
        "white_win_sample_fraction": rate(
            white_win_joint_samples,
            joint_samples,
        ),
        "black_positive_target_rate": rate(side_targets["black"][1], black_samples),
        "white_positive_target_rate": rate(side_targets["white"][1], white_samples),
        "invalid_games": invalid_games,
        "invalid_game_rate": rate(invalid_games, games),
        "missing_policies": missing_policies,
        "injected_opening_games": injected_opening_games,
        "injected_opening_rate": rate(injected_opening_games, games),
        "standard_games": standard_games,
        "standard_black_wins": standard_black_wins,
        "standard_black_win_rate": rate(standard_black_wins, standard_games),
        "injected_games": injected_opening_games,
        "injected_black_wins": injected_black_wins,
        "injected_black_win_rate": rate(
            injected_black_wins,
            injected_opening_games,
        ),
    }
