"""Statistical promotion rules shared by the production and pair-policy loops."""

import math
import statistics
from statistics import NormalDist


def paired_score_statistics(game_scores):
    """Estimate superiority with each color-swapped opening pair as one sample."""

    scores = [float(value) for value in game_scores]
    if not scores:
        return {
            "paired_openings": 0,
            "paired_score_rate": 0.0,
            "paired_score_standard_error": 0.0,
            "improvement_probability": 0.0,
        }
    pair_scores = [
        statistics.fmean(scores[index:index + 2])
        for index in range(0, len(scores), 2)
    ]
    mean = statistics.fmean(pair_scores)
    if len(pair_scores) < 2:
        standard_error = 0.0
        probability = 0.5
    else:
        standard_error = statistics.stdev(pair_scores) / math.sqrt(len(pair_scores))
        if standard_error <= 1e-12:
            probability = 1.0 if mean > 0.5 else 0.5 if mean == 0.5 else 0.0
        else:
            probability = NormalDist().cdf((mean - 0.5) / standard_error)
    return {
        "paired_openings": len(pair_scores),
        "paired_score_rate": mean,
        "paired_score_standard_error": standard_error,
        "improvement_probability": probability,
    }


def approximate_improvement_probability(score_rate, games):
    """Fallback for legacy result files that predate paired statistics."""

    games = int(games)
    score_rate = float(score_rate)
    if games <= 1:
        return 0.5
    variance = max(score_rate * (1.0 - score_rate), 1e-9)
    standard_error = math.sqrt(variance / games)
    return NormalDist().cdf((score_rate - 0.5) / standard_error)


def gate_passes(
    result,
    minimum_score=0.5,
    minimum_confidence=0.90,
    minimum_pairs=50,
    minimum_white_win_rate=0.2,
    minimum_game_black_win_rate=0.35,
    maximum_game_black_win_rate=0.65,
):
    """Apply strength, sample-size, and color-health guards to one gate result."""

    confidence = float(
        result.get(
            "improvement_probability",
            approximate_improvement_probability(
                result.get("score_rate", 0.0),
                result.get("games", 0),
            ),
        )
    )
    paired_openings = int(
        result.get("paired_openings", int(result.get("games", 0)) // 2)
    )
    return (
        paired_openings >= int(minimum_pairs)
        and float(result.get("score_rate", 0.0)) >= float(minimum_score)
        and confidence >= float(minimum_confidence)
        and float(result.get("white_win_rate", 0.0))
        >= float(minimum_white_win_rate)
        and float(result.get("game_black_win_rate", 0.5))
        >= float(minimum_game_black_win_rate)
        and float(result.get("game_black_win_rate", 0.5))
        <= float(maximum_game_black_win_rate)
    )
