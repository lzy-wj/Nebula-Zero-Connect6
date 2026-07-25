"""Bounded feedback controller for black/white self-play data balance."""


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def white_win_weight(
    white_win_rate,
    target_fraction=0.40,
    minimum=1.0,
    maximum=3.0,
):
    """Weight white-win games so their expected loss contribution hits a target."""

    rate = clamp(float(white_win_rate), 0.0, 1.0)
    target = clamp(float(target_fraction), 0.01, 0.99)
    if rate <= 0.0:
        return float(maximum)
    if rate >= target:
        return float(minimum)
    value = target * (1.0 - rate) / (rate * (1.0 - target))
    return float(clamp(value, minimum, maximum))


def combine_opening_counts(*qualities):
    result = {
        "standard_games": 0,
        "standard_black_wins": 0,
        "injected_games": 0,
        "injected_black_wins": 0,
    }
    for quality in qualities:
        if not isinstance(quality, dict):
            continue
        for key in result:
            result[key] += int(quality.get(key, 0))
    return result


def update_opening_ratio(
    current_ratio,
    counts,
    previous_ema=None,
    target_black_rate=0.50,
    deadband=0.03,
    ema_alpha=0.35,
    max_step=0.10,
    minimum_ratio=0.0,
    maximum_ratio=0.90,
    minimum_source_games=20,
):
    """Choose next generation's injected-opening ratio without chasing noise."""

    standard_games = int(counts.get("standard_games", 0))
    injected_games = int(counts.get("injected_games", 0))
    standard_wins = int(counts.get("standard_black_wins", 0))
    injected_wins = int(counts.get("injected_black_wins", 0))
    games = standard_games + injected_games
    black_wins = standard_wins + injected_wins
    observed = black_wins / games if games else 0.5
    ema = (
        observed
        if previous_ema is None
        else (1.0 - ema_alpha) * float(previous_ema) + ema_alpha * observed
    )
    current = clamp(float(current_ratio), minimum_ratio, maximum_ratio)
    next_ratio = current
    reason = "deadband"
    standard_rate = standard_wins / standard_games if standard_games else 0.0
    injected_rate = injected_wins / injected_games if injected_games else 0.0

    enough_sources = (
        standard_games >= minimum_source_games
        and injected_games >= minimum_source_games
    )
    if not enough_sources:
        reason = "insufficient_source_games"
    elif abs(ema - target_black_rate) <= deadband:
        reason = "deadband"
    elif abs(standard_rate - injected_rate) < 0.02:
        reason = "opening_mix_has_no_measurable_effect"
    else:
        desired = (standard_rate - target_black_rate) / (
            standard_rate - injected_rate
        )
        desired = clamp(desired, minimum_ratio, maximum_ratio)
        delta = clamp(desired - current, -max_step, max_step)
        next_ratio = clamp(current + delta, minimum_ratio, maximum_ratio)
        reason = "source_rate_interpolation"

    return {
        "observed_black_win_rate": observed,
        "black_win_rate_ema": ema,
        "standard_games": standard_games,
        "standard_black_win_rate": standard_rate,
        "injected_games": injected_games,
        "injected_black_win_rate": injected_rate,
        "current_opening_ratio": current,
        "next_opening_ratio": next_ratio,
        "target_black_win_rate": target_black_rate,
        "deadband": deadband,
        "reason": reason,
    }
