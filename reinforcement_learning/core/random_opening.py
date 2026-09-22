"""Reproducible network-independent opening prefixes for self-play."""

import numpy as np


def build_random_opening(
    game_seed,
    ratio,
    fixed_stones=5,
    stone_choices=(),
    radius=4,
    uniform_board=False,
    game_index=None,
):
    ratio = min(1.0, max(0.0, float(ratio)))
    choices = tuple(int(value) for value in stone_choices)
    fixed_stones = max(0, int(fixed_stones))
    if any(value < 0 or value > 361 for value in choices):
        raise ValueError("random opening stone counts must be in 0..361")
    if ratio <= 0.0 or (not choices and fixed_stones <= 0):
        return []

    seed = int(game_seed) % 2_147_483_647
    rng = np.random.default_rng((seed + 7919) % 2_147_483_647)
    if float(rng.random()) >= ratio:
        return []

    if choices:
        stones = int(
            rng.choice(choices)
            if game_index is None
            else choices[int(game_index) % len(choices)]
        )
    else:
        stones = fixed_stones
    if stones <= 0:
        return []

    if uniform_board:
        candidates = np.arange(361, dtype=np.int32)
    else:
        radius = min(9, max(0, int(radius)))
        candidates = np.asarray([
            row * 19 + column
            for row in range(9 - radius, 10 + radius)
            for column in range(9 - radius, 10 + radius)
        ], dtype=np.int32)
    count = min(stones, len(candidates))
    return [
        int(value)
        for value in rng.choice(candidates, size=count, replace=False)
    ]
