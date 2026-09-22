import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RL_DIR = os.path.join(ROOT, "reinforcement_learning")
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from core.connect6_game import Connect6Game
from core.random_opening import build_random_opening


def test_uniform_random_openings_are_reproducible_and_legal():
    arguments = {
        "game_seed": 20261004,
        "game_index": 3,
        "ratio": 1.0,
        "stone_choices": (1, 3, 5, 7),
        "uniform_board": True,
    }
    opening = build_random_opening(**arguments)

    assert opening == build_random_opening(**arguments)
    assert len(opening) == 7
    assert len(set(opening)) == len(opening)
    assert all(0 <= move < 361 for move in opening)

    game = Connect6Game()
    for move in opening:
        game.play(move)
    assert len(game.moves) == len(opening)


def test_stratified_prefixes_balance_the_side_to_move():
    next_players = []
    lengths = []
    outside_central_region = False
    for game_index in range(40):
        opening = build_random_opening(
            game_seed=1000 + game_index,
            game_index=game_index,
            ratio=1.0,
            stone_choices=(1, 3, 5, 7),
            uniform_board=True,
        )
        game = Connect6Game()
        for move in opening:
            game.play(move)
        lengths.append(len(opening))
        next_players.append(game.current_player)
        outside_central_region = outside_central_region or any(
            move // 19 < 5
            or move // 19 > 13
            or move % 19 < 5
            or move % 19 > 13
            for move in opening
        )

    assert set(lengths) == {1, 3, 5, 7}
    assert next_players.count(1) == next_players.count(-1)
    assert outside_central_region
