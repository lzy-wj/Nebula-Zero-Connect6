import os
import sys

import numpy as np
import pytest


RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning'))
sys.path.insert(0, RL_DIR)

from core.connect6_game import Connect6Game


@pytest.mark.parametrize("move", [-1, 361, 1.5, "0", True, np.bool_(False)])
def test_play_rejects_invalid_move_indices(move):
    game = Connect6Game()

    with pytest.raises(ValueError, match="Invalid move index"):
        game.play(move)

    assert not game.moves
    assert np.count_nonzero(game.board) == 0


def test_play_accepts_numpy_integer_index():
    game = Connect6Game()

    game.play(np.int64(360))

    assert game.board[18, 18] == 1
    assert game.moves == ["s19"]
