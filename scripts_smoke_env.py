#!/usr/bin/env python3
"""Minimal environment smoke for Nebula-Zero-Connect6."""
import os
import sys
import time
import ctypes
import unittest

import numpy as np
import torch
import tensorrt as trt


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    print("=== 1) env ===")
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
    assert torch.cuda.is_available(), "CUDA not available"
    print("gpu0", torch.cuda.get_device_name(0))
    print("tensorrt", trt.__version__)
    lib_path = os.path.abspath("reinforcement_learning/core/libmcts.so")
    print("libmcts", lib_path, "exists", os.path.exists(lib_path))
    assert os.path.exists(lib_path), "libmcts.so missing; run compile_mcts.py"

    print("=== 2) Connect6Game smoke ===")
    from reinforcement_learning.core.connect6_game import Connect6Game

    game = Connect6Game()
    # center j10 -> row 9, col 9 -> 180
    game.play(180)
    assert game.board[9, 9] == 1
    assert game.current_player == -1
    print("first move center ok, next player", game.current_player)

    print("=== 3) C6TransNet CPU/GPU forward ===")
    from reinforcement_learning.core.model import C6TransNet

    torch.manual_seed(0)
    model = C6TransNet().eval()
    x = torch.randn(2, 17, 19, 19)
    t0 = time.time()
    with torch.no_grad():
        policy1, policy2, value = model(x)
    print(
        "cpu forward",
        round(time.time() - t0, 4),
        "p1",
        tuple(policy1.shape),
        "v",
        tuple(value.shape),
        float(value.mean()),
    )
    assert policy1.shape == (2, 361)
    assert value.shape[0] == 2

    model = model.cuda()
    x = x.cuda()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        policy1, policy2, value = model(x)
    torch.cuda.synchronize()
    print(
        "gpu forward",
        round(time.time() - t0, 4),
        "p1",
        tuple(policy1.shape),
        "v mean",
        float(value.mean()),
    )
    assert torch.isfinite(policy1).all() and torch.isfinite(value).all()

    print("=== 4) libmcts ctypes ===")
    lib = ctypes.CDLL(lib_path)
    lib.init_game()
    lib.play_move(180)
    print("libmcts init+play_move ok")

    print("=== 5) unittest smokes ===")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod in [
        "tests.test_model_attention",
        "tests.test_balance_controller",
        "tests.test_nebula_v3",
    ]:
        suite.addTests(loader.loadTestsFromName(mod))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        print(
            "unittest failed",
            "failures",
            len(result.failures),
            "errors",
            len(result.errors),
        )
        return 1

    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
