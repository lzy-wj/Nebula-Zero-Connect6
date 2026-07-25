"""不依赖 CUDA 的多棋局 MCTS ABI 与缓存回归测试。"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "reinforcement_learning" / "core"


@pytest.fixture(scope="module")
def mcts_library(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("系统没有 g++，跳过 C++ MCTS 测试")

    output = tmp_path_factory.mktemp("mcts") / "libmcts_test.so"
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-O2",
            "-fopenmp",
            "-std=c++17",
            f"-I{CORE}",
            str(CORE / "mcts_engine.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(output))

    callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    )

    def evaluate(batch_size, _boards, policies, values):
        """确定性假网络：中心附近先验更高，价值恒为零。"""

        policy_view = np.ctypeslib.as_array(
            policies,
            shape=(batch_size * 361,),
        ).reshape(batch_size, 361)
        coordinates = np.arange(361)
        rows, columns = np.divmod(coordinates, 19)
        prior = np.exp(-0.15 * ((rows - 9) ** 2 + (columns - 9) ** 2))
        prior = (prior / prior.sum()).astype(np.float32)
        policy_view[:] = prior
        np.ctypeslib.as_array(values, shape=(batch_size,)).fill(0.0)

    callback = callback_type(evaluate)
    library.set_eval_callback.argtypes = [callback_type]
    library.set_eval_callback(callback)
    # 防止 Python 回收 C 回调。
    library._test_callback = callback

    library.set_mcts_params.argtypes = [ctypes.c_int, ctypes.c_int]
    library.set_mcts_params(16, 4)
    library.set_eval_cache_capacity.argtypes = [ctypes.c_longlong]
    library.set_eval_cache_capacity(4096)
    library.create_mcts_context.argtypes = [ctypes.c_int]
    library.create_mcts_context.restype = ctypes.c_void_p
    library.destroy_mcts_context.argtypes = [ctypes.c_void_p]
    library.run_mcts_simulations_multi.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    library.get_root_action_count_context.argtypes = [ctypes.c_void_p]
    library.get_root_action_count_context.restype = ctypes.c_int
    library.get_root_active_action_count_context.argtypes = [ctypes.c_void_p]
    library.get_root_active_action_count_context.restype = ctypes.c_int
    library.get_best_move_context.argtypes = [ctypes.c_void_p, ctypes.c_float]
    library.get_best_move_context.restype = ctypes.c_int
    library.play_move_context.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.get_policy_context.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    for function_name in (
        "get_total_leaf_requests",
        "get_total_unique_evaluations",
        "get_total_cache_hits",
        "get_total_eval_batches",
    ):
        getattr(library, function_name).restype = ctypes.c_longlong
    return library


def test_multi_context_keeps_all_legal_actions_and_reuses_cache(mcts_library):
    contexts = [mcts_library.create_mcts_context(2026 + index) for index in range(8)]
    try:
        handles = (ctypes.c_void_p * len(contexts))(*contexts)
        budgets = (ctypes.c_int * len(contexts))(*([128] * len(contexts)))
        mcts_library.run_mcts_simulations_multi(handles, budgets, len(contexts))

        for context in contexts:
            # 动作总数没有被 top20 永久裁剪；参与 PUCT 的前缀会随访问增长。
            assert mcts_library.get_root_action_count_context(context) == 361
            assert mcts_library.get_root_active_action_count_context(context) > 20

            policy = np.zeros(361, dtype=np.float32)
            mcts_library.get_policy_context(
                context,
                policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            assert np.isfinite(policy).all()
            assert policy.sum() == pytest.approx(1.0, abs=1e-6)

        leaf_requests = mcts_library.get_total_leaf_requests()
        neural_evaluations = mcts_library.get_total_unique_evaluations()
        assert leaf_requests == 8 * 128
        assert 0 < neural_evaluations < leaf_requests
        assert mcts_library.get_total_cache_hits() > 0
        assert mcts_library.get_total_eval_batches() > 0

        # 各棋局拥有独立树，推进一盘不会改变其他棋局的合法动作数。
        move = mcts_library.get_best_move_context(contexts[0], ctypes.c_float(0.0))
        assert 0 <= move < 361
        mcts_library.play_move_context(contexts[0], move)
        assert mcts_library.get_root_action_count_context(contexts[0]) == 360
        assert mcts_library.get_root_action_count_context(contexts[1]) == 361
    finally:
        for context in contexts:
            mcts_library.destroy_mcts_context(context)

