"""Competition 对主线 MCTS/TensorRT 包装器的兼容入口。

网页曾维护一份独立的 17 平面推理代码，但当前生产引擎已经使用融合原始棋盘
输入。继续复制实现既无法加载 current_model.engine，也容易遗漏 CUDA Graph、
动态 batch 和错误检查，因此这里只保留稳定的导入边界。
"""

import importlib.util
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MAIN_MCTS_PATH = os.path.join(
    PROJECT_ROOT,
    "reinforcement_learning",
    "core",
    "mcts.py",
)


def _load_main_mcts_module():
    """用独立模块名加载主线实现，避免与当前 core.mcts 名称递归。"""

    specification = importlib.util.spec_from_file_location(
        "nebula_reinforcement_mcts",
        MAIN_MCTS_PATH,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载主线 MCTS 模块：{MAIN_MCTS_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_main_mcts = _load_main_mcts_module()

MCTSEngine = _main_mcts.MCTSEngine
MCTSGameContext = _main_mcts.MCTSGameContext
mcts_lib = _main_mcts.mcts_lib

__all__ = ["MCTSEngine", "MCTSGameContext", "mcts_lib"]
