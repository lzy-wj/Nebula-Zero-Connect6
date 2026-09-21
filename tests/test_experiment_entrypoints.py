import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "experiments/inference_vnext/export_onnx.py",
    "experiments/inference_vnext/validate_engine.py",
    "experiments/search_sparsity/probe_budget.py",
    "experiments/nebula_v3/evaluate_checkpoints.py",
    "experiments/nebula_v3/export_onnx.py",
    "experiments/nebula_v3/train.py",
    "experiments/pair_policy/build_replay.py",
    "experiments/pair_policy/evaluate_pair.py",
    "experiments/pair_policy/export_pair_onnx.py",
    "experiments/pair_policy/probe.py",
    "experiments/pair_policy/run_loop.py",
    "experiments/pair_policy/train_joint.py",
)


@pytest.mark.parametrize("relative_path", ENTRYPOINTS)
def test_python_entrypoint_help(relative_path):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(ROOT / relative_path), "--help"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
