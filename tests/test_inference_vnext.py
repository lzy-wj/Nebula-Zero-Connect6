import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.inference_vnext.model import (
    BottleneckGlobalNet,
    DualScaleNet,
    SparseStoneNet,
    build_architecture,
)
from experiments.inference_vnext.wrapper import FusedPairSelfPlayWrapper


def assert_policy_value_model(model):
    inputs = torch.randn(2, 5, 19, 19)
    policy, policy2, value, features = model(inputs, return_features=True)
    loss = policy.square().mean() + value.square().mean()
    loss.backward()

    assert policy2 is None
    assert policy.shape == (2, 361)
    assert value.shape == (2, 1)
    assert features.shape[:2] == (2, 361)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_bottleneck_global_forward_backward():
    assert_policy_value_model(BottleneckGlobalNet(
        channels=48,
        hidden_channels=24,
        depth=2,
        global_every=1,
    ))


def test_dual_scale_forward_backward():
    assert_policy_value_model(DualScaleNet(
        channels=48,
        hidden_channels=24,
        local_depth=3,
        global_depth=1,
        num_heads=3,
    ))


def test_sparse_stone_forward_backward():
    assert_policy_value_model(SparseStoneNet(
        feature_dim=48,
        max_stones=16,
        stone_depth=1,
        num_heads=3,
    ))


def test_sparse_stone_selection_is_deterministic_when_capacity_is_exceeded():
    model = SparseStoneNet(
        feature_dim=48,
        max_stones=16,
        stone_depth=1,
        num_heads=3,
    )
    occupancy = torch.ones((2, 361))

    model = model.half()
    selected = model._select_stone_indices(occupancy.half())

    expected = torch.arange(345, 361)
    assert torch.equal(selected[0].sort().values, expected)
    assert torch.equal(selected[1].sort().values, expected)


def test_registered_architectures_build():
    for name in (
        "bottleneck_c192",
        "dual_scale_c192",
        "sparse_stone_c192_k64",
    ):
        model = build_architecture(name)
        assert model.parameter_count > 0


def test_pair_wrapper_outputs_all_conditional_tensors():
    model = BottleneckGlobalNet(
        channels=48,
        hidden_channels=24,
        depth=2,
        global_every=1,
    )
    wrapper = FusedPairSelfPlayWrapper(
        model,
        compute_dtype=torch.float32,
        pair_rank=8,
    )
    board = torch.zeros((2, 19, 19), dtype=torch.int32)
    outputs = wrapper(board)

    assert [tuple(output.shape) for output in outputs] == [
        (2, 361),
        (2, 1),
        (2, 361, 8),
        (2, 361, 8),
        (2, 361),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
