import os
import sys
import json
import importlib

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.nebula_v4.model import NebulaV4, active_parameter_count, parameter_count
from experiments.nebula_v4.launch_compare import discover_generations
from experiments.nebula_v4.build_mmap_cache import (
    create_arrays,
    manifest_policy_width,
    write_rows,
)
from experiments.nebula_v4.mmap_dataset import MMapTurnDataset
from experiments.nebula_v4.train_cpu import hierarchical_auxiliary_loss
from experiments.pair_native.pair_replay import pack_points
from experiments.pair_native.pair_mcts import HierarchicalPairInference
from experiments.pair_native.model import load_checkpoint
from experiments.pair_native.train import PairTurnDataset


def empty_features(batch: int):
    features = torch.zeros(batch, 5, 19, 19)
    features[:, 2].fill_(1.0)
    return features


def test_v4_model_tools_import_without_selfplay_extensions():
    for module in (
        "experiments.nebula_v4.profile_model_stages",
        "experiments.nebula_v4.evaluate_hierarchical_recall",
    ):
        assert importlib.import_module(module)


def test_dense_v4_forward_backward_and_legal_pairs():
    torch.manual_seed(7)
    model = NebulaV4(
        variant="dense",
        stem_channels=8,
        feature_dim=32,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    features = empty_features(2)
    features[:, 0, 0, 0] = 1.0
    features[:, 2, 0, 0] = 0.0
    unary, embeddings, value, occupied = model(features)
    scores = model.pair_scores(unary, embeddings, occupied)
    first, second, top_scores, _, _ = model.forward_topk(features, top_k=8)
    loss = top_scores.mean() + value.square().mean()
    loss.backward()

    assert unary.shape == (2, 361)
    assert embeddings.shape == (2, 361, 16)
    assert scores.shape == (2, 361, 361)
    assert torch.all(first < second)
    assert not torch.any(first.eq(0) | second.eq(0))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_moe_routes_each_complete_position_once():
    torch.manual_seed(11)
    model = NebulaV4(
        variant="moe",
        stem_channels=8,
        feature_dim=32,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
        num_experts=4,
        expert_expansion=2,
        shared_expansion=1,
    )
    outputs = model(empty_features(7), return_aux=True)
    unary, embeddings, value, occupied, auxiliary = outputs
    scores = model.pair_scores(unary, embeddings, occupied)
    loss = (
        scores[:, 0, 1].mean()
        + value.square().mean()
        + 0.01 * auxiliary["balance_loss"]
    )
    loss.backward()

    assert auxiliary["assignments"].shape == (7,)
    assert int(auxiliary["load"].mul(7).round().sum()) == 7
    assert torch.isfinite(auxiliary["balance_loss"])
    assert parameter_count(model) > active_parameter_count(model)
    assert model.moe.router.weight.grad is not None
    assert torch.isfinite(model.moe.router.weight.grad).all()


def test_pair_interaction_is_symmetric_before_canonical_mask():
    model = NebulaV4(
        feature_dim=32,
        stem_channels=8,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    unary, embeddings, _, occupied = model(empty_features(1))
    left, right = embeddings.split(model.pair_rank, dim=-1)
    raw = torch.matmul(left, right.transpose(1, 2))
    interaction = raw + raw.transpose(1, 2)
    assert torch.allclose(interaction, interaction.transpose(1, 2))
    scores = model.pair_scores(unary, embeddings, occupied)
    assert torch.isfinite(scores[0].triu(diagonal=1)).any()


def test_pair_scores_preserve_inference_dtype():
    model = NebulaV4(
        feature_dim=32,
        stem_channels=8,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    unary = torch.zeros(1, 361, dtype=torch.float16)
    embeddings = torch.zeros(1, 361, 16, dtype=torch.float16)
    occupied = torch.zeros(1, 361, dtype=torch.bool)
    assert model.pair_scores(unary, embeddings, occupied).dtype == torch.float16


def test_pair_row_scores_match_canonical_full_matrix():
    torch.manual_seed(19)
    model = NebulaV4(
        feature_dim=32,
        stem_channels=8,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    features = empty_features(2)
    features[0, 0, 0, 0] = 1.0
    features[0, 2, 0, 0] = 0.0
    features[1, 1, 18, 18] = 1.0
    features[1, 2, 18, 18] = 0.0
    unary, embeddings, _, occupied = model(features)
    full = model.pair_scores(unary, embeddings, occupied)
    first_moves = torch.tensor([[1, 100, 300], [0, 180, 359]])
    rows = model.pair_row_scores(unary, embeddings, occupied, first_moves)

    seconds = torch.arange(361)
    canonical_first = torch.minimum(first_moves[..., None], seconds)
    canonical_second = torch.maximum(first_moves[..., None], seconds)
    batch = torch.arange(2)[:, None, None].expand_as(canonical_first)
    expected = full[batch, canonical_first, canonical_second]
    torch.testing.assert_close(rows, expected)


def test_hierarchical_pair_inference_returns_unique_legal_pairs():
    torch.manual_seed(23)
    model = NebulaV4(
        feature_dim=32,
        stem_channels=8,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    features = empty_features(2)
    features[0, 0, 0, 0] = 1.0
    features[0, 2, 0, 0] = 0.0
    features[1, 1, 18, 18] = 1.0
    features[1, 2, 18, 18] = 0.0
    inference = HierarchicalPairInference(model, 8, 4)
    first, second, priors, values = inference(features)

    assert first.shape == second.shape == priors.shape == (2, 32)
    assert values.shape == (2,)
    assert torch.all(first < second)
    assert not torch.any(first[0].eq(0) | second[0].eq(0))
    assert not torch.any(first[1].eq(360) | second[1].eq(360))
    torch.testing.assert_close(priors.sum(dim=1), torch.ones(2))
    for row in range(2):
        pairs = set(zip(first[row].tolist(), second[row].tolist()))
        assert len(pairs) == 32

    noisy = HierarchicalPairInference(model, 8, 4, candidate_gumbel=0.25)
    noisy_outputs = noisy(
        features,
        torch.zeros(2, 361),
        torch.zeros(2, 8, 361),
    )
    for actual, expected in zip(noisy_outputs, (first, second, priors, values)):
        torch.testing.assert_close(actual, expected)


def test_common_checkpoint_loader_supports_v4(tmp_path):
    model = NebulaV4(
        feature_dim=32,
        stem_channels=8,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    checkpoint = tmp_path / "v4.pth"
    torch.save({
        "model_config": {
            "variant": "dense",
            "feature_dim": 32,
            "stem_channels": 8,
            "summary_tokens": 8,
            "depth": 1,
            "num_heads": 4,
            "pair_rank": 8,
        },
        "model_state_dict": model.state_dict(),
    }, checkpoint)
    loaded, payload = load_checkpoint(checkpoint)
    assert isinstance(loaded, NebulaV4)
    assert payload["model_config"]["variant"] == "dense"


def test_hierarchical_auxiliary_loss_is_finite_and_differentiable():
    model = NebulaV4(
        feature_dim=32,
        stem_channels=8,
        summary_tokens=8,
        depth=1,
        num_heads=4,
        pair_rank=8,
    )
    unary, embeddings, _, occupied = model(empty_features(2))
    target_a = torch.tensor([[10, 20], [30, 40]])
    target_b = torch.tensor([[11, 21], [-1, -2]])
    probability = torch.tensor([[0.75, 0.25], [1.0, 0.0]])
    total, first, conditional = hierarchical_auxiliary_loss(
        model,
        unary,
        embeddings,
        occupied,
        target_a,
        target_b,
        probability,
    )
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(first)
    assert torch.isfinite(conditional)
    assert model.unary_head[-1].weight.grad is not None
    assert model.pair_head[-1].weight.grad is not None


def test_compare_launcher_discovers_contiguous_generation_directories(tmp_path):
    for generation in (1370, 1371, 1372):
        directory = tmp_path / f"gen-{generation}-attempt-00"
        directory.mkdir()
        (directory / f"gen-{generation}.manifest.json").write_text("{}")
    assert [value for value, _ in discover_generations(tmp_path)] == [1370, 1371, 1372]


def test_mmap_cache_round_trip_matches_original_dataset(tmp_path):
    empty = np.zeros((19, 19), dtype=bool)
    black = empty.copy()
    black[3, 5] = True
    rows = []
    for turn in range(2):
        rows.append(
            {
                "game_id": "game-1",
                "turn_index": turn,
                "player": 1 if turn == 0 else -1,
                "stones_before": turn,
                "board_black": pack_points(black if turn else empty),
                "board_white": pack_points(empty),
                "chosen_a": 10 + turn,
                "chosen_b": 20 + turn,
                "policy_a": [10 + turn, 30 + turn],
                "policy_b": [20 + turn, 40 + turn],
                "policy_probability": [0.75, 0.25],
                "root_value": 0.125,
            }
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"strict_audit": True, "seed": 7, "policy_width": 64})
    )
    arrays = create_arrays(tmp_path / "train", len(rows))
    write_rows(arrays, 0, rows, [{"game_id": "game-1", "winner": 1}])
    for array in arrays.values():
        array.flush()
    original = PairTurnDataset(rows, {"game-1": 1}, training=False)
    cached = MMapTurnDataset(tmp_path, "train", training=False)
    for index in range(len(rows)):
        expected = original[index]
        actual = cached[index]
        for name in expected:
            if torch.is_tensor(expected[name]):
                assert torch.equal(expected[name], actual[name])
            else:
                assert expected[name] == actual[name]


def test_mmap_cache_derives_and_preserves_top96_policy(tmp_path):
    manifest = tmp_path / "part.manifest.json"
    manifest.write_text(json.dumps({"metadata": {"search": {"top_k": 96}}}))
    assert manifest_policy_width([manifest]) == 96

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(json.dumps({"policy_width": 96}))
    empty = np.zeros(361, dtype=np.uint8)
    row = {
        "game_id": "game-96",
        "turn_index": 2,
        "player": 1,
        "stones_before": 4,
        "board_black": pack_points(empty),
        "board_white": pack_points(empty),
        "chosen_a": 1,
        "chosen_b": 2,
        "policy_a": list(range(96)),
        "policy_b": list(range(96, 192)),
        "policy_probability": [1.0 / 96] * 96,
        "root_value": 0.0,
    }
    arrays = create_arrays(cache / "train", 1, policy_width=96)
    write_rows(
        arrays,
        0,
        [row],
        [{"game_id": "game-96", "winner": 1}],
        {"game-96": 9},
    )
    for array in arrays.values():
        array.flush()
    dataset = MMapTurnDataset(cache, "train", training=False)
    assert len(dataset[0]["policy_a"]) == 96
    assert dataset[0]["remaining_turns"] == 7
