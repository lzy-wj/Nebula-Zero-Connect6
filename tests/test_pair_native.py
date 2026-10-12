import csv
import copy
import ctypes
import json
import subprocess
from collections import Counter
from dataclasses import asdict

import pytest
import torch
import numpy as np

from experiments.pair_native.game_audit import audit_paths, coordinate_to_index
from experiments.pair_native.arena import paired_standard_error, sample_unique_openings
from experiments.pair_native.async_loop import (
    atomic_json,
    dynamic_color_budgets,
    dynamic_opening_temperature,
    gpu_cpu_affinity,
    next_training_round,
    pause_drain_complete,
    promotion_decision,
    quality_retry_opening_control,
    quality_retry_sampling_control,
    parse_simulation_schedule,
    scheduled_simulations,
    select_replay_generations,
    stalled_promotion_candidate,
    validate_gpu_layout,
    validate_replay,
)
from experiments.pair_native.legacy_provenance import (
    default_manifest_path,
    validate_manifest,
    write_manifest,
)
from experiments.pair_native.katagomo_arena import (
    build_override_config,
    from_gtp_vertex,
    to_gtp_vertex,
    wilson_score_interval,
)
from experiments.pair_native.export_release import export_model, sha256_file
from experiments.pair_native.model import (
    PairNativeSparseNet,
    load_checkpoint,
    save_random_checkpoint,
)
from experiments.pair_native.pair_replay import (
    PASS_MOVE,
    PairGameRecord,
    PairTurnRecord,
    audit_shard,
    board_payloads,
    load_and_audit_shard,
    load_verified_games,
    write_shard,
)
from experiments.pair_native.pair_mcts import EVAL_CALLBACK, PairMCTSEngine
from experiments.pair_native.prepare_public_dataset import export_dataset
from experiments.pair_native.selfplay import (
    opening_distribution,
    parse_linux_cpu_list,
    planned_cpu_affinity,
    validate_simulation_budgets,
)
from experiments.pair_native.selfplay_sweep import validate_benchmark_output_root
from experiments.pair_native.replay_load_sweep import data_summary
from experiments.pair_native.train import (
    joint_replay_weights,
    load_data,
    parse_cpu_affinity,
    parse_distribution,
)


BLACK_WIN = (
    "a1",
    "a19", "c19",
    "b1", "c1",
    "e19", "g19",
    "d1", "e1",
    "i19", "k19",
    "f1",
)


def test_replay_load_summary_is_semantic_and_bytes_safe():
    first = (
        {"game": 1},
        {"game": 2},
        [{"board": b"\x00\xff", "nested": {"a": 1, "b": 2}}],
        [{"board": b"\x01"}],
        [{"valid": True, "errors": {"x": 0, "y": 1}}],
    )
    second = (
        {"game": 1},
        {"game": 2},
        [{"nested": {"b": 2, "a": 1}, "board": b"\x00\xff"}],
        [{"board": b"\x01"}],
        [{"errors": {"y": 1, "x": 0}, "valid": True}],
    )

    assert data_summary(first) == data_summary(second)


def test_katagomo_coordinate_adapter_and_fixed_visit_override():
    vertices = {to_gtp_vertex(move) for move in range(19 * 19)}
    assert len(vertices) == 19 * 19
    assert to_gtp_vertex(0) == "A19"
    assert to_gtp_vertex(18) == "T19"
    assert to_gtp_vertex(180) == "K10"
    assert to_gtp_vertex(360) == "T1"
    assert all(from_gtp_vertex(to_gtp_vertex(move)) == move for move in range(361))
    with pytest.raises(ValueError, match="invalid KataGomo"):
        from_gtp_vertex("I10")
    with pytest.raises(ValueError, match="outside"):
        to_gtp_vertex(361)

    override = build_override_config(400, 32, 32, 20, 20260924)
    assert "maxVisits=400" in override
    assert "ponderingEnabled=false" in override
    assert "allowResignation=false" in override
    assert "searchRandSeed=nebula-20260924" in override
    assert wilson_score_interval(0.0, 32) == pytest.approx(
        [0.0, 0.1071827150573635]
    )
    assert wilson_score_interval(1.0, 32) == pytest.approx(
        [0.8928172849426366, 1.0]
    )


def test_async_resource_mapping_and_paired_promotion_statistics(tmp_path):
    assert parse_cpu_affinity("0-2,5,2") == [0, 1, 2, 5]
    assert gpu_cpu_affinity(0) == "0-23"
    assert gpu_cpu_affinity(3) == "72-95"
    assert gpu_cpu_affinity(4) == "96-119"
    assert gpu_cpu_affinity(7) == "168-191"
    assert parse_linux_cpu_list("0-2,5,192-194") == [0, 1, 2, 5, 192, 193, 194]
    assert planned_cpu_affinity(0, 0) == list(range(24))
    assert planned_cpu_affinity(
        0,
        0,
        include_smt=True,
        sibling_resolver=lambda cpu: [cpu, cpu + 192],
    ) == list(range(24)) + list(range(192, 216))
    safe_benchmark = tmp_path / "benchmarks" / "sweep"
    validate_benchmark_output_root(safe_benchmark)
    async_root = tmp_path / "async"
    async_root.mkdir()
    (async_root / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="async-loop root"):
        validate_benchmark_output_root(async_root / "replay-independent-name")
    with pytest.raises(ValueError, match="replay"):
        validate_benchmark_output_root(tmp_path / "standalone" / "replay" / "sweep")
    assert paired_standard_error([0.0, 1.0]) == 0.5
    openings = sample_unique_openings(np.random.default_rng(2026), 64, 4)
    assert len(openings) == len(set(openings)) == 64
    assert all(5 <= divmod(opening, 19)[0] <= 13 for opening in openings)
    assert all(5 <= divmod(opening, 19)[1] <= 13 for opening in openings)
    with pytest.raises(ValueError):
        sample_unique_openings(np.random.default_rng(2026), 82, 4)

    accepted = promotion_decision(
        {"trained_score": 0.60, "trained_score_standard_error": 0.04},
        z_score=1.96,
        margin=0.0,
    )
    rejected = promotion_decision(
        {"trained_score": 0.55, "trained_score_standard_error": 0.04},
        z_score=1.96,
        margin=0.0,
    )
    assert accepted["promote"]
    assert not rejected["promote"]
    advantage = promotion_decision(
        {"trained_score": 0.51, "trained_score_standard_error": 0.04},
        z_score=1.645,
        margin=0.0,
        policy="advantage",
    )
    assert advantage["promote"]
    assert not advantage["confidence_pass"]
    history = [
        {
            "through_round": index,
            "score": score,
            "lower_bound": score - 0.05,
            "promote": False,
            "candidate_checkpoint": f"candidate-{index}",
            "candidate_sha256": f"sha-{index}",
        }
        for index, score in enumerate((0.47, 0.51, 0.49))
    ]
    current = {
        "through_round": 3,
        "score": 0.50,
        "lower_bound": 0.45,
        "promote": False,
        "candidate_checkpoint": "candidate-3",
        "candidate_sha256": "sha-3",
    }
    stalled = stalled_promotion_candidate(history, current, 4, 0.48)
    assert stalled is history[1]
    assert stalled_promotion_candidate(history, current, 5, 0.48) is None
    assert stalled_promotion_candidate(history, current, 4, 0.52) is None
    assert next_training_round({0, 1}, -1) == 0
    assert next_training_round({0, 1}, 0) == 1
    assert next_training_round({1}, -1) is None

    validate_gpu_layout(list(range(7)), 7, 7)
    validate_gpu_layout(list(range(6)), 6, 7)
    with pytest.raises(ValueError, match="unique"):
        validate_gpu_layout([0, 0], 6, 7)
    with pytest.raises(ValueError, match="cannot also be actor"):
        validate_gpu_layout(list(range(7)), 6, 7)
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        validate_gpu_layout([0, 8], 7, 7)
    validate_gpu_layout([7], 7, 7, single_gpu_serial=True)
    with pytest.raises(ValueError, match="one shared"):
        validate_gpu_layout([6, 7], 7, 7, single_gpu_serial=True)
    with pytest.raises(ValueError, match="one shared"):
        validate_gpu_layout([7], 6, 7, single_gpu_serial=True)
    validate_gpu_layout([6, 7], 7, 7, phase_serial=True)
    with pytest.raises(ValueError, match="share an actor"):
        validate_gpu_layout([5, 6], 7, 7, phase_serial=True)
    with pytest.raises(ValueError, match="only one"):
        validate_gpu_layout([7], 7, 7, single_gpu_serial=True, phase_serial=True)

    assert pause_drain_complete(True, False, False, False, None)
    assert not pause_drain_complete(True, True, False, False, None)
    assert not pause_drain_complete(True, False, False, False, 12)
    assert not pause_drain_complete(False, False, False, False, None)


def test_dynamic_color_budget_boosts_only_the_underperforming_side():
    black_heavy = dynamic_color_budgets(
        400,
        [{"winner_counts": {"black": 60, "white": 40, "draw": 3}}],
    )
    assert black_heavy["black_simulations"] == 400
    assert black_heavy["white_simulations"] > 400
    assert black_heavy["boosted_color"] == "white"

    white_heavy = dynamic_color_budgets(
        400,
        [{"winner_counts": {"black": 40, "white": 60, "draw": 3}}],
    )
    assert white_heavy["black_simulations"] > 400
    assert white_heavy["white_simulations"] == 400
    assert white_heavy["boosted_color"] == "black"

    balanced = dynamic_color_budgets(
        400,
        [{"winner_counts": {"black": 54, "white": 46, "draw": 10}}],
    )
    assert balanced["black_simulations"] == balanced["white_simulations"] == 400
    assert balanced["boosted_color"] == "none"

    capped = dynamic_color_budgets(
        400,
        [{"winner_counts": {"black": 100, "white": 0, "draw": 0}}],
        maximum_multiplier=10.0,
        maximum_simulations=1200,
    )
    assert capped["black_simulations"] == 400
    assert capped["white_simulations"] == 1200
    assert capped["budget_multiplier"] == 3.0

    validate_simulation_budgets(400, 400, 1200)
    with pytest.raises(ValueError, match=r"\[400, 1200\]"):
        validate_simulation_budgets(399, 400, 400)
    with pytest.raises(ValueError, match=r"\[400, 1200\]"):
        validate_simulation_budgets(400, 400, 1201)
    validate_simulation_budgets(64, 64, 96, allow_nonproduction=True)


def test_weighted_simulation_schedule_is_exact_and_reproducible():
    schedule = parse_simulation_schedule("400:75,600:15,800:8,1200:2")
    first = [scheduled_simulations(index, schedule, 2026, 400) for index in range(100)]
    second = [scheduled_simulations(index, schedule, 2026, 400) for index in range(100)]
    assert first == second
    assert Counter(first) == {400: 75, 600: 15, 800: 8, 1200: 2}
    assert sum(first) / len(first) == pytest.approx(478.0)
    assert scheduled_simulations(12, [], 2026, 400) == 400
    with pytest.raises(ValueError, match="unique"):
        parse_simulation_schedule("400:1,400:2")


def test_replay_pool_selection_keeps_recent_and_spreads_history():
    available = [{"round_index": index} for index in range(50)]
    selected = select_replay_generations(
        available,
        selected_window=12,
        pool_window=40,
        recent_window=8,
    )
    rounds = [item["round_index"] for item in selected]
    assert rounds[-8:] == list(range(42, 50))
    assert len(rounds) == len(set(rounds)) == 12
    assert rounds[:4] == [10, 20, 31, 41]
    assert [
        item["round_index"]
        for item in select_replay_generations(available, selected_window=3)
    ] == [47, 48, 49]


def test_dynamic_opening_temperature_controls_color_and_concentration():
    collapsed = {
        "winner_counts": {"black": 60, "white": 40, "draw": 2},
        "opening": {"counts": {"182": 93, "178": 7}},
    }
    result = dynamic_opening_temperature(
        0.35,
        [collapsed],
        color_gain=32.0,
        maximum_temperature=1.5,
        maximum_top_share=0.40,
        diversity_gain=2.0,
    )
    assert result["opening_temperature"] == 1.5
    assert result["observed_opening_top_move"] == 182
    assert result["observed_opening_top_share"] == pytest.approx(0.93)
    assert set(result["reasons"]) == {
        "black_overperforming",
        "opening_concentrated",
    }

    black_weak = dynamic_opening_temperature(
        0.35,
        [{"winner_counts": {"black": 40, "white": 60, "draw": 0}}],
        minimum_temperature=0.1,
    )
    assert black_weak["opening_temperature"] == 0.1
    assert black_weak["reasons"] == ["black_underperforming"]

    balanced = dynamic_opening_temperature(
        0.35,
        [{
            "winner_counts": {"black": 52, "white": 48, "draw": 0},
            "opening": {"counts": {"100": 25, "108": 25, "140": 25, "181": 25}},
        }],
    )
    assert balanced["opening_temperature"] == 0.35
    assert balanced["reasons"] == ["base"]

    # Different top moves in each generation must not hide a per-generation
    # collapse when the controller aggregates its rolling window.
    rotating_collapse = dynamic_opening_temperature(
        0.35,
        [
            {"opening": {"counts": {"100": 100}}},
            {"opening": {"counts": {"101": 100}}},
            {"opening": {"counts": {"102": 100}}},
        ],
        maximum_temperature=1.5,
        maximum_top_share=0.40,
        diversity_gain=2.0,
    )
    assert rotating_collapse["observed_aggregate_opening_top_share"] == pytest.approx(1 / 3)
    assert rotating_collapse["observed_max_round_opening_top_share"] == pytest.approx(1.0)
    assert rotating_collapse["opening_temperature"] == pytest.approx(1.5)
    assert "opening_concentrated" in rotating_collapse["reasons"]

    retry = quality_retry_opening_control(
        {"opening_temperature": 1.25, "base_opening_temperature": 1.25, "reasons": ["base"]},
        retry_index=1,
        minimum_temperature=1.0,
        maximum_temperature=1.5,
        diversity_gain=2.0,
    )
    assert retry["opening_temperature"] == 1.5
    assert retry["quality_retry_index"] == 1
    assert retry["reasons"][-1] == "quality_retry"


def test_quality_retry_sampling_changes_seed_and_caps_exploration():
    first = quality_retry_sampling_control(0.25, 0, 0.15, 0.70, 2026, 447, 10_000_019)
    retry = quality_retry_sampling_control(0.25, 3, 0.15, 0.70, 2026, 447, 10_000_019)
    capped = quality_retry_sampling_control(0.25, 9, 0.15, 0.70, 2026, 447, 10_000_019)
    assert first["candidate_gumbel"] == pytest.approx(0.25)
    assert retry["candidate_gumbel"] == pytest.approx(0.70)
    assert capped["candidate_gumbel"] == pytest.approx(0.70)
    assert retry["seed"] - first["seed"] == 3 * 10_000_019


def test_opening_distribution_reports_entropy_and_top_share():
    result = opening_distribution(Counter({182: 6, 178: 2}), 8)
    assert result["top_move"] == 182
    assert result["top_share"] == 0.75
    assert result["unique"] == 2
    assert result["effective_openings"] == pytest.approx(1.7547653506)


def test_joint_replay_sampler_moves_supported_marginals_toward_targets():
    winners = {}
    lengths = {}
    rows = []

    def add(game_id, winner, stones, copies, stones_before):
        winners[game_id] = winner
        lengths[game_id] = stones
        rows.extend({"game_id": game_id, "stones_before": stones_before} for _ in range(copies))

    add("black-short", 1, 30, 60, 10)
    add("white-short", -1, 35, 5, 20)
    add("black-medium", 1, 60, 20, 40)
    add("white-medium", -1, 70, 10, 50)
    add("black-long", 1, 110, 8, 80)
    add("white-long", -1, 120, 12, 90)
    add("black-very-long", 1, 180, 5, 120)
    add("white-very-long", -1, 180, 5, 120)
    add("draw-very-long", 0, 361, 5, 180)

    length_targets = parse_distribution("0.20,0.35,0.30,0.15", 4, "length")
    winner_targets = parse_distribution("0.475,0.475,0.05", 3, "winner")
    weights, stats = joint_replay_weights(
        rows,
        winners,
        lengths,
        length_targets,
        winner_targets,
        maximum_weight_ratio=4.0,
    )

    source_length_error = sum(abs(a - b) for a, b in zip(stats["length_source"], length_targets))
    sampled_length_error = sum(abs(a - b) for a, b in zip(stats["length_sampled"], length_targets))
    source_winner_error = sum(abs(a - b) for a, b in zip(stats["winner_source"], winner_targets))
    sampled_winner_error = sum(abs(a - b) for a, b in zip(stats["winner_sampled"], winner_targets))
    assert sampled_length_error < source_length_error
    assert sampled_winner_error < source_winner_error
    assert weights.shape == (len(rows),)
    assert stats["effective_sample_size"] > 0
    assert stats["joint_counts"]["very_long:draw"] == 5
    assert stats["maximum_to_minimum_weight_ratio"] <= 4.0 + 1.0e-9


def test_atomic_json_never_leaves_a_temporary_commit(tmp_path):
    path = tmp_path / "state.json"
    atomic_json({"generation": 1}, path)
    atomic_json({"generation": 2}, path)

    assert json.loads(path.read_text()) == {"generation": 2}
    assert not path.with_suffix(".json.tmp").exists()


def test_legacy_engine_manifest_binds_checkpoint_and_engine(tmp_path):
    checkpoint = tmp_path / "legacy.pth"
    engine = tmp_path / "legacy.engine"
    checkpoint.write_bytes(b"checkpoint-v1")
    engine.write_bytes(b"engine-v1")
    manifest = default_manifest_path(engine)
    write_manifest(checkpoint, engine, manifest, metadata={"precision": "fp16"})

    payload = validate_manifest(manifest, checkpoint, engine)
    assert payload["checkpoint_sha256"]
    assert payload["engine_sha256"]

    engine.write_bytes(b"engine-v2")
    with pytest.raises(RuntimeError, match="engine manifest"):
        validate_manifest(manifest, checkpoint, engine)


def policies_for(moves):
    return "|".join(f"{coordinate_to_index(move)}:1.0" for move in moves)


def write_games(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("moves", "winner", "policies", "bonuses"))
        writer.writerows(rows)


def test_game_audit_replays_winner_and_detects_post_terminal_move(tmp_path):
    path = tmp_path / "games.csv"
    extra_moves = BLACK_WIN + ("g1",)
    write_games(path, [
        (",".join(BLACK_WIN), "black", policies_for(BLACK_WIN), ""),
        (",".join(extra_moves), "black", policies_for(extra_moves), ""),
    ])

    report = audit_paths([str(path)])

    assert report["games"] == 2
    assert report["valid_games"] == 1
    assert report["terminal_singleton_turns"] == 2
    assert report["errors"]["moves_after_terminal"] == 1
    assert report["native_pair_visit_targets"] == 0


def test_game_audit_rejects_policy_mass_on_occupied_point(tmp_path):
    path = tmp_path / "games.csv"
    policies = policies_for(BLACK_WIN).split("|")
    policies[1] = f"{coordinate_to_index(BLACK_WIN[0])}:1.0"
    write_games(path, [
        (",".join(BLACK_WIN), "black", "|".join(policies), ""),
    ])

    report = audit_paths([str(path)])

    assert report["valid_games"] == 1
    assert report["fully_labeled_valid_games"] == 0
    assert report["errors"]["policy_mass_on_occupied"] == 1
    assert report["errors"]["chosen_move_missing_from_policy"] == 1


def test_pair_native_model_returns_legal_canonical_pairs():
    model = PairNativeSparseNet(
        feature_dim=48,
        max_tokens=16,
        depth=1,
        num_heads=4,
        pair_rank=8,
        latent_tokens=2,
    )
    inputs = torch.zeros((2, 5, 19, 19))
    inputs[:, 2].fill_(1.0)
    inputs[:, 0, 0, 0] = 1.0
    inputs[:, 2, 0, 0] = 0.0

    first, second, scores, value, unary = model.forward_topk(inputs, top_k=16)

    assert first.shape == second.shape == scores.shape == (2, 16)
    assert value.shape == (2,)
    assert unary.shape == (2, 361)
    assert torch.all(first < second)
    assert not torch.any(first == 0)
    assert not torch.any(second == 0)
    assert torch.isfinite(scores).all()


def make_native_records():
    moves = [coordinate_to_index(move) for move in BLACK_WIN]
    game = PairGameRecord(
        game_id="game-2026",
        generation=0,
        seed=2026,
        model_sha256="1" * 64,
        engine_sha256="2" * 64,
        search_config_sha256="3" * 64,
        winner=1,
        termination="six",
        moves=moves,
    )
    board = np.zeros((19, 19), dtype=np.int8)
    turns = []
    turn_index = 0
    move_index = 0
    while move_index < len(moves):
        player = 1 if move_index == 0 else (-1 if ((move_index + 1) // 2) % 2 else 1)
        first = moves[move_index]
        if move_index == 0 or move_index + 1 >= len(moves):
            second = PASS_MOVE
            consumed = 1
        else:
            first, second = sorted((first, moves[move_index + 1]))
            consumed = 2
        black, white, digest = board_payloads(board)
        turns.append(PairTurnRecord(
            game_id=game.game_id,
            turn_index=turn_index,
            player=player,
            stones_before=move_index,
            board_black=black,
            board_white=white,
            state_hash=digest,
            chosen_a=first,
            chosen_b=second,
            policy_a=[first],
            policy_b=[second],
            policy_probability=[1.0],
            root_value=0.0,
            simulations=400,
            unique_nodes=48,
            neural_evaluations=32,
            search_ms=1.0,
        ))
        for offset in range(consumed):
            board.flat[moves[move_index + offset]] = player
        move_index += consumed
        turn_index += 1
    return game, turns


def test_public_model_export_is_loadable_and_preserves_weights(tmp_path):
    source = save_random_checkpoint(
        tmp_path / "source.pth",
        config={
            "feature_dim": 24,
            "max_tokens": 8,
            "depth": 1,
            "num_heads": 4,
            "pair_rank": 4,
            "latent_tokens": 2,
            "dense_channels": 0,
        },
    )
    output = tmp_path / "model"
    source_sha256 = sha256_file(source)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"promotions": [{
        "through_round": 3,
        "promote": True,
        "score": 0.625,
        "standard_error": 0.04,
        "lower_bound": 0.55,
        "required_lower_bound": 0.5,
        "promotion_policy": "advantage",
        "promotion_reason": "advantage",
        "baseline_sha256": "0" * 64,
        "candidate_sha256": source_sha256,
        "gate_config": {"pairs": 8, "simulations": 400},
    }]}), encoding="utf-8")

    manifest = export_model(
        source,
        output,
        state_path=state_path,
        through_round=3,
    )
    source_model, _ = load_checkpoint(source)
    exported_model, exported = load_checkpoint(output / "pair_native_v3_inference.pth")

    assert manifest["source"]["checkpoint_sha256"] == source_sha256
    assert manifest["promotion"]["through_round"] == 3
    assert "optimizer_state_dict" not in exported
    assert exported["release"]["name"] == "pair-native-v3"
    assert source_model.state_dict().keys() == exported_model.state_dict().keys()
    for name, tensor in source_model.state_dict().items():
        assert torch.equal(tensor, exported_model.state_dict()[name])
    assert (output / "README.md").is_file()
    assert (output / "SHA256SUMS").is_file()


def test_public_dataset_export_uses_exact_promoted_replay_and_removes_paths(tmp_path):
    replay = tmp_path / "source" / "gen-0007-attempt-00"
    replay.mkdir(parents=True)
    game, turns = make_native_records()
    game.generation = 7
    write_shard(
        replay,
        "gen-0007-worker-00",
        [game],
        turns,
        {
            "checkpoint": "/private/host/checkpoint.pth",
            "affinity": [0, 1],
            "model_config": {"feature_dim": 24},
            "search": {"simulations": 400},
            "execution": {"affinity": [0, 1], "concurrent_games": 1},
            "worker": 0,
            "gpu": 7,
        },
    )
    state = {
        "promotions": [{
            "through_round": 3,
            "promote": True,
            "candidate_sha256": "4" * 64,
            "replay_directories": [str(replay)],
        }]
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    output = tmp_path / "dataset"

    manifest = export_dataset(state_path, output, through_round=3)

    assert manifest["stats"]["games"] == 1
    assert manifest["stats"]["turns"] == len(turns)
    assert manifest["stats"]["winner_counts"] == {
        "black": 1,
        "white": 0,
        "draw": 0,
    }
    exported_manifest = next((output / "data").rglob("*.manifest.json"))
    payload = exported_manifest.read_text(encoding="utf-8")
    assert "/private/host" not in payload
    assert '"affinity"' not in payload
    assert audit_shard(exported_manifest)["valid"]
    assert (output / "dataset_manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()


def test_pair_replay_shard_round_trip_and_strict_audit(tmp_path):
    game, turns = make_native_records()
    manifest = write_shard(tmp_path, "shard-0000", [game], turns, {"source": "test"})

    report = audit_shard(manifest)

    assert report["valid"]
    assert report["games"] == 1
    assert report["turns"] == len(turns)
    assert report["valid_turns"] == len(turns)

    loaded_report, payload, games, loaded_turns = load_and_audit_shard(manifest)
    assert loaded_report == report
    assert payload["games"] == len(games) == 1
    assert loaded_turns == [asdict(turn) | {"schema_version": 1} for turn in turns]
    verified_payload, verified_games = load_verified_games(manifest)
    assert verified_payload == payload
    assert verified_games == games


def test_single_pass_train_load_is_semantically_identical(tmp_path):
    first_game, first_turns = make_native_records()
    second_game = copy.deepcopy(first_game)
    second_game.game_id = "game-2027"
    second_game.seed = 2027
    second_turns = copy.deepcopy(first_turns)
    for turn in second_turns:
        turn.game_id = second_game.game_id
    write_shard(
        tmp_path,
        "train-load",
        [first_game, second_game],
        first_turns + second_turns,
    )

    legacy = load_data(tmp_path, 0.1, single_pass_audit_load=False)
    single_pass = load_data(tmp_path, 0.1, single_pass_audit_load=True)
    parallel = load_data(tmp_path, 0.1, replay_audit_workers=2)

    assert single_pass == legacy
    assert parallel == legacy


def test_async_replay_validation_recomputes_summary_distributions(tmp_path):
    game, turns = make_native_records()
    for turn in turns:
        turn.simulations = 400 if turn.player == 1 else 600
    write_shard(
        tmp_path,
        "gen-0000-worker-00",
        [game],
        turns,
        {
            "search": {
                "simulations": 400,
                "black_simulations": 400,
                "white_simulations": 600,
                "opening_temperature": 1.25,
            }
        },
    )
    summary_path = tmp_path / "gen-0000-summary.json"
    summary = {
        "games": 1,
        "turns": len(turns),
        "stones": len(game.moves),
        "winner_counts": {"black": 1, "white": 0, "draw": 0},
        "length_buckets": {"short": 1, "medium": 0, "long": 0, "very_long": 0},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    validated = validate_replay(tmp_path, 0, 1, "1" * 64, 400, 600)
    assert validated["winner_counts"] == summary["winner_counts"]
    assert validated["length_buckets"] == summary["length_buckets"]
    assert validated["opening"]["top_share"] == 1.0
    assert validated["diversity"]["unique_complete_game_rate"] == 1.0
    with pytest.raises(RuntimeError, match="white simulation budget"):
        validate_replay(tmp_path, 0, 1, "1" * 64, 400, 601)
    with pytest.raises(RuntimeError, match="opening temperature"):
        validate_replay(
            tmp_path,
            0,
            1,
            "1" * 64,
            400,
            600,
            expected_opening_temperature=1.0,
        )
    with pytest.raises(RuntimeError, match="opening collapse"):
        validate_replay(
            tmp_path,
            0,
            1,
            "1" * 64,
            400,
            600,
            expected_opening_temperature=1.25,
            maximum_opening_top_share=0.9,
        )

    summary["winner_counts"] = {"black": 0, "white": 1, "draw": 0}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="winner_counts"):
        validate_replay(tmp_path, 0, 1, "1" * 64, 400, 600)


def test_async_parallel_replay_audit_matches_serial(tmp_path):
    first_game, first_turns = make_native_records()
    second_game = copy.deepcopy(first_game)
    second_game.game_id = "game-2027"
    second_game.seed = 2027
    second_turns = copy.deepcopy(first_turns)
    for turn in second_turns:
        turn.game_id = second_game.game_id
    for name, game, turns in (
        ("gen-0000-worker-00", first_game, first_turns),
        ("gen-0000-worker-01", second_game, second_turns),
    ):
        write_shard(
                tmp_path,
                name,
                [game],
                turns,
                {
                    "search": {
                        "simulations": 400,
                        "black_simulations": 400,
                        "white_simulations": 400,
                    }
                },
            )
    summary = {
        "games": 2,
        "turns": len(first_turns) + len(second_turns),
        "stones": len(first_game.moves) + len(second_game.moves),
        "winner_counts": {"black": 2, "white": 0, "draw": 0},
        "length_buckets": {"short": 2, "medium": 0, "long": 0, "very_long": 0},
    }
    (tmp_path / "gen-0000-summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    serial = validate_replay(tmp_path, 0, 2, "1" * 64, 400, 400)
    parallel = validate_replay(
        tmp_path,
        0,
        2,
        "1" * 64,
        400,
        400,
        audit_workers=2,
    )

    assert parallel == serial


def test_pair_replay_audit_rejects_manifest_count_mismatch(tmp_path):
    game, turns = make_native_records()
    manifest = write_shard(tmp_path, "wrong-count", [game], turns)
    payload = json.loads(manifest.read_text())
    payload["games"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_shard(manifest)

    assert not report["valid"]
    assert report["errors"]["manifest_game_count_mismatch"] == 1


def test_pair_replay_audit_rejects_policy_without_chosen_action(tmp_path):
    game, turns = make_native_records()
    turns[1].policy_a = [100]
    turns[1].policy_b = [101]
    manifest = write_shard(tmp_path, "bad-shard", [game], turns)

    report = audit_shard(manifest)

    assert not report["valid"]
    assert report["errors"]["chosen_action_missing_from_policy"] == 1


def test_pair_replay_audit_rejects_wrong_declared_simulation_budget(tmp_path):
    game, turns = make_native_records()
    turns[1].simulations = 1
    manifest = write_shard(
        tmp_path,
        "wrong-budget",
        [game],
        turns,
        {"search": {"simulations": 400}},
    )

    report = audit_shard(manifest)

    assert not report["valid"]
    assert report["errors"]["simulation_budget_mismatch"] == 1


def test_pair_replay_audits_adaptive_simulation_stages(tmp_path):
    game, turns = make_native_records()
    for index, turn in enumerate(turns):
        allowed = (400, 800) if turn.player == 1 else (400, 800, 1200)
        turn.simulations = allowed[index % len(allowed)]
    metadata = {
        "search": {
            "simulations": 800,
            "black_simulations": 800,
            "white_simulations": 1200,
            "adaptive_simulations": True,
            "adaptive_min_simulations": 400,
            "adaptive_mid_simulations": 800,
        }
    }
    manifest = write_shard(tmp_path, "adaptive-budget", [game], turns, metadata)
    assert audit_shard(manifest)["valid"]

    bad_turns = copy.deepcopy(turns)
    bad_turns[1].simulations = 600
    bad_manifest = write_shard(
        tmp_path,
        "adaptive-budget-invalid",
        [game],
        bad_turns,
        metadata,
    )
    report = audit_shard(bad_manifest)
    assert not report["valid"]
    assert report["errors"]["simulation_budget_mismatch"] == 1


def test_pair_replay_audits_color_specific_simulation_budgets(tmp_path):
    game, turns = make_native_records()
    for turn in turns:
        turn.simulations = 400 if turn.player == 1 else 600
    manifest = write_shard(
        tmp_path,
        "color-budgets",
        [game],
        turns,
        {
            "search": {
                "simulations": 400,
                "black_simulations": 400,
                "white_simulations": 600,
            }
        },
    )
    assert audit_shard(manifest)["valid"]

    turns[1].simulations = 400
    manifest = write_shard(
        tmp_path,
        "bad-color-budgets",
        [game],
        turns,
        {
            "search": {
                "simulations": 400,
                "black_simulations": 400,
                "white_simulations": 600,
            }
        },
    )
    report = audit_shard(manifest)
    assert not report["valid"]
    assert report["errors"]["simulation_budget_mismatch"] == 1


def test_pair_replay_audits_explicit_uniform_forced_opening(tmp_path):
    game, turns = make_native_records()
    opening = turns[0]
    opening.policy_a = list(range(361))
    opening.policy_b = [PASS_MOVE] * 361
    opening.policy_probability = [1.0 / 361] * 361
    opening.simulations = 0
    opening.unique_nodes = 0
    opening.neural_evaluations = 0
    opening.search_ms = 0.0
    manifest = write_shard(
        tmp_path,
        "forced-opening",
        [game],
        turns,
        {"search": {"forced_opening_radius": 9}},
    )

    report = audit_shard(manifest)

    assert report["valid"]


class StaticPairEvaluator:
    def __init__(self, top_k=16):
        self.top_k = top_k
        self.last_error = None
        self.callback = EVAL_CALLBACK(self._evaluate)

    def _evaluate(
        self,
        batch,
        boards_pointer,
        _players_pointer,
        first_pointer,
        second_pointer,
        prior_pointer,
        value_pointer,
    ):
        boards = np.ctypeslib.as_array(
            boards_pointer,
            shape=(batch * 361,),
        ).reshape(batch, 361)
        first = np.ctypeslib.as_array(
            first_pointer,
            shape=(batch * self.top_k,),
        ).reshape(batch, self.top_k)
        second = np.ctypeslib.as_array(
            second_pointer,
            shape=(batch * self.top_k,),
        ).reshape(batch, self.top_k)
        priors = np.ctypeslib.as_array(
            prior_pointer,
            shape=(batch * self.top_k,),
        ).reshape(batch, self.top_k)
        values = np.ctypeslib.as_array(value_pointer, shape=(batch,))
        for batch_index, board in enumerate(boards):
            legal = np.flatnonzero(board == 0)
            actions = []
            if np.count_nonzero(board) == 0:
                actions = [(int(move), -1) for move in legal[: self.top_k]]
            else:
                for left_index, left in enumerate(legal):
                    for right in legal[left_index + 1 :]:
                        actions.append((int(left), int(right)))
                        if len(actions) == self.top_k:
                            break
                    if len(actions) == self.top_k:
                        break
            for action_index, (left, right) in enumerate(actions):
                first[batch_index, action_index] = left
                second[batch_index, action_index] = right
            priors[batch_index].fill(1.0 / self.top_k)
            values[batch_index] = 0.0
        return 0


def test_pair_mcts_batches_independent_turn_trees(tmp_path):
    library_path = tmp_path / "libpair_mcts.so"
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O2",
            "-fopenmp",
            "experiments/pair_native/pair_mcts.cpp",
            "-o",
            str(library_path),
        ],
        check=True,
    )
    evaluator = StaticPairEvaluator(top_k=16)
    engine = PairMCTSEngine(
        evaluator,
        library_path=library_path,
        top_k=16,
        root_noise_epsilon=0.0,
    )
    contexts = [engine.create_context(2026 + index) for index in range(8)]
    try:
        with pytest.raises(RuntimeError):
            engine.run([contexts[0], contexts[0]], 1)
        opening_budgets = [8] * 4 + [16] * 4
        engine.run(contexts, opening_budgets)
        for context, budget in zip(contexts, opening_budgets):
            assert context.stats()["simulations"] == budget
            first, second = context.choose_action(temperature=0.0)
            assert 0 <= first < 361
            assert second == -1
            context.play(first, second)
            assert context.total_stones == 1
            assert context.current_player == -1

        engine.run(contexts, 16)
        for context in contexts:
            first, second = context.choose_action(temperature=0.0)
            assert 0 <= first < second < 361
            context.play(first, second)
            assert context.total_stones == 3
            assert context.current_player == 1
            assert context.stats()["simulations"] == 16
        stats = engine.global_stats()
        assert stats["neural_evaluations"] > 0
        assert stats["average_batch"] >= 7.0
        engine.run([contexts[0]], 400)
        assert contexts[0].stats()["simulations"] == 400
    finally:
        for context in contexts:
            context.close()


def test_pair_mcts_finds_exact_two_stone_and_single_stone_wins(tmp_path):
    library_path = tmp_path / "libpair_mcts.so"
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O2",
            "-fopenmp",
            "experiments/pair_native/pair_mcts.cpp",
            "-o",
            str(library_path),
        ],
        check=True,
    )
    engine = PairMCTSEngine(
        StaticPairEvaluator(top_k=16),
        library_path=library_path,
        top_k=16,
        root_noise_epsilon=0.0,
        cpu_threads=4,
    )

    pair_context = engine.create_context(1)
    single_context = engine.create_context(2)
    try:
        prefix_for_four = [(0, -1), (300, 301), (1, 2), (302, 303), (3, 100), (304, 305)]
        for action in prefix_for_four:
            pair_context.play(*action)
        engine.run([pair_context], 1)
        assert pair_context.choose_action() == (4, 5)

        prefix_for_five = prefix_for_four + [(4, 101), (306, 307)]
        for action in prefix_for_five:
            single_context.play(*action)
        engine.run([single_context], 1)
        assert single_context.choose_action() == (5, -1)
        stats = engine.global_stats()
        assert stats["simulations"] == 2
        assert stats["tactical_expansions"] == 2
    finally:
        pair_context.close()
        single_context.close()


def test_pair_mcts_openmp_preserves_deterministic_trace(tmp_path):
    library_path = tmp_path / "libpair_mcts.so"
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O2",
            "-fopenmp",
            "experiments/pair_native/pair_mcts.cpp",
            "-o",
            str(library_path),
        ],
        check=True,
    )

    def play_trace(cpu_threads):
        engine = PairMCTSEngine(
            StaticPairEvaluator(top_k=16),
            library_path=library_path,
            top_k=16,
            root_noise_epsilon=0.0,
            cpu_threads=cpu_threads,
        )
        contexts = [engine.create_context(9000 + index) for index in range(8)]
        trace = []
        try:
            for _ in range(4):
                engine.run(contexts, 16)
                actions = [context.choose_action(temperature=0.0) for context in contexts]
                trace.append(actions)
                for context, action in zip(contexts, actions):
                    context.play(*action)
        finally:
            for context in contexts:
                context.close()
        return trace

    assert play_trace(1) == play_trace(4)
