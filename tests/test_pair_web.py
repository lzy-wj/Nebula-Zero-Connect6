import json
import time
from pathlib import Path

from experiments.pair_native.pair_mcts import PASS_MOVE
from experiments.pair_native.web_app import (
    BLACK,
    WHITE,
    ModelRegistry,
    PairWebGame,
    SearchSettings,
    TrainingMonitor,
)


def empty_registry(tmp_path: Path) -> ModelRegistry:
    training = tmp_path / "training"
    legacy = tmp_path / "legacy"
    training.mkdir()
    legacy.mkdir()
    return ModelRegistry(training, legacy)


def test_web_game_tracks_connect6_turn_boundaries(tmp_path):
    game = PairWebGame(empty_registry(tmp_path))

    assert game.play_human_stone(9, 9) == "continue"
    assert game.current_player == WHITE
    assert game.actions == [(9 * 19 + 9, PASS_MOVE)]
    assert game.snapshot()["stones_remaining"] == 2

    assert game.play_human_stone(8, 8) == "continue"
    assert game.current_player == WHITE
    assert game.pending == [8 * 19 + 8]
    assert game.snapshot()["stones_remaining"] == 1

    assert game.play_human_stone(8, 10) == "continue"
    assert game.current_player == BLACK
    assert not game.pending
    assert game.actions[-1] == tuple(sorted((8 * 19 + 8, 8 * 19 + 10)))
    assert game.snapshot()["stones_remaining"] == 2


def test_web_game_accepts_win_on_first_stone_of_turn(tmp_path):
    game = PairWebGame(empty_registry(tmp_path))
    moves = [
        [0, 0],
        [10, 0], [12, 2],
        [0, 1], [0, 2],
        [10, 4], [12, 6],
        [0, 3], [0, 4],
        [10, 8], [12, 10],
        [0, 5],
    ]
    for row, column in moves:
        result = game.play_human_stone(row, column)

    assert result == "win"
    assert game.game_over
    assert game.winner == BLACK
    assert game.actions[-1] == (5, PASS_MOVE)
    assert game.snapshot()["stones_remaining"] == 0


def test_web_game_undo_removes_complete_turns(tmp_path):
    game = PairWebGame(empty_registry(tmp_path))
    for row, column in ([9, 9], [8, 8], [8, 9], [7, 7], [7, 8]):
        assert game.play_human_stone(row, column) == "continue"

    assert len(game.history) == 5
    assert game.undo_turns(1) == 2
    assert len(game.history) == 3
    assert game.current_player == BLACK
    assert game.undo_turns(2) == 3
    assert not game.history
    assert game.current_player == BLACK


def test_search_settings_are_bounded_and_topk_is_supported():
    settings = SearchSettings.from_payload(
        {
            "simulations": 99_999,
            "temperature": -2,
            "top_k": 70,
            "cpu_threads": 100,
            "batch_size": 999,
            "root_noise": 3,
        }
    )
    assert settings.simulations == 10_000
    assert settings.temperature == 0
    assert settings.top_k == 64
    assert settings.cpu_threads == 24
    assert settings.batch_size == 128
    assert settings.root_noise == 1


def test_model_registry_uses_opaque_ids_and_marks_current(tmp_path):
    training = tmp_path / "training"
    legacy = tmp_path / "legacy"
    candidate = training / "run-a" / "candidates" / "through-0002" / "best.pth"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"checkpoint")
    legacy.mkdir()
    (legacy / "best.engine").write_bytes(b"engine")
    state = {
        "status": "running",
        "updated_unix": 1234,
        "accepted_checkpoint": str(candidate),
        "promotions": [],
    }
    (training / "run-a" / "state.json").write_text(json.dumps(state))

    registry = ModelRegistry(training, legacy)
    records, default_id = registry.records()
    current = next(record for record in records if record.current)

    assert current.id == default_id
    assert "path" not in current.public()
    assert registry.resolve(default_id).path == str(candidate.resolve())
    assert {record.kind for record in records} == {"native_pair", "legacy"}


def test_training_monitor_reduces_rounds_and_flags_color_drift(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "training"
    run = root / "run-a"
    run.mkdir(parents=True)
    replay = run / "replay" / "gen-0010-attempt-00"
    replay.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "moves": pa.array(
                    [[180], [180], [181]],
                    type=pa.list_(pa.int16()),
                )
            }
        ),
        replay / "sample.games.parquet",
    )
    rounds = []
    for index in range(3):
        rounds.append(
            {
                "round_index": index,
                "generation": 10 + index,
                "actor_started_unix": 1000 + index * 100,
                "actor_finished_unix": 1090 + index * 100,
                "directory": str(replay),
                "checkpoint_sha256": "a" * 64,
                "color_control": {
                    "black_simulations": 400,
                    "white_simulations": 520,
                    "boosted_color": "white",
                    "observed_black_decisive_rate": 0.6,
                },
                "summary": {
                    "games": 100,
                    "turns": 5000,
                    "stones": 9900,
                    "winner_counts": {"black": 60, "white": 35, "draw": 5},
                    "length_buckets": {
                        "short": 10,
                        "medium": 45,
                        "long": 35,
                        "very_long": 10,
                    },
                    "seconds": 20,
                    "games_per_second": 5,
                    "turns_per_second": 250,
                    "workers": [
                        {
                            "audit": {
                                "valid": True,
                                "valid_turns": 5000,
                                "unique_states": 4500,
                            }
                        }
                    ],
                },
            }
        )
    state = {
        "status": "running",
        "updated_unix": time.time(),
        "config": {"rounds": 10, "simulations": 400},
        "completed_rounds": rounds,
        "trained_through": 1,
        "promotions": [],
        "accepted_sha256": "b" * 64,
    }
    (run / "state.json").write_text(json.dumps(state))

    monitor = TrainingMonitor(root)
    records, default_id = monitor.runs()
    payload = monitor.snapshot(default_id)

    assert records[0].name == "run-a"
    assert payload["cumulative"]["games"] == 300
    assert payload["cumulative"]["black_decisive"] == 60 / 95
    assert payload["rounds"][-1]["rolling_black_decisive"] == 60 / 95
    assert payload["rounds"][-1]["white_simulations"] == 520
    assert payload["rounds"][-1]["opening_top_share"] == 2 / 3
    assert payload["rounds"][-1]["unique_complete_game_rate"] == 2 / 3
    assert payload["cumulative"]["invalid_audits"] == 0
    assert {alert["title"] for alert in payload["alerts"]} >= {
        "黑白胜率离开目标区间",
        "颜色补偿已触发",
    }


def test_training_monitor_defaults_to_latest_terminal_run(tmp_path):
    root = tmp_path / "training"
    old = root / "old-paused"
    new = root / "new-complete"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "state.json").write_text(
        json.dumps(
            {
                "status": "paused",
                "updated_unix": 100.0,
                "config": {"rounds": 100},
                "completed_rounds": [],
            }
        )
    )
    (new / "state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "updated_unix": 200.0,
                "config": {"rounds": 100},
                "completed_rounds": [],
            }
        )
    )
    monitor = TrainingMonitor(root)
    _, default_id = monitor.runs()
    selected = monitor.resolve(default_id)
    assert selected is not None
    assert selected.name == "new-complete"
