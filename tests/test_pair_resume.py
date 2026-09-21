import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.pair_policy import run_loop


class PairResumeTest(unittest.TestCase):
    def test_phase_order_only_runs_pending_work(self):
        self.assertFalse(run_loop.should_run_phase("evaluation", "selfplay"))
        self.assertFalse(run_loop.should_run_phase("evaluation", "training"))
        self.assertFalse(run_loop.should_run_phase("evaluation", "engine"))
        self.assertTrue(run_loop.should_run_phase("evaluation", "evaluation"))
        self.assertTrue(run_loop.should_run_phase("training", "training"))
        self.assertTrue(run_loop.should_run_phase("training", "evaluation"))

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(RuntimeError):
            run_loop.should_run_phase("done", "evaluation")

    def test_empty_pair_targets_are_rejected_before_training(self):
        with self.assertRaisesRegex(RuntimeError, "train=0, validation=0"):
            run_loop.require_training_positions({
                "train_positions": 0,
                "validation_positions": 0,
            })

    def test_pair_generation_uses_production_refresh_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            values = run_loop.generation_environment("heads.pt", 2)
        self.assertEqual(values["NEBULA_PAIR_REFRESH_VISITS_BLACK"], 1_000_000_000)
        self.assertEqual(values["NEBULA_PAIR_REFRESH_VISITS_WHITE"], 2)
        self.assertEqual(values["NEBULA_PAIR_DEFER_REFRESH"], 0)

    def test_complete_gate_requires_json_and_every_game(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gate_path = Path(temp_dir) / "gate.json"
            data_path = Path(temp_dir) / "gate.csv"
            gate_path.write_text(json.dumps({"games": 2}), encoding="utf-8")
            data_path.write_text(
                "moves,winner,policies,bonuses\n"
                '"a1",black,"0:1","0"\n',
                encoding="utf-8",
            )
            self.assertIsNone(
                run_loop.complete_gate_result(str(gate_path), str(data_path), 2)
            )
            with data_path.open("a", encoding="utf-8") as output:
                output.write('"b1",white,"1:1","0"\n')
            self.assertEqual(
                run_loop.complete_gate_result(str(gate_path), str(data_path), 2),
                {"games": 2},
            )

    def test_evaluation_resume_does_not_retrain_or_rebuild(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            replay = root / "replay"
            checkpoints = root / "checkpoints"
            logs = root / "logs"
            summaries = logs / "generation_summaries"
            runtime = root / "runtime"
            candidate = checkpoints / "gen_0007"
            for directory in (data, replay, candidate, summaries, runtime):
                directory.mkdir(parents=True, exist_ok=True)

            header = "moves,winner,policies,bonuses\n"
            row = '"a1",black,"0:1","0"\n'
            (data / "gen_0007_pair.csv").write_text(header + row, encoding="utf-8")
            (data / "gen_0007_anchor.csv").write_text(header + row, encoding="utf-8")
            (replay / "gen_0007_train.csv").write_text(header + row, encoding="utf-8")
            (replay / "gen_0007_validation.csv").write_text(
                header + row,
                encoding="utf-8",
            )
            (replay / "gen_0007_stats.json").write_text("{}", encoding="utf-8")
            for name in (
                "main.pth",
                "pair_heads.pt",
                "gen_0007_pair.engine",
                "gen_0007_exact.engine",
            ):
                (candidate / name).write_bytes(b"candidate")

            def fake_evaluation(command, env_vars=None):
                self.assertIn("evaluate_pair.py", " ".join(command))
                (logs / "gen_0007_gate.json").write_text(
                    json.dumps({
                        "games": 2,
                        "wins": 0,
                        "losses": 2,
                        "draws": 0,
                        "score_rate": 0.0,
                        "black_win_rate": 0.0,
                        "white_win_rate": 0.0,
                        "game_black_win_rate": 0.5,
                        "elapsed_seconds": 1.0,
                    }),
                    encoding="utf-8",
                )
                (data / "gen_0007_gate.csv").write_text(
                    header + row + row,
                    encoding="utf-8",
                )

            forbidden_generate = mock.Mock(side_effect=AssertionError("selfplay reran"))
            forbidden_replay = mock.Mock(side_effect=AssertionError("replay rebuilt"))
            forbidden_bundle = mock.Mock(side_effect=AssertionError("engine rebuilt"))

            patched_paths = {
                "RUN_ROOT": str(root),
                "DATA_DIR": str(data),
                "REPLAY_DIR": str(replay),
                "CHECKPOINT_DIR": str(checkpoints),
                "LOG_DIR": str(logs),
                "RUNTIME_DIR": str(runtime),
                "SUMMARY_DIR": str(summaries),
                "STATE_PATH": str(logs / "loop_state.json"),
                "CONTROLLER_PATH": str(logs / "balance_controller.json"),
                "CURRENT_MAIN": str(checkpoints / "current_main.pth"),
                "CURRENT_HEADS": str(checkpoints / "current_pair_heads.pt"),
                "CURRENT_PAIR_ENGINE": str(checkpoints / "current_pair.engine"),
                "CURRENT_EXACT_ENGINE": str(checkpoints / "current_exact.engine"),
                "MCTS_LIBRARY": str(runtime / "libmcts.so"),
            }
            environment = {
                "NEBULA_PAIR_ONLINE_GAMES": "1",
                "NEBULA_PAIR_ANCHOR_GAMES": "1",
                "NEBULA_PAIR_EVAL_GAMES": "2",
                "NEBULA_AUTO_BALANCE": "0",
            }
            quality = {
                "standard_games": 1,
                "standard_black_wins": 1,
                "injected_games": 0,
                "injected_black_wins": 0,
            }

            with mock.patch.multiple(run_loop, **patched_paths), mock.patch.dict(
                os.environ,
                environment,
                clear=False,
            ), mock.patch.object(
                run_loop,
                "load_state",
                return_value={
                    "generation": 7,
                    "accepted_generation": 6,
                    "phase": "evaluation",
                },
            ), mock.patch.object(
                run_loop,
                "init_generation_swanlab",
                return_value=False,
            ), mock.patch.object(
                run_loop,
                "finish_swanlab",
            ), mock.patch.object(
                run_loop,
                "initialize_bundle",
            ), mock.patch.object(
                run_loop,
                "generate_games",
                forbidden_generate,
            ), mock.patch.object(
                run_loop,
                "build_replay",
                forbidden_replay,
            ), mock.patch.object(
                run_loop,
                "build_bundle",
                forbidden_bundle,
            ), mock.patch.object(
                run_loop,
                "analyze_games",
                return_value=quality,
            ), mock.patch.object(
                run_loop,
                "load_candidate_metrics",
                return_value={},
            ), mock.patch.object(
                run_loop,
                "run_command",
                side_effect=fake_evaluation,
            ) as evaluation:
                summary = run_loop.run_one_generation("disabled")

            self.assertEqual(evaluation.call_count, 1)
            self.assertEqual(summary["resumed_from_phase"], "evaluation")
            self.assertEqual(summary["generation"], 7)
            state = json.loads((logs / "loop_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["generation"], 8)
            self.assertEqual(state["phase"], "selfplay")


if __name__ == "__main__":
    unittest.main()
