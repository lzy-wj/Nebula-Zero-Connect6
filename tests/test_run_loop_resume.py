import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RL_DIR = os.path.join(ROOT, 'reinforcement_learning')
sys.path.insert(0, ROOT)
sys.path.insert(0, RL_DIR)

import config
import run_loop


class ProductionResumeTest(unittest.TestCase):
    def test_phase_order_only_runs_pending_work(self):
        self.assertFalse(run_loop.should_run_phase('evaluation', 'selfplay'))
        self.assertFalse(run_loop.should_run_phase('evaluation', 'buffer'))
        self.assertFalse(run_loop.should_run_phase('evaluation', 'training'))
        self.assertFalse(run_loop.should_run_phase('evaluation', 'engine'))
        self.assertTrue(run_loop.should_run_phase('evaluation', 'evaluation'))

    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(RuntimeError):
            run_loop.should_run_phase('done', 'evaluation')

    def test_progress_closes_post_training_and_post_engine_crash_windows(self):
        self.assertEqual(
            run_loop.reconcile_resume_phase('training', 'training'),
            'engine',
        )
        self.assertEqual(
            run_loop.reconcile_resume_phase('engine', 'engine'),
            'evaluation',
        )
        self.assertEqual(
            run_loop.reconcile_resume_phase('evaluation', 'evaluation'),
            'evaluation',
        )

    def test_incumbent_manifest_verifies_both_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / 'checkpoints'
            checkpoints.mkdir()
            model = root / 'candidate.pth'
            engine = root / 'candidate.engine'
            model.write_bytes(b'model-v1')
            engine.write_bytes(b'engine-v1')

            with mock.patch.object(config, 'CHECKPOINT_DIR', str(checkpoints)), mock.patch.object(
                config,
                'CURRENT_MODEL_PTH',
                str(checkpoints / 'best.pth'),
            ), mock.patch.object(
                config,
                'CURRENT_ENGINE_PATH',
                str(checkpoints / 'current_model.engine'),
            ):
                record = run_loop.create_incumbent_bundle(
                    str(model),
                    str(engine),
                    4,
                    '0',
                )
                active_model, active_engine = run_loop.validate_bundle(record)
                self.assertEqual(Path(active_model).read_bytes(), b'model-v1')
                self.assertEqual(Path(active_engine).read_bytes(), b'engine-v1')
                self.assertEqual(run_loop.load_active_bundle()['generation'], 4)

                Path(active_engine).write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError, '引擎哈希不匹配'):
                    run_loop.load_active_bundle()

    def test_evaluation_resume_skips_completed_phases_and_reuses_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / 'raw'
            buffer_dir = root / 'buffer'
            checkpoints = root / 'checkpoints'
            logs = root / 'logs'
            for directory in (raw, buffer_dir, checkpoints, logs):
                directory.mkdir(parents=True)

            header = 'moves,winner,policies,bonuses\n'
            row = '"a1",black,"0:1","0"\n'
            (raw / 'gen_7.csv').write_text(header + row, encoding='utf-8')
            (buffer_dir / 'replay_buffer.csv').write_text(
                header + row,
                encoding='utf-8',
            )
            for name in ('candidate_gen_7.pth', 'model_gen_7.pth', 'model_gen_7.engine'):
                (checkpoints / name).write_bytes(name.encode())
            (logs / 'eval_gen_7.json').write_text(
                json.dumps({
                    'incumbent': {
                        'games': 2,
                        'wins': 0,
                        'losses': 2,
                        'draws': 0,
                        'score_rate': 0.0,
                        'white_win_rate': 0.0,
                        'game_black_win_rate': 0.5,
                        'paired_openings': 1,
                        'improvement_probability': 0.0,
                    }
                }),
                encoding='utf-8',
            )

            config_values = {
                'RAW_DATA_DIR': str(raw),
                'BUFFER_DIR': str(buffer_dir),
                'CHECKPOINT_DIR': str(checkpoints),
                'LOG_DIR': str(logs),
                'GAMES_PER_LOOP': 1,
                'HOT_START_MIN_BUFFER': 1,
                'EVAL_GAMES': 2,
                'GATING_MIN_PAIRS': 1,
            }
            forbidden = mock.Mock(side_effect=AssertionError('completed phase reran'))
            with mock.patch.multiple(config, **config_values), mock.patch.object(
                config,
                'ensure_dirs',
            ), mock.patch.object(
                run_loop,
                'ensure_initial_assets',
            ), mock.patch.object(
                run_loop,
                'load_loop_state',
                return_value={
                    'generation': 7,
                    'phase': 'evaluation',
                    'current_epochs': 3,
                    'accepted_generation': 6,
                },
            ), mock.patch.object(
                run_loop,
                'initialize_swanlab',
                return_value=False,
            ), mock.patch.object(
                run_loop,
                'finish_swanlab',
            ), mock.patch.object(
                run_loop,
                'run_command',
                forbidden,
            ), mock.patch.object(
                run_loop,
                'build_engine_from_checkpoint',
                forbidden,
            ), mock.patch.object(
                run_loop,
                'analyze_generation_data',
                return_value={},
            ), mock.patch.object(
                run_loop,
                'update_history_plot',
            ):
                run_loop.run_training_loop(
                    max_generations=1,
                    swanlab_mode='disabled',
                )

            forbidden.assert_not_called()
            state = json.loads((logs / 'loop_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['generation'], 8)
            self.assertEqual(state['phase'], 'selfplay')


if __name__ == '__main__':
    unittest.main()
