import os
import sys
import tempfile
import unittest
from unittest import mock


RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning'))
sys.path.insert(0, RL_DIR)

from pipeline import remote_generate


class RemoteGenerateTest(unittest.TestCase):
    def _write_csv(self, path, rows):
        with open(path, 'w', encoding='utf-8') as file:
            file.write(remote_generate.CSV_HEADER + '\n')
            for row in rows:
                file.write(row + '\n')

    def test_csv_validation_counts_only_nonempty_games(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'games.csv')
            self._write_csv(path, ['"a1",black,"0:1","0"', ''])
            self.assertEqual(remote_generate.validate_csv(path, expected_rows=1), 1)

    def test_bad_csv_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'games.csv')
            with open(path, 'w', encoding='utf-8') as file:
                file.write('bad,header\n')
            with self.assertRaises(ValueError):
                remote_generate.validate_csv(path)

    def test_reconcile_pulls_more_complete_remote_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, 'gen_1.csv')
            self._write_csv(local_path, ['"a1",black,"0:1","0"'])
            with mock.patch.object(remote_generate, 'remote_csv_rows', return_value=3), mock.patch.object(
                remote_generate, 'atomic_pull'
            ) as pull:
                count = remote_generate.reconcile_generation_file(
                    'host', local_path, '/remote/gen_1.csv'
                )
            self.assertEqual(count, 3)
            pull.assert_called_once_with('host', '/remote/gen_1.csv', local_path)

    def test_reconcile_pushes_more_complete_local_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, 'gen_1.csv')
            self._write_csv(
                local_path,
                ['"a1",black,"0:1","0"', '"b1",white,"1:1","0"'],
            )
            with mock.patch.object(remote_generate, 'remote_csv_rows', return_value=1), mock.patch.object(
                remote_generate, 'atomic_push'
            ) as push:
                count = remote_generate.reconcile_generation_file(
                    'host', local_path, '/remote/gen_1.csv'
                )
            self.assertEqual(count, 2)
            push.assert_called_once_with('host', local_path, '/remote/gen_1.csv')


if __name__ == '__main__':
    unittest.main()
