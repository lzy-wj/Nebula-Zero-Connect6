import os
import sys
import tempfile
import unittest
from unittest import mock


RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning'))
sys.path.insert(0, RL_DIR)

import run_loop


class ReplayBufferTest(unittest.TestCase):
    def _write_csv(self, path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('moves,winner,policies,bonuses\n')
            for row in rows:
                f.write(row + '\n')

    def test_repeated_ingestion_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, 'gen_0.csv')
            rows = [
                '"a1,b1",black,"0:1|1:1","0,0"',
                '"c1,d1",white,"2:1|3:1","0,0"',
            ]
            self._write_csv(source, rows)

            with mock.patch.object(run_loop.config, 'BUFFER_DIR', temp_dir), mock.patch.object(
                run_loop.config, 'BUFFER_SIZE', 10
            ):
                first = run_loop.manage_buffer(source)
                second = run_loop.manage_buffer(source)

            self.assertEqual(first, {'added': 2, 'duplicates': 0, 'size': 2})
            self.assertEqual(second, {'added': 0, 'duplicates': 2, 'size': 2})

    def test_buffer_keeps_latest_unique_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_source = os.path.join(temp_dir, 'first.csv')
            second_source = os.path.join(temp_dir, 'second.csv')
            self._write_csv(first_source, ['"a1",black,"0:1","0"', '"b1",white,"1:1","0"'])
            self._write_csv(second_source, ['"c1",black,"2:1","0"', '"d1",white,"3:1","0"'])

            with mock.patch.object(run_loop.config, 'BUFFER_DIR', temp_dir), mock.patch.object(
                run_loop.config, 'BUFFER_SIZE', 3
            ):
                run_loop.manage_buffer(first_source)
                result = run_loop.manage_buffer(second_source)

            self.assertEqual(result['size'], 3)
            buffer_path = os.path.join(temp_dir, 'replay_buffer.csv')
            with open(buffer_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertNotIn('"a1"', content)
            self.assertIn('"d1"', content)


if __name__ == '__main__':
    unittest.main()
