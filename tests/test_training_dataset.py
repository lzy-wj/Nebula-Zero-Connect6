import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning'))
sys.path.insert(0, RL_DIR)

from pipeline.train import Connect6Dataset, resolve_precision


class TrainingDatasetTest(unittest.TestCase):
    def _dataset(self, policies):
        temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(temp_dir.name, 'games.csv')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('moves,winner,policies,bonuses\n')
            f.write(f'"a1,b1,c1",white,"{policies}","0,0,0"\n')
        return temp_dir, Connect6Dataset([path])

    def test_value_is_from_player_to_move_perspective(self):
        temp_dir, dataset = self._dataset('0:1|1:1|2:1')
        self.addCleanup(temp_dir.cleanup)
        with mock.patch('numpy.random.randint', side_effect=[1, 0]), mock.patch(
            'numpy.random.rand', return_value=1.0
        ):
            features, policy, value, _ = dataset[0]

        self.assertEqual(float(value), 1.0)
        self.assertEqual(float(policy.sum()), 1.0)
        self.assertEqual(float(features[0].sum()), 0.0)
        self.assertEqual(float(features[1].sum()), 1.0)

    def test_missing_search_policy_falls_back_to_played_move(self):
        temp_dir, dataset = self._dataset('0:1||')
        self.addCleanup(temp_dir.cleanup)
        with mock.patch('numpy.random.randint', side_effect=[2, 0]), mock.patch(
            'numpy.random.rand', return_value=1.0
        ):
            _, policy, _, _ = dataset[0]

        self.assertEqual(float(policy.sum()), 1.0)
        self.assertEqual(int(policy.argmax()), 2)

    def test_cpu_precision_falls_back_to_fp32(self):
        import torch

        precision, dtype = resolve_precision(torch.device('cpu'), 'bf16')
        self.assertEqual(precision, 'fp32')
        self.assertIsNone(dtype)


if __name__ == '__main__':
    unittest.main()
