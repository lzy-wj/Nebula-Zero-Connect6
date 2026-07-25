import os
import sys
import unittest

import torch


PAIR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "pair_policy")
)
sys.path.insert(0, PAIR_DIR)

from train_joint import normalized_outcome_weights


class PairTrainingBalanceTest(unittest.TestCase):
    def test_white_win_weight_is_normalized_without_losing_ratio(self):
        winners = torch.tensor([1, -1, 1, -1], dtype=torch.int64)
        weights = normalized_outcome_weights(winners, 2.0)

        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[1] / weights[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
