import os
import random
import sys
import unittest


PAIR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "pair_policy")
)
sys.path.insert(0, PAIR_DIR)

from build_replay import stratified_validation_split


class PairReplayTest(unittest.TestCase):
    def test_validation_is_color_balanced_when_both_outcomes_exist(self):
        games = [
            (f"b{index}", "black", "0:1", "0") for index in range(16)
        ] + [
            (f"w{index}", "white", "0:1", "0") for index in range(4)
        ]
        training, validation = stratified_validation_split(
            games,
            validation_ratio=0.2,
            rng=random.Random(7),
        )

        self.assertEqual(len(validation), 4)
        self.assertEqual(sum(game[1] == "black" for game in validation), 2)
        self.assertEqual(sum(game[1] == "white" for game in validation), 2)
        self.assertEqual(len(training), 16)
        self.assertFalse(set(training) & set(validation))


if __name__ == "__main__":
    unittest.main()
