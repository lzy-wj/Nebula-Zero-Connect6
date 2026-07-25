import csv
import os
import sys
import tempfile
import unittest


PAIR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "pair_policy")
)
sys.path.insert(0, PAIR_DIR)

from data_metrics import analyze_games


class PairDataMetricsTest(unittest.TestCase):
    def test_quality_and_target_balance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "games.csv")
            with open(path, "w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["moves", "winner", "policies", "bonuses"])
                writer.writerow([
                    "j10,i10,i11,j11,j12",
                    "black",
                    "180:1|179:1|160:1|181:1|162:1",
                    "0,0,0,0,0",
                ])
                writer.writerow([
                    "j10,k10,k11,l10,l11",
                    "white",
                    "180:1|181:1|162:1|182:1|163:1",
                    "0,0,0,0,0",
                ])
                writer.writerow([
                    "h8,i8,i9,j8,j9,k8,k9",
                    "white",
                    "|||||143:1|162:1",
                    "0,0,0,0,0,0,0",
                ])

            metrics = analyze_games(path)

        self.assertEqual(metrics["games"], 3)
        self.assertEqual(metrics["joint_samples"], 5)
        self.assertAlmostEqual(metrics["white_win_sample_fraction"], 3 / 5)
        self.assertAlmostEqual(metrics["black_win_rate"], 1 / 3)
        self.assertEqual(metrics["black_positive_target_rate"], 0.5)
        self.assertAlmostEqual(metrics["white_positive_target_rate"], 2 / 3)
        self.assertEqual(metrics["standard_games"], 2)
        self.assertEqual(metrics["injected_games"], 1)
        self.assertEqual(metrics["injected_black_win_rate"], 0.0)
        self.assertEqual(metrics["invalid_games"], 0)


if __name__ == "__main__":
    unittest.main()
