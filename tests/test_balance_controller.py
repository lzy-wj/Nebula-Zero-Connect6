import os
import sys
import unittest


PAIR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "pair_policy")
)
sys.path.insert(0, PAIR_DIR)

from balance_controller import update_opening_ratio, white_win_weight


class BalanceControllerTest(unittest.TestCase):
    def test_white_weight_targets_effective_fraction(self):
        weight = white_win_weight(0.21, target_fraction=0.40)
        effective = 0.21 * weight / (0.21 * weight + 0.79)
        self.assertAlmostEqual(effective, 0.40)

    def test_opening_ratio_moves_at_most_ten_percent(self):
        result = update_opening_ratio(
            0.50,
            {
                "standard_games": 100,
                "standard_black_wins": 80,
                "injected_games": 100,
                "injected_black_wins": 40,
            },
        )
        self.assertAlmostEqual(result["next_opening_ratio"], 0.60)

    def test_controller_holds_inside_deadband(self):
        result = update_opening_ratio(
            0.50,
            {
                "standard_games": 100,
                "standard_black_wins": 60,
                "injected_games": 100,
                "injected_black_wins": 40,
            },
        )
        self.assertEqual(result["reason"], "deadband")
        self.assertEqual(result["next_opening_ratio"], 0.50)

    def test_controller_reduces_harmful_injected_openings(self):
        result = update_opening_ratio(
            0.50,
            {
                "standard_games": 100,
                "standard_black_wins": 40,
                "injected_games": 100,
                "injected_black_wins": 80,
            },
        )
        self.assertAlmostEqual(result["next_opening_ratio"], 0.40)


if __name__ == "__main__":
    unittest.main()
