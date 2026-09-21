import unittest

from experiments.pair_policy.gating import gate_passes, paired_score_statistics


class PairGatingTest(unittest.TestCase):
    def test_equal_candidate_does_not_pass_confidence_gate(self):
        stats = paired_score_statistics([1.0, 0.0] * 100)
        self.assertEqual(stats["paired_score_rate"], 0.5)
        self.assertEqual(stats["improvement_probability"], 0.5)
        self.assertFalse(
            gate_passes({
                "games": 200,
                "score_rate": 0.5,
                "white_win_rate": 0.5,
                "game_black_win_rate": 0.5,
                **stats,
            })
        )

    def test_clear_paired_improvement_passes(self):
        game_scores = [1.0, 1.0] * 60 + [0.0, 0.0] * 40
        stats = paired_score_statistics(game_scores)
        self.assertAlmostEqual(stats["paired_score_rate"], 0.6)
        self.assertGreater(stats["improvement_probability"], 0.90)
        self.assertTrue(
            gate_passes({
                "games": 200,
                "score_rate": 0.6,
                "white_win_rate": 0.6,
                "game_black_win_rate": 0.5,
                **stats,
            })
        )

    def test_color_health_guard_can_reject_strong_candidate(self):
        self.assertFalse(
            gate_passes({
                "games": 200,
                "score_rate": 0.6,
                "improvement_probability": 0.99,
                "white_win_rate": 0.7,
                "game_black_win_rate": 0.2,
            })
        )

    def test_small_match_cannot_promote_a_candidate(self):
        self.assertFalse(
            gate_passes({
                "games": 4,
                "paired_openings": 2,
                "score_rate": 1.0,
                "improvement_probability": 1.0,
                "white_win_rate": 1.0,
                "game_black_win_rate": 0.5,
            })
        )


if __name__ == "__main__":
    unittest.main()
