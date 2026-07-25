import os
import sys
import unittest

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V3_DIR = os.path.join(ROOT, 'experiments', 'nebula_v3')
RL_DIR = os.path.join(ROOT, 'reinforcement_learning')
sys.path.insert(0, V3_DIR)
sys.path.insert(0, RL_DIR)

from dataset import GameRecord, SelfPlayPositionDataset, player_at_move
from model import NebulaNetV3
from model_fast import FastC6NetV4
from core.model import C6TransNet


class NebulaV3DatasetTest(unittest.TestCase):
    def test_player_sequence_matches_connect6_turns(self):
        self.assertEqual(
            [player_at_move(index) for index in range(7)],
            [1, -1, -1, 1, 1, -1, -1],
        )

    def test_features_phase_value_and_legal_policy(self):
        record = GameRecord(
            moves=(0, 1, 2, 3, 4),
            winner=-1,
            policies=(
                '0:1',
                '1:1',
                '0:0.5;2:0.5',
                '3:1',
                '4:1',
            ),
            generation=806,
            is_evaluation=False,
        )
        dataset = SelfPlayPositionDataset(
            [record],
            training=False,
            positions_per_game=1,
        )
        features, policy, value = dataset[0]

        # 固定验证位置位于 move_index=2：白方本回合第二颗子。
        self.assertEqual(float(features[0].sum()), 1.0)
        self.assertEqual(float(features[1].sum()), 1.0)
        self.assertEqual(float(features[2].sum()), 359.0)
        self.assertEqual(float(features[3].sum()), 0.0)
        self.assertEqual(float(features[4].sum()), 361.0)
        self.assertEqual(float(policy[0]), 0.0)
        self.assertEqual(float(policy[2]), 1.0)
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertEqual(float(value), 1.0)


class NebulaV3ModelTest(unittest.TestCase):
    def test_forward_backward_is_finite(self):
        torch.manual_seed(7)
        model = NebulaNetV3(
            channels=64,
            conv_depth=2,
            transformer_depth=1,
            num_heads=2,
            drop_path_rate=0.0,
        )
        inputs = torch.randn(2, 5, 19, 19)
        policy, policy2, value = model(inputs)
        loss = policy.square().mean() + value.square().mean()
        loss.backward()

        self.assertIsNone(policy2)
        self.assertEqual(tuple(policy.shape), (2, 361))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertTrue(torch.isfinite(policy).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_fast_v4_bn_folding_preserves_full_v2(self):
        torch.manual_seed(11)
        teacher = C6TransNet(input_planes=17).eval()
        student = FastC6NetV4(
            source_block_indices=(0, 1, 2, 3, 4, 5),
        ).eval().initialize_from_v2(teacher)

        features = torch.randn(1, 5, 19, 19)
        old_features = torch.zeros(1, 17, 19, 19)
        old_features[:, 0] = features[:, 0]
        old_features[:, 1] = features[:, 1]
        old_features[:, 16] = features[:, 3]
        with torch.no_grad():
            old_policy, _, old_value = teacher(old_features)
            new_policy, new_policy2, new_value = student(features)

        self.assertIsNone(new_policy2)
        self.assertTrue(torch.allclose(old_policy, new_policy, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(old_value, new_value, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
