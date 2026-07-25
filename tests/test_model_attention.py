import os
import sys
import unittest

import torch


RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reinforcement_learning'))
sys.path.insert(0, RL_DIR)

from core.model import C6TransNet


class ModelAttentionTest(unittest.TestCase):
    def test_fused_attention_matches_explicit_attention(self):
        torch.manual_seed(7)
        model = C6TransNet().eval()
        inputs = torch.randn(1, 17, 19, 19)
        with torch.no_grad():
            fused_policy, _, fused_value = model(inputs)
            explicit_policy, _, explicit_value, _ = model(inputs, return_attn=True)

        self.assertTrue(torch.allclose(fused_policy, explicit_policy, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(fused_value, explicit_value, atol=2e-5, rtol=2e-5))


if __name__ == '__main__':
    unittest.main()
