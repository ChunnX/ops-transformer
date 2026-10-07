"""
Tests for accumulator bit-level integer precision probe up to 2^24 boundary.
Reference: LowPrecisionDotProduct Section VIII-D, SageAttention2 Section 2.4.
Verifies integer exact domain, float32 24-bit mantissa exactness,
and SageAttention2 INT8 QK^T / FP8 PV accumulation headroom.
"""

import unittest
import numpy as np
import torch


class TestAccumulatorBitProbe(unittest.TestCase):
    def test_fp32_mantissa_exact_range(self):
        """
        Verify that IEEE 754 Float32 has 24 bits of exact integer precision [-(2^24), 2^24].
        Boundary value: 2^24 = 16,777,216.
        """
        max_exact = 1 << 24  # 16777216

        # In float32, integers from 0 to 2^24 are all exactly representable
        f_max = np.float32(max_exact)
        self.assertEqual(int(f_max), max_exact)

        # 2^24 - 1 is exact
        f_prev = np.float32(max_exact - 1)
        self.assertEqual(int(f_prev), max_exact - 1)

        # 2^24 + 1 cannot be represented exactly in float32 and rounds to 2^24 under RNE
        f_next = np.float32(max_exact) + np.float32(1.0)
        # In float32 RNE, 16777216 + 1 rounds to 16777216 (even tie breaking)
        self.assertEqual(int(f_next), max_exact)

        # 2^24 + 2 is representable (step size becomes 2)
        f_next2 = np.float32(max_exact) + np.float32(2.0)
        self.assertEqual(int(f_next2), max_exact + 2)

    def test_plus_minus_one_accumulation_to_boundary(self):
        """
        Test step-by-step ±1 integer accumulation up to 2^24 boundary in FP32.
        Verifies that no precision drift occurs within [-2^24, 2^24].
        """
        # Test around boundary: start at 2^24 - 1024, accumulate step-by-step
        start_val = (1 << 24) - 1024
        curr = np.float32(start_val)

        for i in range(1024):
            expected = start_val + i + 1
            curr = curr + np.float32(1.0)
            self.assertEqual(int(curr), expected)

        # Now curr is exactly 2^24
        self.assertEqual(int(curr), 1 << 24)

        # Adding 1 fails to increment
        curr_plus_1 = curr + np.float32(1.0)
        self.assertEqual(int(curr_plus_1), 1 << 24)

    def test_sageattention2_qk_accumulator_bound(self):
        """
        Verify INT8 QK^T accumulation bound for D=128:
        max |q_i * k_i| = 127 * 127 = 16,129.
        For D=128: max dot product = 128 * 16,129 = 2,064,512.
        2,064,512 < 2^21 (2,097,152) < 2^24 (16,777,216).
        Hence, accumulator never exceeds 2^21 and stays strictly in float32 exact domain.
        """
        max_q = 127
        max_k = 127
        d = 128
        max_dot = d * max_q * max_k
        self.assertEqual(max_dot, 2064512)
        self.assertLess(max_dot, 1 << 21)
        self.assertLess(max_dot, 1 << 24)

        # PyTorch / float32 representation check
        t_dot = torch.tensor(max_dot, dtype=torch.float32)
        self.assertEqual(int(t_dot.item()), max_dot)

    def test_two_level_pv_accumulation_bound(self):
        """
        Verify two-level accumulation bound for fp32+fp16 mode:
        k_step = 16.
        P max in 448 amplification domain: P <= 448.0 (sum P over row = 448.0).
        V max: scale_max = 2.25.
        Sub-block accumulation over k_step=16: sum(P_i * V_i) <= 448 * 2.25 = 1008.0 <= 2047.
        Strictly fits in FP16 normalized range without overflow or precision collapse.
        """
        p_row_sum_max = 448.0
        v_scale_max = 2.25
        max_subblock_val = p_row_sum_max * v_scale_max
        self.assertEqual(max_subblock_val, 1008.0)
        self.assertLessEqual(max_subblock_val, 2047.0)

        # Verify in FP16
        t_fp16 = torch.tensor(max_subblock_val, dtype=torch.float16)
        self.assertAlmostEqual(t_fp16.item(), 1008.0, places=1)


if __name__ == "__main__":
    unittest.main()
