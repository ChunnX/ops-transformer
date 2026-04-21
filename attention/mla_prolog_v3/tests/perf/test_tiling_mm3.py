#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for mm3_block_num / mm3_single_core_n selection in compute_tiling.

Covers each live branch of the MM3 N-dim core allocation logic in
perf/tiling/config.py:177-192:
  - default split-N: BF16 (NO_QUANT) and PARTIAL_QUANT (uqqr_dtype_size=2)
  - enable_dequant_opt path: FULL_QUANT and MXFP8 (uqqr_dtype_size=1) with N >= 8
  - enable_group_compute_opt path: PARTIAL_QUANT, T=1, Nkv=8, cv_ratio != 1
  - 910B vs 950 core-count differences (aic_num=24 vs 32)

The split_m_flag branch in compute_tiling is currently dead code (the flag is
hardcoded to False at line 115); split-M is dispatched through
estimate_all_stages_split_m instead. Not covered here.

Each test asserts both mm3_block_num and mm3_single_core_n.

Run with: python test_tiling_mm3.py
(Same convention as test_perf_unit.py — pytest invocation is blocked by the
 empty tests/perf/__init__.py shadowing the inner perf/ package.)
"""

import unittest

from perf.model.hardware import ASCEND_910B, ASCEND_950
from perf.model.operator_params import OperatorParams
from perf.tiling.config import compute_tiling


def _make_params(head_num=128, D=128, Dr=64, batch_size=1, seq_len=1, Nkv=1,
                 weight_quant_mode=0, kv_cache_quant_mode=0):
    return OperatorParams(
        batch_size=batch_size, seq_len=seq_len, head_num=head_num, Nkv=Nkv,
        He=7168, Hcq=1536, Hckv=512, D=D, Dr=Dr,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode,
    )


class TestMm3DefaultSplitN(unittest.TestCase):
    """Default split-N branch: BF16 / PARTIAL_QUANT with uqqr_dtype_size=2.

    Formula: align = BLOCK_SIZE // uqqr_dtype_size = 32 // 2 = 16
             single_n = ceil_div(N*(D+Dr), align*aic_num) * align
             block_num = ceil_div(N*(D+Dr), single_n)
    """

    def _check(self, head_num, D, Dr, expected_block, expected_single_n):
        params = _make_params(head_num=head_num, D=D, Dr=Dr)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertFalse(cfg.enable_group_compute_opt)
        self.assertFalse(cfg.enable_dequant_opt)
        self.assertEqual(cfg.mm3_block_num, expected_block,
                         f"block_num for hd={head_num}")
        self.assertEqual(cfg.mm3_single_core_n, expected_single_n,
                         f"single_core_n for hd={head_num}")
        self.assertGreaterEqual(cfg.mm3_block_num * cfg.mm3_single_core_n,
                                head_num * (D + Dr),
                                "must cover full N dimension")

    def test_hd128_bf16_950(self):
        # ori_m3=24576, align=16, single_n=768, block=32
        self._check(128, 128, 64, expected_block=32, expected_single_n=768)

    def test_hd32_bf16_950(self):
        # ori_m3=6144, single_n=192, block=32
        self._check(32, 128, 64, expected_block=32, expected_single_n=192)

    def test_hd8_bf16_950(self):
        # ori_m3=1536, single_n=48, block=32
        self._check(8, 128, 64, expected_block=32, expected_single_n=48)

    def test_hd64_d256_bf16_950(self):
        # Larger D: ori_m3=64*320=20480, single_n=640, block=32
        self._check(64, 256, 64, expected_block=32, expected_single_n=640)


class TestMm3DequantOptPath(unittest.TestCase):
    """enable_dequant_opt branch: FULL_QUANT or MXFP8, N >= 8.

    Formula: single_n = ceil_div(N*(D+Dr), (D+Dr)*aic_num) * (D+Dr)
    """

    def test_full_quant_kv_no_quant_hd128_950(self):
        # uqqr=1, N=128: ori_m3=24576, align=192, single_n=768, block=32
        params = _make_params(weight_quant_mode=2, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertTrue(cfg.enable_dequant_opt)
        self.assertFalse(cfg.enable_group_compute_opt)
        self.assertEqual(cfg.mm3_block_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 768)

    def test_full_quant_kv_per_tensor_hd128_950(self):
        params = _make_params(weight_quant_mode=2, kv_cache_quant_mode=1)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertTrue(cfg.enable_dequant_opt)
        self.assertEqual(cfg.mm3_block_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 768)

    def test_mxfp8_kv_no_quant_hd128_950(self):
        params = _make_params(weight_quant_mode=3, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertTrue(cfg.enable_dequant_opt)
        self.assertEqual(cfg.mm3_block_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 768)

    def test_mxfp8_kv_per_tensor_hd128_950(self):
        params = _make_params(weight_quant_mode=3, kv_cache_quant_mode=1)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertTrue(cfg.enable_dequant_opt)
        self.assertEqual(cfg.mm3_block_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 768)

    def test_full_quant_small_n_hd8_950(self):
        # Boundary: N=8 still triggers (>= 8)
        # ori_m3=1536, align=192, single_n=192, block=8
        params = _make_params(head_num=8, weight_quant_mode=2, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertTrue(cfg.enable_dequant_opt)
        self.assertEqual(cfg.mm3_block_num, 8)
        self.assertEqual(cfg.mm3_single_core_n, 192)

    def test_partial_quant_does_not_trigger(self):
        """PARTIAL_QUANT (weight_quant_mode=1) keeps uqqr=2 → default branch."""
        # T=1, Nkv=1 (default) so group_compute_opt also off
        params = _make_params(weight_quant_mode=1, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertFalse(cfg.enable_group_compute_opt)
        self.assertFalse(cfg.enable_dequant_opt)
        # default branch numbers
        self.assertEqual(cfg.mm3_block_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 768)


class TestMm3GroupComputeOpt(unittest.TestCase):
    """enable_group_compute_opt branch.

    Conditions: PARTIAL_QUANT, T=1, Nkv=8, aic>=16, aiv>=32, cv_ratio != 1.
    After enable: aiv_num=32, aic_num=16.
    Formula: single_n = ceil_div(N*D, GROUP=8 * D) * D
    """

    def test_partial_quant_t1_nkv8_950(self):
        # cv_ratio = 64/32 = 2 ≠ 1 → triggers
        # After enable: aic_num=16
        # ori_m3 = 128*192 = 24576
        # single_n = calc_single_core_n(N*D=16384, GROUP=8, D=128) = 16*128 = 2048
        # block_num = ceil_div(24576, 2048) = 12
        params = _make_params(Nkv=8, weight_quant_mode=1, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertTrue(cfg.enable_group_compute_opt)
        self.assertEqual(cfg.aic_num, 16)
        self.assertEqual(cfg.aiv_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 2048)
        self.assertEqual(cfg.mm3_block_num, 12)

    def test_skipped_when_T_gt_1(self):
        """T > 1 disables group_compute_opt → default branch."""
        params = _make_params(batch_size=2, Nkv=8,
                              weight_quant_mode=1, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_950)
        self.assertFalse(cfg.enable_group_compute_opt)
        self.assertEqual(cfg.aic_num, 32)
        self.assertEqual(cfg.mm3_block_num, 32)
        self.assertEqual(cfg.mm3_single_core_n, 768)


class TestMm3Ascend910B(unittest.TestCase):
    """910B has aic_num=24 → different single_n / block_num than 950."""

    def test_910b_default_split_n_hd128(self):
        # ori_m3=24576, align=16, aic=24
        # single_n = ceil_div(24576, 384)*16 = 64*16 = 1024
        # block = ceil_div(24576, 1024) = 24
        params = _make_params(head_num=128, weight_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_910B)
        self.assertFalse(cfg.enable_dequant_opt)
        self.assertEqual(cfg.aic_num, 24)
        self.assertEqual(cfg.mm3_block_num, 24)
        self.assertEqual(cfg.mm3_single_core_n, 1024)

    def test_910b_dequant_opt_hd128(self):
        # aic=24, align=192
        # single_n = ceil_div(24576, 192*24)*192 = ceil_div(24576, 4608)*192
        #          = 6*192 = 1152
        # block = ceil_div(24576, 1152) = 22 (22*1152=25344>=24576)
        params = _make_params(head_num=128, weight_quant_mode=2, kv_cache_quant_mode=0)
        cfg = compute_tiling(params, ASCEND_910B)
        self.assertTrue(cfg.enable_dequant_opt)
        self.assertEqual(cfg.mm3_single_core_n, 1152)
        self.assertEqual(cfg.mm3_block_num, 22)


class TestMm3FullCoverageInvariant(unittest.TestCase):
    """For any branch, block_num*single_n must cover full N*(D+Dr)."""

    def _check_invariant(self, params, hw):
        cfg = compute_tiling(params, hw)
        self.assertGreaterEqual(
            cfg.mm3_block_num * cfg.mm3_single_core_n,
            params.N * (params.D + params.Dr),
            f"coverage failed for hd={params.N}, qm={params.quant_mode}")
        self.assertLessEqual(cfg.mm3_block_num, cfg.aic_num,
                             "block_num must not exceed aic_num")

    def test_invariant_950_cases(self):
        cases = [
            _make_params(head_num=128),                               # default
            _make_params(head_num=32),
            _make_params(head_num=8),
            _make_params(head_num=128, weight_quant_mode=2),          # dequant_opt
            _make_params(head_num=128, weight_quant_mode=3),          # mxfp8
        ]
        for params in cases:
            with self.subTest(head_num=params.N, qm=params.quant_mode):
                self._check_invariant(params, ASCEND_950)


if __name__ == "__main__":
    unittest.main()
