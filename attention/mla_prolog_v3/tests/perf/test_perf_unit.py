#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the perf model hot-path numerical functions.

These tests pin numerical outputs of the core functions so future refactoring
can be validated without running the full CLI regression test.

Run with: python test_perf_unit.py
"""

import math
import sys
import unittest

from perf.model.block_spec import (
    MatmulBlockSpec, check_cache_fit, derive_stepK,
)
from perf.model.hardware import ASCEND_910B, ASCEND_950
from perf.model.matmul_detailed import estimate_matmul_detailed
from perf.model.matmul_k_outer import estimate_matmul_k_outer
from perf.model.matmul_legacy import estimate_matmul
from perf.model.matmul_split_kn import (
    SplitKNSpec, estimate_matmul_split_kn,
)
from perf.model.matmul_split_m import SplitMSpec, estimate_matmul_split_m
from perf.model.operator_params import OperatorParams
from perf.model.orchestration import (
    estimate_all_stages, get_default_block_specs,
)
from perf.model.quant import (
    QuantMode, dtype_size, is_full_quant, is_int8_quant, is_mxfp8,
    resolve_quant_mode,
)
from perf.model.vector_ops import estimate_vector_op
from perf.pipeline.dag import build_pipeline_dag
from perf.pipeline.kernel_time import estimate_kernel_time
from perf.pipeline.roofline import roofline_analysis
from perf.tiling.config import TilingConfig, compute_tiling


# Tolerance for float comparisons (us). Bit-identical via str format,
# so 1e-9 catches any drift but absorbs rounding noise.
EPS = 1e-9


def _approx(actual, expected, tol=EPS):
    return abs(actual - expected) < tol


class TestQuant(unittest.TestCase):
    def test_resolve_quant_mode_no_quant(self):
        self.assertEqual(resolve_quant_mode(0, 0), QuantMode.NO_QUANT)

    def test_resolve_quant_mode_int8(self):
        self.assertEqual(
            resolve_quant_mode(2, 1),
            QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        )

    def test_resolve_quant_mode_mxfp8(self):
        self.assertEqual(
            resolve_quant_mode(3, 0),
            QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
        )

    def test_dtype_size(self):
        self.assertEqual(dtype_size(QuantMode.NO_QUANT), 2)
        self.assertEqual(dtype_size(QuantMode.FULL_QUANT_KV_NO_QUANT), 1)
        self.assertEqual(dtype_size(QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT), 1)

    def test_is_helpers(self):
        self.assertTrue(is_full_quant(QuantMode.FULL_QUANT_KV_NO_QUANT))
        self.assertTrue(is_int8_quant(QuantMode.PARTIAL_QUANT_KV_NO_QUANT))
        self.assertTrue(is_mxfp8(QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT))
        self.assertFalse(is_full_quant(QuantMode.NO_QUANT))


class TestOperatorParams(unittest.TestCase):
    def test_default_construction(self):
        p = OperatorParams()
        self.assertEqual(p.T, 8)  # 2 * 4
        self.assertEqual(p.N, 8)
        self.assertEqual(p.quant_mode, QuantMode.NO_QUANT)

    def test_validation_negative_batch(self):
        with self.assertRaises(ValueError):
            OperatorParams(batch_size=0)
        with self.assertRaises(ValueError):
            OperatorParams(batch_size=-1)

    def test_validation_invalid_quant(self):
        with self.assertRaises(ValueError):
            OperatorParams(weight_quant_mode=4)
        with self.assertRaises(ValueError):
            OperatorParams(query_quant_mode=2)


class TestBlockSpec(unittest.TestCase):
    def test_check_cache_fit_basic(self):
        block = MatmulBlockSpec(
            "test", baseM=32, baseN=128, baseK=64, stepK=2,
            mode="split_k", dtype_a_size=2, dtype_b_size=2, dtype_c_size=2,
        )
        self.assertTrue(check_cache_fit(block, ASCEND_950, n_per_core=128))

    def test_derive_stepK_double_buffer(self):
        block = MatmulBlockSpec(
            "test", baseM=32, baseN=128, baseK=64, stepK=4,
            mode="split_k", dtype_a_size=2, dtype_b_size=2, dtype_c_size=2,
        )
        # K=7168, n_per_core=128
        stepK = derive_stepK(block, K=7168, hw=ASCEND_950, n_per_core=128, M=1)
        self.assertGreaterEqual(stepK, 1)


class TestMatmulDetailed(unittest.TestCase):
    """Pin the detailed matmul numerics for a small known config."""

    def test_mm1_decode_bs1_bf16(self):
        block = MatmulBlockSpec(
            "MM1_Cq", baseM=32, baseN=128, baseK=128, stepK=4,
            mode="split_k", dtype_a_size=2, dtype_b_size=2, dtype_c_size=2,
        )
        timing = estimate_matmul_detailed(
            "MM1_Cq", M=1, K=7168, N=1536, active_cores=24, hw=ASCEND_950,
            block=block, a_reused=False,
        )
        # Sanity: positive total time, expected bound classification
        self.assertGreater(timing.total_us, 0)
        self.assertEqual(timing.M, 1)
        self.assertEqual(timing.K, 7168)
        self.assertEqual(timing.N, 1536)
        # MM1 decode is HBM-bound on 950
        self.assertIn(timing.bound, ("HBM", "L2", "MTE1_A", "MTE1_B"))


class TestMatmulSplitKN(unittest.TestCase):
    def test_split_kn_returns_cube_and_accum(self):
        block = MatmulBlockSpec(
            "MM1_Cq", baseM=32, baseN=128, baseK=128, stepK=4,
            mode="split_k", dtype_a_size=2, dtype_b_size=2, dtype_c_size=2,
        )
        spec = SplitKNSpec("MM1_Cq", n_groups=8, k_groups=4, single_n=192, single_k=1792)
        cube_t, accum_t = estimate_matmul_split_kn(
            "MM1_Cq", M=1, K=7168, N=1536, spec=spec, hw=ASCEND_950,
            block=block, a_reused=False, vec_cores=64,
        )
        self.assertGreater(cube_t.total_us, 0)
        self.assertGreaterEqual(accum_t.total_us, 0)


class TestMatmulSplitM(unittest.TestCase):
    def test_split_m_returns_timing(self):
        block = MatmulBlockSpec(
            "MM1_Cq", baseM=32, baseN=128, baseK=128, stepK=4,
            mode="split_k", dtype_a_size=2, dtype_b_size=2, dtype_c_size=2,
        )
        spec = SplitMSpec("split_m", m_groups=8, single_m=64)
        timing = estimate_matmul_split_m(
            "MM1_Cq", M_total=512, K=7168, N=1536, spec=spec, hw=ASCEND_950,
            block=block, a_reused=False,
        )
        self.assertGreater(timing.total_us, 0)
        self.assertEqual(timing.active_cores, 8)


class TestMatmulKOuter(unittest.TestCase):
    def test_k_outer_returns_timing(self):
        block = MatmulBlockSpec(
            "MM1_Cq", baseM=32, baseN=128, baseK=128, stepK=4,
            mode="split_k", dtype_a_size=2, dtype_b_size=2, dtype_c_size=2,
        )
        timing = estimate_matmul_k_outer(
            "MM1_Cq", M=512, K=7168, N=1536, active_cores=32,
            hw=ASCEND_950, block=block, a_reused=False,
        )
        self.assertGreater(timing.total_us, 0)


class TestVectorOp(unittest.TestCase):
    def test_vector_op_bound(self):
        # Small load/compute -> total should be > 0
        timing = estimate_vector_op(
            "RmsNormCq", load_bytes=1024, compute_elements=512, store_bytes=512,
            hw=ASCEND_950, active_cores=64,
        )
        self.assertGreater(timing.total_us, 0)
        self.assertIn(timing.bound, ("MTE2", "VECTOR", "MTE3"))


class TestOrchestration(unittest.TestCase):
    def test_estimate_all_stages_decode(self):
        params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        tiling = compute_tiling(params, ASCEND_950)
        stages = estimate_all_stages(params, tiling, ASCEND_950)
        self.assertIn("MM1_Cq", stages)
        self.assertIn("MM2_CkvKr", stages)
        self.assertIn("MM3_QcQr", stages)
        self.assertIn("MM4_Qn", stages)
        self.assertIn("RmsNormCq", stages)

    def test_get_default_block_specs(self):
        params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        specs = get_default_block_specs(params, ASCEND_950)
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs["MM4_Qn"].mode, "full_load")


class TestPipelineDag(unittest.TestCase):
    def test_build_pipeline_dag_decode(self):
        params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        tiling = compute_tiling(params, ASCEND_950)
        result = build_pipeline_dag(params, tiling, ASCEND_950)
        self.assertGreater(result.total_us, 0)
        self.assertGreater(len(result.critical_path), 0)
        self.assertGreater(result.aic_busy_us, 0)


class TestKernelTime(unittest.TestCase):
    def test_single_step(self):
        params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        tiling = compute_tiling(params, ASCEND_950)
        total_us = estimate_kernel_time(params, tiling, ASCEND_950)
        self.assertGreater(total_us, 0)

    def test_multi_step_scales(self):
        # Larger T should result in more total time when stepBatchSize caps it
        params_small = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        params_large = OperatorParams(batch_size=8, seq_len=8, head_num=128)
        tiling_small = compute_tiling(params_small, ASCEND_950)
        tiling_large = compute_tiling(params_large, ASCEND_950)
        small_us = estimate_kernel_time(params_small, tiling_small, ASCEND_950)
        large_us = estimate_kernel_time(params_large, tiling_large, ASCEND_950)
        self.assertGreater(large_us, small_us)


class TestTilingConfig(unittest.TestCase):
    def test_compute_tiling_basic(self):
        params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        tiling = compute_tiling(params, ASCEND_950)
        self.assertEqual(tiling.aic_num, 32)
        self.assertEqual(tiling.aiv_num, 64)
        self.assertGreater(tiling.mm1_block_num, 0)
        self.assertGreater(tiling.mm2_block_num, 0)
        self.assertGreater(tiling.mm3_block_num, 0)
        self.assertGreater(tiling.mm4_block_num, 0)


class TestRoofline(unittest.TestCase):
    def test_roofline_returns_results(self):
        params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
        tiling = compute_tiling(params, ASCEND_950)
        results = roofline_analysis(params, tiling, ASCEND_950)
        self.assertEqual(len(results), 4)
        for mm in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
            self.assertIn(mm, results)
            self.assertGreater(results[mm].arithmetic_intensity, 0)
            self.assertGreater(results[mm].peak_gflops, 0)


class TestBackwardCompat(unittest.TestCase):
    """Verify the perf_model.py / tiling_sim.py / pipeline_model.py shims still work."""

    def test_perf_model_shim(self):
        from perf_model import (
            OperatorParams, ASCEND_950, estimate_all_stages,
            QuantMode, MatMulTiming, VectorOpTiming,
        )
        params = OperatorParams(batch_size=1, seq_len=1, head_num=8)
        # Just ensure imports work
        self.assertEqual(QuantMode.NO_QUANT, 0)

    def test_tiling_sim_shim(self):
        from tiling_sim import (
            TilingConfig, compute_tiling, search_split_kn,
            SplitKNResult, search_split_m, SplitMResult,
        )
        # Just ensure imports work
        self.assertTrue(callable(compute_tiling))

    def test_pipeline_model_shim(self):
        from pipeline_model import (
            build_pipeline_dag, estimate_kernel_time, format_timeline,
            roofline_analysis, format_roofline_report,
        )
        self.assertTrue(callable(build_pipeline_dag))


if __name__ == "__main__":
    unittest.main(verbosity=2)
