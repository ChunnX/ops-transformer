"""
Precision verification test suite for SageAttention2 against Golden dataset.
Verifies L-A (GPU Golden bit-level / tight-tolerance parity) and L-B (FP32 SDPA baseline error ratios)
across all 25 bucketed cases in golden/manifest.json.

Conforms to:
- task/04_local_test.md Section 8 and Quality Gates.
- task/01_design.md Section 1.3 / 2.6.
- task.md M3 precision criteria.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

import torch

current_file = Path(__file__).resolve()
OP_ROOT = current_file.parent.parent

sys.path.insert(0, str(current_file.parent))  # tests/ 共享模块（判据唯一定义点）
from _golden_utils import (  # noqa: E402
    find_golden_dir,
    load_case_tensors,
    l_a_metrics,
    l_b_ratios,
    MATCHED_RATIO_MIN,
    LB_RATIO_LIMITS,
    LB_RATIO_EPS,
    LSE_TOL,
)
if str(OP_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(OP_ROOT / "python"))

import sageattention2 as sage2_pkg


class TestGoldenPrecision(unittest.TestCase):
    """Local precision smoke test suite comparing against golden/manifest.json."""

    @classmethod
    def setUpClass(cls):
        os.environ["SAGEATTN_ALLOW_CPU_STUB"] = "1"
        cls.manifest_path = find_golden_dir() / "manifest.json"
        if not cls.manifest_path.exists():
            raise FileNotFoundError(f"Golden manifest not found at {cls.manifest_path}")

        with open(cls.manifest_path, encoding="utf-8") as f:
            cls.manifest = json.load(f)

        cls.cases = cls.manifest.get("cases", {})
        if len(cls.cases) != 25:
            raise ValueError(f"Expected 25 golden cases, but found {len(cls.cases)}")

    def _execute_case(self, case_meta: dict):
        tensors = load_case_tensors(case_meta)

        q = tensors["q"]
        k = tensors["k"]
        v = tensors["v"]

        out, lse = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
            q=q,
            k=k,
            v=v,
            tensor_layout=case_meta["tensor_layout"],
            is_causal=case_meta["is_causal"],
            qk_quant_gran=case_meta["qk_quant_gran"],
            pv_accum_dtype=case_meta["pv_accum_dtype"],
            smooth_k=case_meta["smooth_k"],
            smooth_v=case_meta["smooth_v"],
            return_lse=True,
        )

        return out, lse, tensors

    def test_all_25_cases_l_a_precision(self):
        """L-A precision gate: matched_ratio >= 0.99, max_abs_error <= limit, lse_diff <= 1e-3."""
        for case_id, c in self.cases.items():
            with self.subTest(case_id=case_id, test="L-A"):
                out, lse, tensors = self._execute_case(c)
                ref_out = tensors["out"]
                ref_lse = tensors["lse"]

                is_bf16 = c.get("dtype") in ("bfloat16", "bf16")
                m = l_a_metrics(out, ref_out, is_bf16)

                # 1. Output tensor shape and finiteness
                self.assertEqual(out.shape, ref_out.shape, f"Shape mismatch for {case_id}")
                self.assertTrue(torch.isfinite(out).all(), f"Non-finite values in output for {case_id}")

                # 2. Output tensor error against Golden（判据唯一定义点 _golden_utils）
                self.assertGreaterEqual(
                    m["matched_ratio"],
                    MATCHED_RATIO_MIN,
                    f"[{case_id}] L-A matched ratio {m['matched_ratio']:.6f} < {MATCHED_RATIO_MIN} "
                    f"(atol={m['atol']:.6g}, max_abs={m['max_abs_error']:.6g})",
                )
                self.assertLessEqual(
                    m["max_abs_error"],
                    m["max_abs_limit"],
                    f"[{case_id}] L-A max abs error {m['max_abs_error']:.6g} exceeds limit {m['max_abs_limit']}",
                )

                # 3. LSE verification
                self.assertEqual(lse.shape, ref_lse.shape, f"LSE shape mismatch for {case_id}")
                self.assertTrue(torch.isfinite(lse).all(), f"Non-finite values in LSE for {case_id}")
                max_lse_diff = (lse.float() - ref_lse.float()).abs().max().item()
                self.assertLessEqual(
                    max_lse_diff,
                    LSE_TOL,
                    f"[{case_id}] LSE difference {max_lse_diff:.6g} exceeds tolerance {LSE_TOL}",
                )

    def test_all_25_cases_l_b_precision(self):
        """L-B precision gate: ratio_max <= 2.0, ratio_mean <= 1.2, ratio_rmse <= 1.2 vs FP32 SDPA."""
        for case_id, c in self.cases.items():
            with self.subTest(case_id=case_id, test="L-B"):
                out, _, tensors = self._execute_case(c)
                sdpa_ref = tensors["out_sdpa_fp32"]
                r = l_b_ratios(out, sdpa_ref, c)

                self.assertLessEqual(
                    r["ratio_max"],
                    LB_RATIO_LIMITS[0] + LB_RATIO_EPS,
                    f"[{case_id}] L-B ratio_max {r['ratio_max']:.4f} > {LB_RATIO_LIMITS[0]}",
                )
                self.assertLessEqual(
                    r["ratio_mean"],
                    LB_RATIO_LIMITS[1] + LB_RATIO_EPS,
                    f"[{case_id}] L-B ratio_mean {r['ratio_mean']:.4f} > {LB_RATIO_LIMITS[1]}",
                )
                self.assertLessEqual(
                    r["ratio_rmse"],
                    LB_RATIO_LIMITS[2] + LB_RATIO_EPS,
                    f"[{case_id}] L-B ratio_rmse {r['ratio_rmse']:.4f} > {LB_RATIO_LIMITS[2]}",
                )

    def test_benchmark_subset_p01_p07(self):
        """Explicit verification for benchmark subset P-01 through P-07."""
        p_cases = [k for k in self.cases if k.startswith("P-")]
        self.assertEqual(len(p_cases), 7)
        for case_id in sorted(p_cases):
            case_info = self.cases[case_id]
            out, _, tensors = self._execute_case(case_info)
            ref_out = tensors["out"]
            diff = (out.float() - ref_out.float()).abs().max().item()
            # 容差按输入 dtype 分档（收敛轮 A/CLEANUP-6：bf16 有效位 ~8bit，
            # 与 test_l_error_ratios 的 2^-6 口径一致；孪生位级一致时 diff 应为 0）
            is_bf16 = case_info.get("dtype") in ("bfloat16", "bf16")
            tol = 2.0 ** (-6) if is_bf16 else 2.0 ** (-9)
            self.assertLess(diff, tol, f"P-case {case_id} exceeded error limit {tol}: {diff}")

    def test_functional_subset_tc01_tc10(self):
        """Explicit verification for functional subset TC-01 through TC-10."""
        tc_cases = [k for k in self.cases if k.startswith("TC-")]
        self.assertGreaterEqual(len(tc_cases), 10)
        for case_id in sorted(tc_cases):
            out, _, tensors = self._execute_case(self.cases[case_id])
            ref_out = tensors["out"]
            diff = (out.float() - ref_out.float()).abs().max().item()
            self.assertLess(diff, 1e-4, f"TC-case {case_id} exceeded error limit: {diff}")

    def test_sageattn_dispatch_precision_equivalence(self):
        """Verify calling top-level sageattn API matches sageattn_qk_int8_pv_fp8_asc precision."""
        case_id = "P-01_per_thread_fp32+fp16_fp16_HND"
        c = self.cases[case_id]
        tensors = load_case_tensors(c)

        out_direct, lse_direct = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
            tensors["q"], tensors["k"], tensors["v"],
            tensor_layout=c["tensor_layout"],
            is_causal=c["is_causal"],
            qk_quant_gran=c["qk_quant_gran"],
            pv_accum_dtype=c["pv_accum_dtype"],
            smooth_k=c["smooth_k"],
            smooth_v=c["smooth_v"],
            return_lse=True,
        )

        out_routed, lse_routed = sage2_pkg.sageattn(
            tensors["q"], tensors["k"], tensors["v"],
            tensor_layout=c["tensor_layout"],
            is_causal=c["is_causal"],
            qk_quant_gran=c["qk_quant_gran"],
            pv_accum_dtype=c["pv_accum_dtype"],
            smooth_k=c["smooth_k"],
            smooth_v=c["smooth_v"],
            return_lse=True,
        )

        self.assertTrue(torch.equal(out_direct, out_routed))
        self.assertTrue(torch.equal(lse_direct, lse_routed))


if __name__ == "__main__":
    unittest.main()
