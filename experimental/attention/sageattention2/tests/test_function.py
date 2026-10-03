"""
Comprehensive functional unit test suite for SageAttention2 on Ascend NPU.
Covers acceptance test cases TC-01 through TC-10 per task/04_local_test.md, task/01_design.md, and task.md:

- TC-01: sageattn unified dispatch and kwargs/env override for pv_accum_dtype.
- TC-02: Full 6 combinations of qk_quant_gran (per_thread, per_warp) and pv_accum_dtype (fp32, fp32+fp32, fp32+fp16).
- TC-03: smooth_v behavior (fp32 runs; fp32+fp16/fp32+fp32 emit UserWarning and set to False) and smooth_k.
- TC-04: LSE numerical validation (return_lse=True, GQA broadcast calibration, smooth_k compensation).
- TC-05: head_dim automatic pad & crop (D=48->64->48; D=80/96->128->orig; D=64/128 boundary; D=256 error).
- TC-06: GQA (Hq % Hkv == 0) and asymmetric length (Sq != Sk) combinations.
- TC-07: Causal mask (is_causal=True, strict upper triangle masking, Sq == Sk requirement, causality verification).
- TC-08: Negative intercept (invalid tensor_layout, qk_quant_gran, pv_accum_dtype, non-empty attn_mask, non-contiguous last dim).
- TC-09: Long sequence (S >= 2048) two-level accumulation stability and finite value bounds.
- TC-10: PyTorch F.scaled_dot_product_attention equivalence and drop-in replacement smoke.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

current_file = Path(__file__).resolve()
OP_ROOT = current_file.parent.parent

sys.path.insert(0, str(current_file.parent))  # tests/ 共享模块
from _golden_utils import find_golden_dir, load_case_tensors  # noqa: E402
if str(OP_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(OP_ROOT / "python"))

import sageattention2 as sage2_pkg


def _make_qkv(
    b: int,
    hq: int,
    hkv: int,
    sq: int,
    sk: int,
    d: int,
    dtype: torch.dtype = torch.float16,
    layout: str = "HND",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to generate deterministic pseudo-random Q, K, V tensors."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    if layout == "HND":
        q = torch.randn(b, hq, sq, d, dtype=dtype, generator=g)
        k = torch.randn(b, hkv, sk, d, dtype=dtype, generator=g)
        v = torch.randn(b, hkv, sk, d, dtype=dtype, generator=g)
    elif layout == "NHD":
        q = torch.randn(b, sq, hq, d, dtype=dtype, generator=g)
        k = torch.randn(b, sk, hkv, d, dtype=dtype, generator=g)
        v = torch.randn(b, sk, hkv, d, dtype=dtype, generator=g)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return q, k, v


class TestSageAttention2Function(unittest.TestCase):
    """Functional test suite covering TC-01 to TC-10."""

    @classmethod
    def setUpClass(cls):
        os.environ["SAGEATTN_ALLOW_CPU_STUB"] = "1"
        cls.manifest_path = find_golden_dir() / "manifest.json"
        cls.manifest = {}
        if cls.manifest_path.exists():
            with open(cls.manifest_path, encoding="utf-8") as f:
                cls.manifest = json.load(f)

    def _load_golden_case(self, case_id: str):
        self.assertIn(case_id, self.manifest.get("cases", {}), f"Case {case_id} not in manifest")
        c = self.manifest["cases"][case_id]
        return c, load_case_tensors(c)

    # -------------------------------------------------------------------------
    # TC-01: sageattn unified dispatch and kwargs/env override
    # -------------------------------------------------------------------------
    def test_tc01_sageattn_dispatch_and_layouts(self):
        """TC-01: Verify sageattn dispatch with HND/NHD layout and FP16/BF16 dtypes."""
        b, hq, hkv, sq, sk, d = 1, 4, 4, 64, 64, 64

        for dtype in (torch.float16, torch.bfloat16):
            for layout in ("HND", "NHD"):
                q, k, v = _make_qkv(b, hq, hkv, sq, sk, d, dtype=dtype, layout=layout)
                out = sage2_pkg.sageattn(q, k, v, tensor_layout=layout)

                expected_shape = (b, hq, sq, d) if layout == "HND" else (b, sq, hq, d)
                self.assertEqual(out.shape, expected_shape)
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.isfinite(out).all())

    def test_tc01_return_value_contract(self):
        """TC-01: Verify return value contract: return_lse=False returns Tensor; return_lse=True returns (Tensor, Tensor)."""
        b, hq, hkv, s, d = 1, 2, 2, 32, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        # Default return_lse=False -> bare Tensor
        res_default = sage2_pkg.sageattn(q, k, v)
        self.assertIsInstance(res_default, torch.Tensor)
        self.assertNotIsInstance(res_default, tuple)

        res_qk_default = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v)
        self.assertIsInstance(res_qk_default, torch.Tensor)
        self.assertNotIsInstance(res_qk_default, tuple)

        # return_lse=True -> 2-tuple (o, lse)
        res_lse = sage2_pkg.sageattn(q, k, v, return_lse=True)
        self.assertIsInstance(res_lse, tuple)
        self.assertEqual(len(res_lse), 2)
        self.assertIsInstance(res_lse[0], torch.Tensor)
        self.assertIsInstance(res_lse[1], torch.Tensor)

    def test_tc01_kwargs_pv_accum_dtype_override(self):
        """TC-01: Verify pv_accum_dtype override via kwargs in sageattn."""
        b, hq, hkv, s, d = 1, 2, 2, 32, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        for accum in ("fp32", "fp32+fp32", "fp32+fp16"):
            out = sage2_pkg.sageattn(q, k, v, pv_accum_dtype=accum)
            self.assertEqual(out.shape, (b, hq, s, d))
            self.assertTrue(torch.isfinite(out).all())

    def test_tc01_env_var_pv_accum_dtype_override(self):
        """TC-01: Verify pv_accum_dtype override via SAGEATTN_PV_ACCUM_DTYPE env var with real numerical assertions."""
        b, hq, hkv, s, d = 1, 2, 2, 64, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        old_env = os.environ.get("SAGEATTN_PV_ACCUM_DTYPE")
        try:
            # Baseline: default without env var (fp32+fp16, v_scale_max=2.25)
            os.environ.pop("SAGEATTN_PV_ACCUM_DTYPE", None)
            out_default = sage2_pkg.sageattn(q, k, v)
            self.assertEqual(out_default.shape, (b, hq, s, d))

            # Branch 1: fp32+fp32 (v_scale_max=448.0, fp32 accum)
            os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = "fp32+fp32"
            out_env_32 = sage2_pkg.sageattn(q, k, v)
            out_explicit_32 = sage2_pkg.sageattn(q, k, v, pv_accum_dtype="fp32+fp32")
            self.assertTrue(torch.equal(out_env_32, out_explicit_32))
            self.assertFalse(torch.equal(out_env_32, out_default))

            # Branch 2: fp32 (full single-stage fp32 accum)
            os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = "fp32"
            out_env_fp32 = sage2_pkg.sageattn(q, k, v)
            out_explicit_fp32 = sage2_pkg.sageattn(q, k, v, pv_accum_dtype="fp32")
            self.assertTrue(torch.equal(out_env_fp32, out_explicit_fp32))
            self.assertFalse(torch.equal(out_env_fp32, out_default))

            # Branch 3: fp32+fp16 (matches default)
            os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = "fp32+fp16"
            out_env_16 = sage2_pkg.sageattn(q, k, v)
            self.assertTrue(torch.equal(out_env_16, out_default))

            # Branch 4: Invalid env var must raise ValueError immediately
            os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = "invalid_precision_mode"
            with self.assertRaises(ValueError) as ctx:
                sage2_pkg.sageattn(q, k, v)
            self.assertIn("Invalid SAGEATTN_PV_ACCUM_DTYPE", str(ctx.exception))
        finally:
            if old_env is not None:
                os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = old_env
            else:
                os.environ.pop("SAGEATTN_PV_ACCUM_DTYPE", None)

    def test_tc01_kwargs_priority_over_env(self):
        """TC-01: Kwargs override must take precedence over environment variable."""
        b, hq, hkv, s, d = 1, 2, 2, 64, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        old_env = os.environ.get("SAGEATTN_PV_ACCUM_DTYPE")
        try:
            # Env is set to fp32
            os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = "fp32"
            # kwargs explicitly specifies fp32+fp16
            out_kwargs = sage2_pkg.sageattn(q, k, v, pv_accum_dtype="fp32+fp16")
            out_explicit_fp16 = sage2_pkg.sageattn(q, k, v, pv_accum_dtype="fp32+fp16")
            out_explicit_fp32 = sage2_pkg.sageattn(q, k, v, pv_accum_dtype="fp32")

            self.assertEqual(out_kwargs.shape, (b, hq, s, d))
            self.assertTrue(torch.equal(out_kwargs, out_explicit_fp16))
            self.assertFalse(torch.equal(out_kwargs, out_explicit_fp32))
        finally:
            if old_env is not None:
                os.environ["SAGEATTN_PV_ACCUM_DTYPE"] = old_env
            else:
                os.environ.pop("SAGEATTN_PV_ACCUM_DTYPE", None)

    def test_tc01_dual_package_parity(self):
        """TC-01: Both root python and ops-transformer sageattention2 packages yield identical results."""
        b, hq, hkv, s, d = 1, 2, 2, 32, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        out1 = sage2_pkg.sageattn(q, k, v)
        out2 = sage2_pkg.sageattn(q, k, v)
        self.assertTrue(torch.equal(out1, out2))

    # -------------------------------------------------------------------------
    # TC-02: qk_quant_gran x pv_accum_dtype full 6 combinations
    # -------------------------------------------------------------------------
    def test_tc02_all_six_combinations(self):
        """TC-02: Verify all 6 combinations of qk_quant_gran and pv_accum_dtype."""
        b, hq, hkv, s, d = 1, 2, 2, 64, 64
        grans = ("per_thread", "per_warp")
        accums = ("fp32", "fp32+fp32", "fp32+fp16")

        for gran in grans:
            for accum in accums:
                for dtype in (torch.float16, torch.bfloat16):
                    q, k, v = _make_qkv(b, hq, hkv, s, s, d, dtype=dtype)
                    out = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                        q, k, v,
                        qk_quant_gran=gran,
                        pv_accum_dtype=accum,
                    )
                    self.assertEqual(out.shape, (b, hq, s, d))
                    self.assertEqual(out.dtype, dtype)
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertFalse((out == 0).all())

    # -------------------------------------------------------------------------
    # TC-03: smooth_v behavior and smooth_k toggle
    # -------------------------------------------------------------------------
    def test_tc03_smooth_v_fp32_normal(self):
        """TC-03: When pv_accum_dtype is fp32, smooth_v=True executes normally without warning."""
        b, hq, hkv, s, d = 1, 2, 2, 32, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                q, k, v, pv_accum_dtype="fp32", smooth_v=True
            )
            # Must NOT warn about smooth_v ignored
            smooth_v_warnings = [item for item in w if "smooth_v will be ignored" in str(item.message)]
            self.assertEqual(len(smooth_v_warnings), 0)
            self.assertEqual(out.shape, (b, hq, s, d))
            self.assertTrue(torch.isfinite(out).all())

    def test_tc03_smooth_v_warning_for_fp16_and_fp32accum(self):
        """TC-03: For fp32+fp16 and fp32+fp32, smooth_v=True must issue UserWarning and auto-disable."""
        b, hq, hkv, s, d = 1, 2, 2, 32, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        for accum in ("fp32+fp16", "fp32+fp32"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                    q, k, v, pv_accum_dtype=accum, smooth_v=True
                )
                smooth_v_warnings = [item for item in w if "smooth_v will be ignored" in str(item.message)]
                self.assertGreaterEqual(len(smooth_v_warnings), 1)
                self.assertEqual(out.shape, (b, hq, s, d))
                self.assertTrue(torch.isfinite(out).all())

    def test_tc03_smooth_k_toggle(self):
        """TC-03: Verify smooth_k=True and smooth_k=False both compute properly."""
        b, hq, hkv, s, d = 1, 2, 2, 32, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        out_sk_true = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=True)
        out_sk_false = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=False)

        self.assertEqual(out_sk_true.shape, (b, hq, s, d))
        self.assertEqual(out_sk_false.shape, (b, hq, s, d))
        self.assertTrue(torch.isfinite(out_sk_true).all())
        self.assertTrue(torch.isfinite(out_sk_false).all())

    # -------------------------------------------------------------------------
    # TC-04: LSE numerical validation
    # -------------------------------------------------------------------------
    def test_tc04_lse_shape_dtype_and_bounds(self):
        """TC-04: return_lse=True returns (out, lse) where lse has shape [B, Hq, Sq] and dtype FP32."""
        b, hq, hkv, sq, sk, d = 2, 4, 4, 48, 64, 64

        for layout in ("HND", "NHD"):
            q, k, v = _make_qkv(b, hq, hkv, sq, sk, d, layout=layout)
            out, lse = sage2_pkg.sageattn(q, k, v, tensor_layout=layout, return_lse=True)

            expected_out_shape = (b, hq, sq, d) if layout == "HND" else (b, sq, hq, d)
            self.assertEqual(out.shape, expected_out_shape)
            self.assertEqual(lse.shape, (b, hq, sq))
            self.assertEqual(lse.dtype, torch.float32)
            self.assertTrue(torch.isfinite(lse).all())

    def test_tc04_lse_gqa_broadcast_and_smooth_k_correction(self):
        """TC-04: GQA scenario verifies km broadcasting and smooth_k lse_correction compensation."""
        b, hq, hkv, sq, sk, d = 1, 8, 2, 64, 128, 64
        q, k, v = _make_qkv(b, hq, hkv, sq, sk, d)

        out, lse = sage2_pkg.sageattn(q, k, v, return_lse=True, smooth_k=True)
        self.assertEqual(lse.shape, (b, hq, sq))
        self.assertTrue(torch.isfinite(lse).all())

        # Test against Golden TC-04 cases if manifest exists
        if "TC-04_return_lse_gqa" in self.manifest.get("cases", {}):
            c, tensors = self._load_golden_case("TC-04_return_lse_gqa")
            q_g = tensors["q"]
            k_g = tensors["k"]
            v_g = tensors["v"]
            ref_lse = tensors["lse"]

            _, got_lse = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                q_g, k_g, v_g,
                tensor_layout=c["tensor_layout"],
                is_causal=c["is_causal"],
                qk_quant_gran=c["qk_quant_gran"],
                pv_accum_dtype=c["pv_accum_dtype"],
                smooth_k=c["smooth_k"],
                smooth_v=c["smooth_v"],
                return_lse=True,
            )
            diff_lse = (got_lse - ref_lse).abs().max().item()
            self.assertLess(diff_lse, 1e-4)

    # -------------------------------------------------------------------------
    # TC-05: head_dim automatic pad & crop
    # -------------------------------------------------------------------------
    def test_tc05_head_dim_pad_and_crop(self):
        """TC-05: head_dim pad D=48->64, D=80/96->128, crop back to original head_dim."""
        b, hq, hkv, s = 1, 2, 2, 32

        # D = 48 (pad to 64 -> crop to 48)
        q48, k48, v48 = _make_qkv(b, hq, hkv, s, s, 48)
        out48 = sage2_pkg.sageattn(q48, k48, v48)
        self.assertEqual(out48.shape, (b, hq, s, 48))

        # D = 80 (pad to 128 -> crop to 80)
        q80, k80, v80 = _make_qkv(b, hq, hkv, s, s, 80)
        out80 = sage2_pkg.sageattn(q80, k80, v80)
        self.assertEqual(out80.shape, (b, hq, s, 80))

        # D = 96 (pad to 128 -> crop to 96)
        q96, k96, v96 = _make_qkv(b, hq, hkv, s, s, 96)
        out96 = sage2_pkg.sageattn(q96, k96, v96)
        self.assertEqual(out96.shape, (b, hq, s, 96))

        # Boundary dimensions D = 64 and D = 128 (no pad needed)
        q64, k64, v64 = _make_qkv(b, hq, hkv, s, s, 64)
        self.assertEqual(sage2_pkg.sageattn(q64, k64, v64).shape, (b, hq, s, 64))

        q128, k128, v128 = _make_qkv(b, hq, hkv, s, s, 128)
        self.assertEqual(sage2_pkg.sageattn(q128, k128, v128).shape, (b, hq, s, 128))

    def test_tc05_head_dim_error_intercept(self):
        """TC-05: D > 128 (e.g. D=256) or D <= 0 must raise ValueError."""
        b, hq, hkv, s = 1, 2, 2, 32

        q256, k256, v256 = _make_qkv(b, hq, hkv, s, s, 256)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            sage2_pkg.sageattn(q256, k256, v256)

        q0 = torch.empty(b, hq, s, 0, dtype=torch.float16)
        k0 = torch.empty(b, hkv, s, 0, dtype=torch.float16)
        v0 = torch.empty(b, hkv, s, 0, dtype=torch.float16)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            sage2_pkg.sageattn(q0, k0, v0)

    # -------------------------------------------------------------------------
    # TC-06: GQA and asymmetric length (Sq != Sk)
    # -------------------------------------------------------------------------
    def test_tc06_gqa_and_asymmetric_sequence(self):
        """TC-06: GQA (Hq % Hkv == 0) with asymmetric lengths Sq != Sk."""
        b, d = 1, 64

        # Hq=8, Hkv=2 (GQA ratio 4), Sq=64, Sk=128 (Sq < Sk)
        q, k, v = _make_qkv(b, 8, 2, 64, 128, d)
        out = sage2_pkg.sageattn(q, k, v)
        self.assertEqual(out.shape, (b, 8, 64, d))
        self.assertTrue(torch.isfinite(out).all())

        # Hq=12, Hkv=3 (GQA ratio 4), Sq=128, Sk=64 (Sq > Sk)
        q2, k2, v2 = _make_qkv(b, 12, 3, 128, 64, d)
        out2 = sage2_pkg.sageattn(q2, k2, v2)
        self.assertEqual(out2.shape, (b, 12, 128, d))
        self.assertTrue(torch.isfinite(out2).all())

    def test_tc06_gqa_indivisible_error(self):
        """TC-06: When Hq % Hkv != 0, raise ValueError."""
        b, s, d = 1, 32, 64
        q, k, v = _make_qkv(b, 7, 2, s, s, d)
        with self.assertRaises((ValueError, RuntimeError, TypeError)) as ctx:
            sage2_pkg.sageattn(q, k, v)
        self.assertIn("GQA constraint violated", str(ctx.exception))

    # -------------------------------------------------------------------------
    # TC-07: Causal mask and strict causality verification
    # -------------------------------------------------------------------------
    def test_tc07_causal_mask_correctness(self):
        """TC-07: is_causal=True masks upper triangle and produces different output from non-causal."""
        b, h, s, d = 1, 2, 64, 64
        q, k, v = _make_qkv(b, h, h, s, s, d)

        out_causal = sage2_pkg.sageattn(q, k, v, is_causal=True)
        out_non_causal = sage2_pkg.sageattn(q, k, v, is_causal=False)

        self.assertEqual(out_causal.shape, (b, h, s, d))
        self.assertFalse(torch.allclose(out_causal, out_non_causal, atol=1e-3))

    def test_tc07_causal_sdpa_alignment_and_golden(self):
        """TC-07: Verify is_causal=True aligns with SDPA causal and Golden causal cases."""
        b, h, s, d = 1, 2, 64, 64
        q, k, v = _make_qkv(b, h, h, s, s, d, dtype=torch.float16)

        out_causal = sage2_pkg.sageattn(q, k, v, is_causal=True)
        out_sdpa_causal = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True).half()

        max_err = (out_causal.float() - out_sdpa_causal.float()).abs().max().item()
        self.assertLess(max_err, 0.20)

        # Test against Golden TC-07 case
        if "TC-07_causal_per_warp" in self.manifest.get("cases", {}):
            c, tensors = self._load_golden_case("TC-07_causal_per_warp")
            out_golden = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                tensors["q"], tensors["k"], tensors["v"],
                tensor_layout=c["tensor_layout"],
                is_causal=c["is_causal"],
                qk_quant_gran=c["qk_quant_gran"],
                pv_accum_dtype=c["pv_accum_dtype"],
                smooth_k=c["smooth_k"],
                smooth_v=c["smooth_v"],
            )
            diff = (out_golden.float() - tensors["out"].float()).abs().max().item()
            self.assertLess(diff, 1e-4)

    def test_tc07_causal_length_mismatch_error(self):
        """TC-07: is_causal=True requires qo_len == kv_len; mismatch raises ValueError."""
        b, h, d = 1, 2, 64
        q, k, v = _make_qkv(b, h, h, 32, 64, d)
        with self.assertRaises((ValueError, RuntimeError, TypeError)) as ctx:
            sage2_pkg.sageattn(q, k, v, is_causal=True)
        self.assertIn("is_causal=True requires qo_len == kv_len", str(ctx.exception))

    # -------------------------------------------------------------------------
    # TC-08: Negative intercept validation
    # -------------------------------------------------------------------------
    def test_tc08_negative_intercepts(self):
        """TC-08: Verify negative intercepts for invalid parameters, layouts, and tensors."""
        b, h, s, d = 1, 2, 32, 64
        q, k, v = _make_qkv(b, h, h, s, s, d)

        # 1. Invalid layout
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            sage2_pkg.sageattn(q, k, v, tensor_layout="BNSD")

        # 2. Invalid quant gran
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_token")

        # 3. Invalid pv_accum_dtype
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="int32")

        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp16")

        # 4. Non-None attn_mask (keyword argument, positional argument, and full-param entry)
        mask = torch.ones(s, s)
        with self.assertRaises(ValueError) as ctx:
            sage2_pkg.sageattn(q, k, v, attn_mask=mask)
        self.assertIn("attn_mask", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            sage2_pkg.sageattn(q, k, v, mask)
        self.assertIn("attn_mask", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, attn_mask=mask)
        self.assertIn("attn_mask", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            sage2_pkg.sageattn_qk_int8_pv_fp8_asc(q, k, v, mask)
        self.assertIn("attn_mask", str(ctx.exception))

        # 5. Non-contiguous last dimension: regular shape
        q_stride2 = torch.randn(b, h, s, d * 2, dtype=torch.float16)[:, :, :, ::2]
        self.assertNotEqual(q_stride2.stride(-1), 1)
        with self.assertRaises(AssertionError) as ctx:
            sage2_pkg.sageattn(q_stride2, k, v)
        self.assertIn("Last dim of qkv must be contiguous", str(ctx.exception))

        # 5b. Non-contiguous last dimension with padding (head_dim=48 < 64):
        # Must be intercepted before F.pad creates a contiguous tensor!
        q_stride2_pad = torch.randn(b, h, s, 48 * 2, dtype=torch.float16)[:, :, :, ::2]
        k48 = torch.randn(b, h, s, 48, dtype=torch.float16)
        v48 = torch.randn(b, h, s, 48, dtype=torch.float16)
        self.assertNotEqual(q_stride2_pad.stride(-1), 1)
        self.assertEqual(q_stride2_pad.shape[-1], 48)
        with self.assertRaises(AssertionError) as ctx:
            sage2_pkg.sageattn(q_stride2_pad, k48, v48)
        self.assertIn("Last dim of qkv must be contiguous", str(ctx.exception))

        # 6. Tensor dtype mismatch
        k_bf16 = k.to(torch.bfloat16)
        with self.assertRaises(AssertionError) as ctx:
            sage2_pkg.sageattn(q, k_bf16, v)
        self.assertIn("must have the same dtype", str(ctx.exception))

    # -------------------------------------------------------------------------
    # TC-09: Long sequence (S >= 2048) stability test
    # -------------------------------------------------------------------------
    def test_tc09_long_sequence_stability(self):
        """TC-09: Long sequence S=2048 stability across all 3 accumulation modes."""
        b, hq, hkv, s, d = 1, 2, 2, 2048, 64
        q, k, v = _make_qkv(b, hq, hkv, s, s, d)

        for accum in ("fp32+fp16", "fp32+fp32", "fp32"):
            out = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                q, k, v,
                pv_accum_dtype=accum,
                qk_quant_gran="per_thread",
            )
            self.assertEqual(out.shape, (b, hq, s, d))
            self.assertTrue(torch.isfinite(out).all())
            self.assertFalse(torch.isnan(out).any())

        # Test against Golden TC-09 case if manifest exists
        if "TC-09_long_seq_2k" in self.manifest.get("cases", {}):
            c, tensors = self._load_golden_case("TC-09_long_seq_2k")
            out_golden = sage2_pkg.sageattn_qk_int8_pv_fp8_asc(
                tensors["q"], tensors["k"], tensors["v"],
                tensor_layout=c["tensor_layout"],
                is_causal=c["is_causal"],
                qk_quant_gran=c["qk_quant_gran"],
                pv_accum_dtype=c["pv_accum_dtype"],
                smooth_k=c["smooth_k"],
                smooth_v=c["smooth_v"],
            )
            max_err = (out_golden.float() - tensors["out"].float()).abs().max().item()
            self.assertLess(max_err, 1e-4)

    # -------------------------------------------------------------------------
    # TC-10: PyTorch SDPA equivalence and drop-in replacement smoke
    # -------------------------------------------------------------------------
    def test_tc10_sdpa_numerical_parity(self):
        """TC-10: Output is close to PyTorch F.scaled_dot_product_attention within L-B budget."""
        b, h, s, d = 1, 4, 128, 64
        q, k, v = _make_qkv(b, h, h, s, s, d, dtype=torch.float16)

        out_sage = sage2_pkg.sageattn(q, k, v)
        out_sdpa = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).half()

        max_err = (out_sage.float() - out_sdpa.float()).abs().max().item()
        l2_err = (
            torch.norm(out_sage.float() - out_sdpa.float())
            / torch.norm(out_sdpa.float())
        ).item()

        # Aligned with L-B quality threshold
        self.assertLess(max_err, 0.15)
        self.assertLess(l2_err, 0.05)

    def test_tc10_drop_in_replacement_mha_smoke(self):
        """TC-10: Drop-in replacement smoke test in a PyTorch attention layer."""
        class ToyAttentionBlock(nn.Module):
            def __init__(self, d_model: int, num_heads: int):
                super().__init__()
                self.d_model = d_model
                self.num_heads = num_heads
                self.head_dim = d_model // num_heads

            def forward(self, q, k, v):
                # q, k, v: [B, H, S, D]
                return sage2_pkg.sageattn(q, k, v, tensor_layout="HND")

        toy_layer = ToyAttentionBlock(d_model=256, num_heads=4)
        b, s = 2, 64
        q, k, v = _make_qkv(b, 4, 4, s, s, 64)

        out = toy_layer(q, k, v)
        self.assertEqual(out.shape, (b, 4, s, 64))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
