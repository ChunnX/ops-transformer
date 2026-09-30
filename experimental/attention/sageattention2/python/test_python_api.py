"""Acceptance unit test suite for SageAttention2 on Ascend 950PR."""
from __future__ import annotations

import math
import os
import sys
import unittest
import warnings
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock

import torch
import torch.nn.functional as F

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from sageattention2 import replace_sdpa, run, sageattn, sageattn_qk_int8_pv_fp8_asc


class TestSageAttention2PythonAPI(unittest.TestCase):
    # PyTorch / Python API acceptance verification test suite.

    def setUp(self) -> None:
        torch.random.manual_seed(42)

    def _build_test_tensors(self, b: int, h: int, s: int, d: int,
                            dtype: torch.dtype = torch.float16,
                            layout: str = "HND") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor_dims = (b, h, s, d) if layout == "HND" else (b, s, h, d)
        t_list = [torch.randn(tensor_dims, dtype=dtype) for _ in range(3)]
        return t_list[0], t_list[1], t_list[2]

    def _assert_tensor_valid(self, tensor_val: torch.Tensor, expected_shape: Tuple[int, ...],
                             expected_type: Optional[torch.dtype] = None) -> None:
        self.assertEqual(tuple(tensor_val.shape), tuple(expected_shape))
        if expected_type is not None:
            self.assertEqual(tensor_val.dtype, expected_type)
        self.assertFalse(torch.isnan(tensor_val).any().item())
        self.assertFalse(torch.isinf(tensor_val).any().item())

    def test_tc00_signatures_and_core_module(self) -> None:
    # TC-00: Verify function signatures, default arguments, and core module exports.
        import inspect
        from sageattention2 import core as s2_core

        self.assertIs(s2_core.sageattn, sageattn)
        self.assertIs(s2_core.sageattn_qk_int8_pv_fp8_asc, sageattn_qk_int8_pv_fp8_asc)
        self.assertIs(s2_core.replace_sdpa, replace_sdpa)
        self.assertIs(s2_core.run, run)

        sig_sage = inspect.signature(sageattn)
        params_sage = sig_sage.parameters
        self.assertEqual(params_sage["tensor_layout"].default, "HND")
        self.assertEqual(params_sage["is_causal"].default, False)
        self.assertIsNone(params_sage["sm_scale"].default)
        self.assertEqual(params_sage["return_lse"].default, False)

        sig_asc = inspect.signature(sageattn_qk_int8_pv_fp8_asc)
        params_asc = sig_asc.parameters
        self.assertEqual(params_asc["tensor_layout"].default, "HND")
        self.assertEqual(params_asc["is_causal"].default, False)
        self.assertEqual(params_asc["qk_quant_gran"].default, "per_thread")
        self.assertIsNone(params_asc["sm_scale"].default)
        self.assertEqual(params_asc["pv_accum_dtype"].default, "fp32+fp16")
        self.assertEqual(params_asc["smooth_k"].default, True)
        self.assertEqual(params_asc["smooth_v"].default, False)
        self.assertEqual(params_asc["return_lse"].default, False)

    def test_tc01_basic_functionality_and_dtypes(self) -> None:
    # TC-01: Verify FP16 and BF16 dtypes, causal/non-causal, and HND/NHD layouts.
        b, h, s, d = 2, 4, 128, 64

        for dtype in (torch.float16, torch.bfloat16):
            for layout in ("HND", "NHD"):
                for is_causal in (False, True):
                    with self.subTest(dtype=dtype, layout=layout, is_causal=is_causal):
                        q, k, v = self._build_test_tensors(b, h, s, d, dtype=dtype, layout=layout)
                        expected_shape = (b, h, s, d) if layout == "HND" else (b, s, h, d)

                        out_auto = sageattn(q, k, v, tensor_layout=layout, is_causal=is_causal)
                        self._assert_tensor_valid(out_auto, expected_shape, dtype)

                        out_asc = sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout=layout, is_causal=is_causal, pv_accum_dtype="fp32+fp16")
                        self._assert_tensor_valid(out_asc, expected_shape, dtype)

    def test_tc02_quant_gran_and_accum_dtypes(self) -> None:
    # TC-02: Verify per_thread/per_warp x fp32/fp32+fp32/fp32+fp16 combinations.
        b, h, s, d = 1, 4, 64, 64
        q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)
        scale_val = 1.0 / math.sqrt(d)
        truth = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale_val)
        results = {}
        for gran in ("per_thread", "per_warp"):
            for accum in ("fp32+fp16", "fp32+fp32", "fp32"):
                with self.subTest(gran=gran, accum=accum):
                    out = sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran=gran, pv_accum_dtype=accum)
                    self._assert_tensor_valid(out, (b, h, s, d), torch.float16)
                    self.assertLess((out.float() - truth).abs().max().item(), 0.05)
                    results[(gran, accum)] = out

        for gran in ("per_thread", "per_warp"):
            self.assertGreater(
                (results[(gran, "fp32+fp16")] - results[(gran, "fp32")]).abs().max().item(), 0.0,
                f"{gran}: fp32+fp16 aliased fp32")
        for accum in ("fp32+fp16", "fp32+fp32", "fp32"):
            self.assertGreater(
                (results[("per_thread", accum)] - results[("per_warp", accum)]).abs().max().item(), 0.0,
                f"{accum}: per_warp aliased per_thread")

        for gran in ("per_thread", "per_warp"):
            self.assertTrue(
                torch.equal(results[(gran, "fp32")], results[(gran, "fp32+fp32")]),
                f"{gran}: fp32 and fp32+fp32 must share one datapath")

    def test_tc11_env_var_override_of_hidden_params(self) -> None:
    # TC-11: SAGEATTN_* env vars retune the sunk path; explicit arguments win.
        b, h, s, d = 1, 4, 64, 64
        q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)
        env = {"SAGEATTN_PV_ACCUM_DTYPE": "fp32", "SAGEATTN_QK_QUANT_GRAN": "per_warp"}
        with mock.patch.dict(os.environ, env):
            self.assertTrue(
                torch.equal(sageattn(q, k, v), sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_warp", pv_accum_dtype="fp32")),
                "env override did not reach the datapath")
            self.assertTrue(
                torch.equal(
                    sageattn(q, k, v, pv_accum_dtype="fp32+fp16", qk_quant_gran="per_thread"),
                    sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16")),
                "explicit argument lost to the environment")
            with mock.patch.dict(os.environ, {"SAGEATTN_PV_ACCUM_DTYPE": "fp16+fp16"}, clear=True):
                with self.assertRaises(ValueError):
                    sageattn(q, k, v)
            with mock.patch.dict(os.environ, {"SAGEATTN_QK_QUANT_GRAN": "per_block"}, clear=True):
                with self.assertRaises(ValueError):
                    sageattn(q, k, v)
            with mock.patch.dict(os.environ, {"SAGEATTN_PV_ACCUM_DTYPE": ""}, clear=True):
                self.assertTrue(
                    torch.equal(sageattn(q, k, v), sageattn_qk_int8_pv_fp8_asc(q, k, v)),
                    "empty environment value was treated as a setting")

    def test_tc11b_env_override_survives_plan_cache(self) -> None:
    # TC-11b: The cached plan is keyed by the resolved parameters.
        b, h, s, d = 1, 4, 64, 64
        q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)
        with mock.patch.dict(os.environ, {}, clear=True):
            default_out = sageattn(q, k, v)
        with mock.patch.dict(os.environ, {"SAGEATTN_QK_QUANT_GRAN": "per_warp"}, clear=True):
            widened_out = sageattn(q, k, v)
        with mock.patch.dict(os.environ, {}, clear=True):
            after_out = sageattn(q, k, v)
        self.assertTrue(torch.equal(default_out, after_out), "plan cache leaked the overridden granularity")
        self.assertFalse(torch.equal(default_out, widened_out), "per_warp produced the per_thread result")

    def test_tc03_smooth_k_and_smooth_v_behavior(self) -> None:
    # TC-03: Verify the smooth_k toggle and that smooth_v is honored under fp32, ignored elsewhere.
        b, h, s, d = 1, 4, 64, 64
        q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)

        out_smooth = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=True)
        out_no_smooth = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=False)
        self.assertEqual(out_smooth.shape, (b, h, s, d))
        self.assertEqual(out_no_smooth.shape, (b, h, s, d))

        for accum in ("fp32+fp16", "fp32+fp32"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_v = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=True, pv_accum_dtype=accum)
                self.assertEqual(out_v.shape, (b, h, s, d))
                self.assertTrue(
                    any("smooth_v" in str(item.message) and "ignored" in str(item.message) for item in w),
                    f"expected smooth_v ignore warning for {accum}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out_v32 = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=True, pv_accum_dtype="fp32")
            self.assertEqual(out_v32.shape, (b, h, s, d))
            self.assertFalse(any("smooth_v" in str(item.message) and "ignored" in str(item.message) for item in w))

        out_v32_off = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=False, pv_accum_dtype="fp32")
        truth = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=1.0 / math.sqrt(d))
        self.assertGreater((out_v32 - out_v32_off).abs().max().item(), 0.0, "smooth_v aliased its off state")
        for name, out in (("on", out_v32), ("off", out_v32_off)):
            self.assertLess((out.float() - truth).abs().max().item(), 0.05, f"smooth_v={name} out of envelope")

    def test_non_contiguous_and_strides(self) -> None:
    # Verify handling of non-contiguous input tensors with stride(-1)==1.
        q_raw, k_raw, v_raw = self._build_test_tensors(2, 4, 128, 64, dtype=torch.float16)

        q_sliced = q_raw[:, :, ::2, :]
        k_sliced = k_raw[:, :, ::2, :]
        v_sliced = v_raw[:, :, ::2, :]

        self.assertFalse(q_sliced.is_contiguous())
        self.assertEqual(q_sliced.stride(-1), 1)

        out_sliced = sageattn(q_sliced, k_sliced, v_sliced, tensor_layout="HND")
        self._assert_tensor_valid(out_sliced, (2, 4, 64, 64), torch.float16)

        q_perm = q_raw.permute(1, 0, 2, 3)
        k_perm = k_raw.permute(1, 0, 2, 3)
        v_perm = v_raw.permute(1, 0, 2, 3)

        self.assertFalse(q_perm.is_contiguous())
        self.assertEqual(q_perm.stride(-1), 1)

        out_perm = sageattn(q_perm, k_perm, v_perm, tensor_layout="HND")
        self._assert_tensor_valid(out_perm, (4, 2, 128, 64))

        q_asym, _, _ = self._build_test_tensors(1, 4, 32, 64, dtype=torch.float16)
        k_asym, v_asym, _ = self._build_test_tensors(1, 4, 96, 64, dtype=torch.float16)

        out_asym = sageattn(q_asym, k_asym, v_asym, is_causal=False)
        self.assertEqual(out_asym.shape, (1, 4, 32, 64))

    def test_tc04_lse_export_and_gqa_broadcast(self) -> None:
    # TC-04: Verify LSE output tuple, shape [B, Hq, Sq], and GQA broadcast correction.
        b, hq, hk, sq, sk, d = 2, 8, 2, 64, 64, 64
        q, _, _ = self._build_test_tensors(b, hq, sq, d, dtype=torch.float16)
        k, v, _ = self._build_test_tensors(b, hk, sk, d, dtype=torch.float16)

        out, lse = sageattn(q, k, v, tensor_layout="HND", return_lse=True)
        self._assert_tensor_valid(out, (b, hq, sq, d))
        self._assert_tensor_valid(lse, (b, hq, sq), torch.float32)

        out_no_smooth, lse_no_smooth = sageattn_qk_int8_pv_fp8_asc(
            q, k, v, tensor_layout="HND", smooth_k=False, return_lse=True)
        self.assertEqual(lse_no_smooth.shape, (b, hq, sq))
        self.assertFalse(torch.isnan(lse_no_smooth).any())

        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()

        out_nhd, lse_nhd = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD", return_lse=True)
        self._assert_tensor_valid(out_nhd, (b, sq, hq, d))
        self._assert_tensor_valid(lse_nhd, (b, hq, sq), torch.float32)

    def test_tc01_dispatch_targets_fp8_pv(self) -> None:
    # TC-01: sageattn must dispatch to the FP8 PV path with fp32+fp16 by default.
        from sageattention2 import _ops
        if _ops._FORWARD_FN is None:
            self.skipTest("NPU extension not loaded; dispatch interception requires the device build")
        captured = {}
        original = _ops._FORWARD_FN

        def spy(*args, **kwargs):
            captured["pv_accum_dtype"] = args[7]
            captured["qk_quant_gran"] = args[5]
            return original(*args, **kwargs)

        q = torch.randn(1, 2, 128, 64, dtype=torch.float16, device="npu")
        k = torch.randn(1, 2, 128, 64, dtype=torch.float16, device="npu")
        v = torch.randn(1, 2, 128, 64, dtype=torch.float16, device="npu")
        _ops._FORWARD_FN = spy
        try:
            sageattn(q, k, v)
        finally:
            _ops._FORWARD_FN = original
        self.assertEqual(captured.get("pv_accum_dtype"), "fp32+fp16")
        self.assertEqual(captured.get("qk_quant_gran"), "per_thread")

    @staticmethod
    def _open_source_pv_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            sm_scale: float, is_causal: bool,
                            q_group: int = 4, k_group: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-source SageAttention2 sm89 FP8-path numeric emulation."""
        b, hq, sq, d = [int(dim) for dim in q.shape]
        sk = int(k.shape[2])

        def quantize_token_matrix(inp: torch.Tensor, grp_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
            pad_val = int((-inp.shape[-2]) % grp_size)
            padded = torch.nn.functional.pad(inp.to(torch.float32), (0, 0, 0, pad_val))
            grouped = padded.reshape(*inp.shape[:-2], -1, grp_size, inp.shape[-1])
            peak_amp = grouped.abs().amax(dim=(-2, -1), keepdim=True)
            scale_fac = torch.where(peak_amp > 0, peak_amp / 127.0, torch.ones_like(peak_amp))
            q_res = (grouped / scale_fac).round().clamp(-127, 127)
            expanded_sc = scale_fac.expand(-1, -1, -1, grp_size, -1).reshape(*inp.shape[:-2], -1)[..., :inp.shape[-2], None]
            return q_res.reshape(*inp.shape[:-2], -1, inp.shape[-1])[..., :inp.shape[-2], :], expanded_sc

        k_mean_vec = k.to(torch.float32).mean(dim=-2, keepdim=True)
        q8, qs = quantize_token_matrix(q, q_group)
        k8, ks = quantize_token_matrix(k.to(torch.float32) - k_mean_vec, k_group)
        v_peak = v.to(torch.float32).abs().amax(dim=-2, keepdim=True)
        v_scale_unit = torch.where(v_peak > 0, v_peak / 448.0, torch.ones_like(v_peak))
        dequant_v = (v.to(torch.float32) / v_scale_unit).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32) * v_scale_unit

        out_mat = torch.zeros([b, hq, sq, d], dtype=torch.float32)
        cur_m = torch.full([b, hq, sq, 1], -torch.inf)
        cur_den = torch.zeros_like(cur_m)
        log448_c = math.log(448.0)

        for step_j in range(0, sk, 128):
            step_je = min(step_j + 128, sk)
            s_prod = q8.to(torch.float32) @ k8[:, :, step_j:step_je].to(torch.float32).transpose(-2, -1)
            qk_val = (s_prod * (qs * ks[:, :, step_j:step_je].transpose(-2, -1)) * sm_scale).to(torch.float16)
            if bool(is_causal) is True:
                c_mask = torch.arange(step_j, step_je)[None, :] > torch.arange(sq)[:, None]
                qk_val = qk_val.masked_fill(c_mask, -torch.inf)
            b_max = qk_val.to(torch.float32).amax(dim=-1, keepdim=True) - log448_c
            nxt_m = torch.maximum(b_max, cur_m)
            alpha_term = torch.where(torch.isfinite(cur_m - nxt_m), (cur_m - nxt_m).exp(), torch.zeros_like(cur_m - nxt_m))
            p_val = (qk_val.to(torch.float32) - nxt_m).exp()
            p8_val = p_val.clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
            cur_den = cur_den * alpha_term + p8_val.sum(dim=-1, keepdim=True)
            pv_sub = (p8_val @ dequant_v[:, :, step_j:step_je]).to(torch.float16).to(torch.float32)
            out_mat = out_mat * alpha_term + pv_sub
            cur_m = nxt_m

        res_final = (out_mat / cur_den).to(q.dtype).to(torch.float32)
        lse_final = (cur_m.squeeze(-1) + cur_den.squeeze(-1).log()) + (q.to(torch.float32) * k_mean_vec).sum(dim=-1) * sm_scale
        return res_final, lse_final

    def test_tc04b_lb_precision_gates(self) -> None:
    # TC-04 (L-B layer): error ratios vs FP32 SDPA within max<=2 / mean<=1.2 / RMSE<=1.2.
        try:
            from sageattention2.reference import attention as cpu_golden
        except ImportError:
            self.skipTest("reference golden unavailable")
        cases = [
            (2, 512, 64, torch.float16, False, (4, 16)),
            (2, 1024, 128, torch.float16, True, (4, 16)),
            (1, 512, 128, torch.bfloat16, False, (4, 16)),
            (2, 512, 128, torch.float16, False, (32, 64)),
        ]
        for heads, seq, dim, dtype, causal, (qg, kg) in cases:
            with self.subTest(heads=heads, seq=seq, dim=dim, dtype=dtype, causal=causal, groups=(qg, kg)):
                q, k, v = self._build_test_tensors(1, heads, seq, dim, dtype=dtype)
                scale_val = 1.0 / math.sqrt(dim)

                npu_out, npu_lse = cpu_golden(
                    q, k, v, sm_scale=scale_val, is_causal=causal,
                    qk_quant_gran="per_thread" if qg == 4 else "per_warp",
                    return_lse=True)
                gpu_out, gpu_lse = self._open_source_pv_fp8(q, k, v, scale_val, causal, qg, kg)

                truth = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal, scale=scale_val)
                err = (npu_out.float() - truth).abs()
                berr = (gpu_out - truth).abs()
                ratio_max = (err.amax() / berr.amax().clamp_min(1e-9)).item()
                ratio_mean = (err.mean() / berr.mean().clamp_min(1e-12)).item()
                ratio_rmse = (err.pow(2).mean().sqrt() / berr.pow(2).mean().sqrt().clamp_min(1e-12)).item()
                self.assertLessEqual(ratio_max, 2.0, f"L-B max ratio {ratio_max:.3f}")
                self.assertLessEqual(ratio_mean, 1.2, f"L-B mean ratio {ratio_mean:.3f}")
                self.assertLessEqual(ratio_rmse, 1.2, f"L-B rmse ratio {ratio_rmse:.3f}")

                lse_rel = ((npu_lse - gpu_lse).abs() / npu_lse.abs().clamp_min(1.0)).amax().item()
                self.assertLessEqual(lse_rel, 5e-2, f"LSE relative deviation {lse_rel:.4f}")

    def test_tc05_head_dim_auto_pad_and_crop(self) -> None:
    # TC-05: Verify auto-padding for D <= 64 -> 64, 64 < D <= 128 -> 128, and cropping.
        b, h, s = 1, 4, 64

        q48, k48, v48 = self._build_test_tensors(b, h, s, 48, dtype=torch.float16)
        out48 = sageattn(q48, k48, v48)
        self._assert_tensor_valid(out48, (b, h, s, 48))

        q96, k96, v96 = self._build_test_tensors(b, h, s, 96, dtype=torch.float16)
        out96 = sageattn(q96, k96, v96)
        self._assert_tensor_valid(out96, (b, h, s, 96))

        q32, k32, v32 = self._build_test_tensors(b, h, s, 32, dtype=torch.float16)
        out32 = sageattn(q32, k32, v32)
        self.assertEqual(out32.shape, (b, h, s, 32))

        q64, _, _ = self._build_test_tensors(b, h, s, 64, dtype=torch.float16)
        out64 = sageattn(q64, q64, q64)
        self.assertEqual(out64.shape, (b, h, s, 64))

        q128, _, _ = self._build_test_tensors(b, h, s, 128, dtype=torch.float16)
        out128 = sageattn(q128, q128, q128)
        self.assertEqual(out128.shape, (b, h, s, 128))

        q144, _, _ = self._build_test_tensors(b, h, s, 144, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "Only head_dim <= 128 is supported"):
            sageattn(q144, q144, q144)

    def test_tc06_gqa_and_asymmetric_sequence_lengths(self) -> None:
    # TC-06: Verify GQA (Hq % Hk == 0) with asymmetric sequence lengths (Sq != Sk).
        b, hq, hk, sq, sk, d = 2, 8, 2, 32, 96, 64
        q, _, _ = self._build_test_tensors(b, hq, sq, d, dtype=torch.float16)
        k, v, _ = self._build_test_tensors(b, hk, sk, d, dtype=torch.float16)

        out_hnd = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        self.assertEqual(out_hnd.shape, (b, hq, sq, d))

        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()
        out_nhd = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD", is_causal=False)
        self.assertEqual(out_nhd.shape, (b, sq, hq, d))

    def test_tc07_causal_attention(self) -> None:
    # TC-07: Verify causal attention masking (future key positions masked).
        b, h, d = 1, 2, 64
        for s in [64, 128, 256]:
            with self.subTest(seq_len=s):
                q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)

                out_causal = sageattn(q, k, v, is_causal=True)
                self._assert_tensor_valid(out_causal, (b, h, s, d))

                k_modified = torch.clone(k)
                v_modified = torch.clone(v)
                k_modified[:, :, 1:, :] = torch.randn_like(k_modified[:, :, 1:, :])
                v_modified[:, :, 1:, :] = torch.randn_like(v_modified[:, :, 1:, :])
                out_modified = sageattn(q, k_modified, v_modified, is_causal=True)
                diff = (out_causal[:, :, 0, :] - out_modified[:, :, 0, :]).abs().max()
                self.assertLess(diff.item(), 0.05)

                boundary = 32
                k_future = torch.clone(k)
                v_future = torch.clone(v)
                k_future[:, :, boundary:, :] = torch.randn_like(k_future[:, :, boundary:, :])
                v_future[:, :, boundary:, :] = torch.randn_like(v_future[:, :, boundary:, :])
                out_exact = sageattn(q, k, v, is_causal=True, smooth_k=False)
                out_exact_mod = sageattn(q, k_future, v_future, is_causal=True, smooth_k=False)
                diff_exact = (out_exact[:, :, :boundary, :] - out_exact_mod[:, :, :boundary, :]).abs().max()
                self.assertEqual(diff_exact.item(), 0.0)

                mid = (s // 2) // boundary * boundary
                self.assertGreaterEqual(mid, boundary)
                k_mid = torch.clone(k)
                v_mid = torch.clone(v)
                k_mid[:, :, mid:, :] = torch.randn_like(k_mid[:, :, mid:, :])
                v_mid[:, :, mid:, :] = torch.randn_like(v_mid[:, :, mid:, :])
                out_mid = sageattn(q, k_mid, v_mid, is_causal=True, smooth_k=False)
                diff_mid = (out_exact[:, :, :mid, :] - out_mid[:, :, :mid, :]).abs().max()
                self.assertEqual(diff_mid.item(), 0.0)

    def test_tc08_strict_exception_interception(self) -> None:
    # TC-08: Verify strict error checking for invalid arguments, layouts, and shapes.
        q, k, v = self._build_test_tensors(1, 4, 64, 64, dtype=torch.float16)

        q_bad_stride = q.transpose(-2, -1)
        with self.assertRaises(ValueError):
            sageattn(q_bad_stride, k, v)

        with self.assertRaises(ValueError):
            sageattn(q, k, v, tensor_layout="BSND")

        for gran in ("per_head", "per_token", "per_block"):
            with self.subTest(qk_quant_gran=gran):
                with self.assertRaises(ValueError):
                    sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran=gran)

        with self.assertRaises(ValueError):
            sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="int32")

        q_causal, _, _ = self._build_test_tensors(1, 4, 64, 64, dtype=torch.float16)
        k_causal, _, _ = self._build_test_tensors(1, 4, 32, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q_causal, k_causal, k_causal, is_causal=True)

        dummy_mask = torch.zeros(1, 1, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q, k, v, attn_mask=dummy_mask)

        q_gqa, _, _ = self._build_test_tensors(1, 7, 64, 64, dtype=torch.float16)
        k_gqa, _, _ = self._build_test_tensors(1, 2, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q_gqa, k_gqa, k_gqa)

        q_3d = torch.randn(4, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q_3d, q_3d, q_3d)

        k_bf16 = k.to(torch.bfloat16)
        with self.assertRaises(ValueError):
            sageattn(q, k_bf16, v)

        q_fp32 = q.float()
        with self.assertRaises(ValueError):
            sageattn(q_fp32, q_fp32, q_fp32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=True, pv_accum_dtype="fp32+fp16")
            self.assertTrue(any("smooth_v is ignored" in str(item.message) for item in w))

    def test_tc09_long_sequence_handling(self) -> None:
    # TC-09: Verify long sequence execution (S>=8K; 16K tier on NPU).
        is_npu = hasattr(torch, "npu") and torch.npu.is_available()
        device_str = "npu" if is_npu else "cpu"
        heads = 4 if is_npu else 2
        if bool(is_npu) is True:
            for s in (4096, 8192, 16384):
                with self.subTest(seq_len=s):
                    q = torch.randn(1, heads, s, 128, dtype=torch.float16, device=device_str)
                    k = torch.randn(1, heads, s, 128, dtype=torch.float16, device=device_str)
                    v = torch.randn(1, heads, s, 128, dtype=torch.float16, device=device_str)
                    out = sageattn(q, k, v, tensor_layout="HND")
                    self._assert_tensor_valid(out, (1, heads, s, 128))
            return

        torch.random.manual_seed(909)
        for s, causal, both_accums in ((4096, False, True), (8192, True, True)):
            with self.subTest(seq_len=s, causal=causal):
                q = torch.randn(1, 1, s, 128, dtype=torch.float16)
                k = torch.randn(1, 1, s, 128, dtype=torch.float16)
                v = torch.randn(1, 1, s, 128, dtype=torch.float16)
                kwargs = {"is_causal": causal, "return_lse": True}
                if bool(both_accums) is True:
                    folded = sageattn_qk_int8_pv_fp8_asc(q, k, v, is_causal=causal, pv_accum_dtype="fp32+fp32")
                out, lse = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp16", **kwargs)
                self.assertEqual(out.shape, (1, 1, s, 128))
                self.assertFalse(torch.isnan(out).any(), f"NaN at S={s}, causal={causal}")
                self.assertTrue(torch.isfinite(out).all(), f"non-finite output at S={s}")
                if bool(both_accums) is True:
                    self.assertLess(
                        (out.float() - folded.float()).abs().max().item(), 5e-2,
                        f"fp32+fp16 fold diverged from fp32+fp32 at S={s}")
                self.assertEqual(lse.shape, (1, 1, s))
                self.assertEqual(lse.dtype, torch.float32)
                self.assertTrue(torch.isfinite(lse).all(), f"non-finite LSE at S={s}")

    def test_tc10_sdpa_seamless_substitution(self) -> None:
    # TC-10: Verify replace_sdpa monkey-patching and restoring F.scaled_dot_product_attention.
        b, h, s, d = 1, 4, 64, 64
        q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)
        orig_sdpa = F.scaled_dot_product_attention

        try:
            replace_sdpa(enable=True)
            self.assertNotEqual(F.scaled_dot_product_attention, orig_sdpa)

            out_sdpa = F.scaled_dot_product_attention(q, k, v)
            self._assert_tensor_valid(out_sdpa, (b, h, s, d), torch.float16)

            out_sdpa_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            self.assertEqual(out_sdpa_causal.shape, (b, h, s, d))

            custom_scale = 0.125
            out_sdpa_scale = F.scaled_dot_product_attention(q, k, v, scale=custom_scale)
            self.assertEqual(out_sdpa_scale.shape, (b, h, s, d))

            dummy_mask = torch.zeros(b, h, s, s, dtype=torch.float16)
            out_fallback = F.scaled_dot_product_attention(q, k, v, attn_mask=dummy_mask)
            self.assertEqual(out_fallback.shape, (b, h, s, d))

            class MockMultiheadAttention(torch.nn.Module):
                def forward(self, q_t, k_t, v_t):
                    return F.scaled_dot_product_attention(q_t, k_t, v_t)

            model = MockMultiheadAttention()
            model_out = model(q, k, v)
            self.assertEqual(model_out.shape, (b, h, s, d))
        finally:
            replace_sdpa(enable=False)
            self.assertEqual(F.scaled_dot_product_attention, orig_sdpa)

        restored_out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(restored_out.shape, (b, h, s, d))

    def test_fast_path_and_prevalidation_optimization(self) -> None:
    # TC-12: Verify fast-path static caching, inference loop consistency, and error safety.
        b, h, s, d = 2, 4, 64, 64
        q, k, v = self._build_test_tensors(b, h, s, d, dtype=torch.float16)

        out_first = sageattn(q, k, v, tensor_layout="HND")
        for _ in range(10):
            out_fast = sageattn(q, k, v, tensor_layout="HND")
            self.assertTrue(torch.equal(out_first, out_fast))

        q_dec = torch.randn(b, h, 1, d, dtype=torch.float16)
        k_dec, v_dec, _ = self._build_test_tensors(b, h, 65, d, dtype=torch.float16)
        out_dec = sageattn(q_dec, k_dec, v_dec, is_causal=False)
        self.assertEqual(out_dec.shape, (b, h, 1, d))

        out_rewarm = sageattn(q, k, v, tensor_layout="HND")
        self.assertTrue(torch.equal(out_first, out_rewarm))

        q_bad = q.transpose(-2, -1)
        with self.assertRaises(ValueError):
            sageattn(q_bad, k, v)

        q_grad = q.clone().requires_grad_(True)
        with self.assertRaises(ValueError):
            sageattn(q_grad, k, v)

        hq, hk = 8, 2
        q_gqa, _, _ = self._build_test_tensors(b, hq, s, d, dtype=torch.float16)
        k_gqa, v_gqa, _ = self._build_test_tensors(b, hk, s, d, dtype=torch.float16)

        out_gqa1, lse_gqa1 = sageattn(q_gqa, k_gqa, v_gqa, return_lse=True)
        for _ in range(5):
            out_gqa_fast, lse_gqa_fast = sageattn(q_gqa, k_gqa, v_gqa, return_lse=True)
            self.assertTrue(torch.equal(out_gqa1, out_gqa_fast))
            self.assertTrue(torch.equal(lse_gqa1, lse_gqa_fast))

    def test_tc13_fp16_partial_overflow_envelope(self) -> None:
    # TC-13: A full-width 256-token PV block must not overflow the FP16 fold.
        d = 64
        for s in (256, 1024, 4096):
            for mag in (1.0, 10.0, 100.0):
                q = torch.zeros(1, 1, s, d, dtype=torch.float16)
                k = torch.zeros(1, 1, s, d, dtype=torch.float16)
                v = torch.full([1, 1, s, d], mag, dtype=torch.float16)
                folded = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp32")
                for accum in ("fp32+fp16", "fp32+fp32"):
                    with self.subTest(s=s, mag=mag, accum=accum):
                        out = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype=accum)
                        self.assertTrue(torch.isfinite(out).all().item(), f"{accum} overflowed at S={s}, max|V|={mag}")
                        self.assertLess((out.float() - folded.float()).abs().max().item() / mag, 1e-2)

        big = 1024
        q = torch.zeros(1, 1, big, d, dtype=torch.float16)
        k = torch.zeros(1, 1, big, d, dtype=torch.float16)
        v = torch.full([1, 1, big, d], 1024.0, dtype=torch.float16)
        out_fp32 = sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp32+fp32")
        self.assertTrue(torch.isfinite(out_fp32).all().item(), "fp32+fp32 must not overflow")


def main() -> None:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSageAttention2PythonAPI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if bool(result.wasSuccessful()) is False:
        sys.exit(1)


if __name__ == "__main__":
    main()