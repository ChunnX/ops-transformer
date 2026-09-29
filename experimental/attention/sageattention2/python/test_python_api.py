# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Comprehensive Python API acceptance test suite for SageAttention2 on Ascend 950PR.

Covers task specification requirements (SageAttention2_task_doc.md §2.1 ~ §2.4):
- TC-00: Function signatures, default arguments, and core module exports
- TC-01: Basic functionality & dtype (FP16/BF16, causal/non-causal, HND/NHD) and
  dispatch fixed to the FP8 PV path
- TC-02: qk_quant_gran x pv_accum_dtype matrix (per_thread/per_warp x fp32/fp32+fp32/fp32+fp16)
- TC-03: smooth_k toggle; smooth_v honored under fp32 and warn-ignored otherwise
- TC-04: LSE export, GQA broadcast correction, and L-B precision gates vs the
  open-source FP8-path emulation (error ratios max<=2 / mean<=1.2 / RMSE<=1.2)
- TC-05: Head dimension auto-padding & unpadding/cropping (D=48->64, D=96->128, D>128 error)
- TC-06: GQA + asymmetric sequence lengths
- TC-07: Causal attention
- TC-08: Strict exception interception (D>128, non-contiguous last dim, causal mismatch, attn_mask, GQA, illegal enums)
- TC-09: Long sequence execution (S>=8K, 16K on NPU)
- TC-10: Seamless SDPA monkey-patch replacement
Note: the formal L-A comparison against GPU sageattn_qk_int8_pv_fp8_cuda and
AscendOpTest acceptance run on real hardware; this suite bounds the numeric
datapath end-to-end on the CPU golden path (L-B layer).
"""

import math
import sys
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

# Add package directory to path
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from sageattention2 import replace_sdpa, run, sageattn, sageattn_qk_int8_pv_fp8_asc


class TestSageAttention2PythonAPI(unittest.TestCase):
    """Formal acceptance test suite for SageAttention2 PyTorch / Python API."""

    def setUp(self):
        torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # TC-00: 函数签名、默认参数与 core 模块对齐
    # -------------------------------------------------------------------------
    def test_tc00_signatures_and_core_module(self):
        """TC-00: Verify function signatures, default arguments, and core module exports."""
        import inspect
        from sageattention2 import core as s2_core
        from sageattention2.core import replace_sdpa as core_replace_sdpa
        from sageattention2.core import run as core_run
        from sageattention2.core import sageattn as core_sageattn
        from sageattention2.core import sageattn_qk_int8_pv_fp8_asc as core_sageattn_qk_int8_pv_fp8_asc

        self.assertIs(s2_core.sageattn, sageattn)
        self.assertIs(s2_core.sageattn_qk_int8_pv_fp8_asc, sageattn_qk_int8_pv_fp8_asc)
        self.assertIs(s2_core.replace_sdpa, replace_sdpa)
        self.assertIs(s2_core.run, run)

        # Inspect sageattn signature
        sig_sage = inspect.signature(sageattn)
        params_sage = sig_sage.parameters
        self.assertEqual(params_sage["tensor_layout"].default, "HND")
        self.assertEqual(params_sage["is_causal"].default, False)
        self.assertIsNone(params_sage["sm_scale"].default)
        self.assertEqual(params_sage["return_lse"].default, False)

        # Inspect sageattn_qk_int8_pv_fp8_asc signature
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

    # -------------------------------------------------------------------------
    # TC-01: 基本功能与精度/数据类型 (FP16 / BF16, Causal / Non-Causal, HND / NHD)
    # -------------------------------------------------------------------------
    def test_tc01_basic_functionality_and_dtypes(self):
        """TC-01: Verify FP16 and BF16 dtypes, causal/non-causal, and HND/NHD layouts."""
        b, h, s, d = 2, 4, 128, 64

        for dtype in (torch.float16, torch.bfloat16):
            for layout in ("HND", "NHD"):
                for is_causal in (False, True):
                    with self.subTest(dtype=dtype, layout=layout, is_causal=is_causal):
                        if layout == "HND":
                            q = torch.randn(b, h, s, d, dtype=dtype)
                            k = torch.randn(b, h, s, d, dtype=dtype)
                            v = torch.randn(b, h, s, d, dtype=dtype)
                            expected_shape = (b, h, s, d)
                        else:  # NHD
                            q = torch.randn(b, s, h, d, dtype=dtype)
                            k = torch.randn(b, s, h, d, dtype=dtype)
                            v = torch.randn(b, s, h, d, dtype=dtype)
                            expected_shape = (b, s, h, d)

                        # Test sageattn auto-dispatcher
                        out_auto = sageattn(
                            q, k, v, tensor_layout=layout, is_causal=is_causal
                        )
                        self.assertEqual(out_auto.shape, expected_shape)
                        self.assertEqual(out_auto.dtype, dtype)
                        self.assertFalse(torch.isnan(out_auto).any())
                        self.assertFalse(torch.isinf(out_auto).any())

                        # Test sageattn_qk_int8_pv_fp8_asc explicit interface
                        out_asc = sageattn_qk_int8_pv_fp8_asc(
                            q,
                            k,
                            v,
                            tensor_layout=layout,
                            is_causal=is_causal,
                            pv_accum_dtype="fp32+fp16",
                        )
                        self.assertEqual(out_asc.shape, expected_shape)
                        self.assertEqual(out_asc.dtype, dtype)
                        self.assertFalse(torch.isnan(out_asc).any())

    # -------------------------------------------------------------------------
    # TC-02: 量化粒度与累加精度组合 (per_thread/per_warp x fp32/fp32+fp32/fp32+fp16)
    # -------------------------------------------------------------------------
    def test_tc02_quant_gran_and_accum_dtypes(self):
        """TC-02: Verify per_thread/per_warp x fp32/fp32+fp32/fp32+fp16 combinations."""
        b, h, s, d = 1, 4, 64, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)
        scale = 1.0 / math.sqrt(d)
        truth = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), scale=scale
        )
        results = {}
        for gran in ("per_thread", "per_warp"):
            for accum in ("fp32+fp16", "fp32+fp32", "fp32"):
                with self.subTest(gran=gran, accum=accum):
                    out = sageattn_qk_int8_pv_fp8_asc(
                        q, k, v, qk_quant_gran=gran, pv_accum_dtype=accum
                    )
                    self.assertEqual(out.shape, (b, h, s, d))
                    self.assertEqual(out.dtype, torch.float16)
                    self.assertFalse(torch.isnan(out).any())
                    # Every configuration must stay inside the FP8-path error envelope.
                    self.assertLess((out.float() - truth).abs().max().item(), 0.05)
                    results[(gran, accum)] = out

        # The knobs must take effect: coarser groups and the FP16 partial roundtrip
        # each produce a different result rather than an alias of the same datapath.
        for gran in ("per_thread", "per_warp"):
            self.assertGreater(
                (results[(gran, "fp32+fp16")] - results[(gran, "fp32")]).abs().max().item(),
                0.0,
                f"{gran}: fp32+fp16 aliased fp32",
            )
        for accum in ("fp32+fp16", "fp32+fp32", "fp32"):
            self.assertGreater(
                (results[("per_thread", accum)] - results[("per_warp", accum)]).abs().max().item(),
                0.0,
                f"{accum}: per_warp aliased per_thread",
            )

    # -------------------------------------------------------------------------
    # TC-03: smooth_k / smooth_v 开关行为
    # -------------------------------------------------------------------------
    def test_tc03_smooth_k_and_smooth_v_behavior(self):
        """TC-03: Verify the smooth_k toggle and that smooth_v is honored under fp32, ignored elsewhere."""
        b, h, s, d = 1, 4, 64, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)

        # smooth_k=True vs smooth_k=False
        out_smooth = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=True)
        out_no_smooth = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_k=False)
        self.assertEqual(out_smooth.shape, (b, h, s, d))
        self.assertEqual(out_no_smooth.shape, (b, h, s, d))

        # smooth_v=True on the two-level formats warns and is ignored
        for accum in ("fp32+fp16", "fp32+fp32"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_v = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=True, pv_accum_dtype=accum)
                self.assertEqual(out_v.shape, (b, h, s, d))
                self.assertTrue(
                    any("smooth_v" in str(item.message) and "ignored" in str(item.message) for item in w),
                    f"expected smooth_v ignore warning for {accum}",
                )

        # smooth_v=True on single-level fp32 is honored without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out_v32 = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=True, pv_accum_dtype="fp32")
            self.assertEqual(out_v32.shape, (b, h, s, d))
            self.assertFalse(any("smooth_v" in str(item.message) and "ignored" in str(item.message) for item in w))

        # The honored switch must reach the datapath: V centering changes the
        # quantised operands, so the result must differ from the same run with
        # smooth_v=False while both stay inside the FP8-path error envelope.
        out_v32_off = sageattn_qk_int8_pv_fp8_asc(q, k, v, smooth_v=False, pv_accum_dtype="fp32")
        truth = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=1.0 / math.sqrt(d))
        self.assertGreater((out_v32 - out_v32_off).abs().max().item(), 0.0, "smooth_v aliased its off state")
        for name, out in (("on", out_v32), ("off", out_v32_off)):
            self.assertLess((out.float() - truth).abs().max().item(), 0.05, f"smooth_v={name} out of envelope")

    # -------------------------------------------------------------------------
    # 非连续跨步与 Stride 适应 (Non-contiguous strides with stride(-1)==1)
    # -------------------------------------------------------------------------
    def test_non_contiguous_and_strides(self):
        """Verify handling of non-contiguous input tensors with stride(-1)==1."""
        # 1. Sliced sequence dimension (stride(-1)==1 preserved, but non-contiguous overall)
        q_raw = torch.randn(2, 4, 128, 64, dtype=torch.float16)
        k_raw = torch.randn(2, 4, 128, 64, dtype=torch.float16)
        v_raw = torch.randn(2, 4, 128, 64, dtype=torch.float16)

        q_sliced = q_raw[:, :, ::2, :]  # shape: [2, 4, 64, 64], stride: [524288, 131072, 128, 1]
        k_sliced = k_raw[:, :, ::2, :]
        v_sliced = v_raw[:, :, ::2, :]

        self.assertFalse(q_sliced.is_contiguous())
        self.assertEqual(q_sliced.stride(-1), 1)

        out_sliced = sageattn(q_sliced, k_sliced, v_sliced, tensor_layout="HND")
        self.assertEqual(out_sliced.shape, (2, 4, 64, 64))
        self.assertEqual(out_sliced.dtype, torch.float16)
        self.assertFalse(torch.isnan(out_sliced).any())

        # 2. Permuted batch and heads dimension
        q_perm = q_raw.permute(1, 0, 2, 3)  # shape: [4, 2, 128, 64]
        k_perm = k_raw.permute(1, 0, 2, 3)
        v_perm = v_raw.permute(1, 0, 2, 3)

        self.assertFalse(q_perm.is_contiguous())
        self.assertEqual(q_perm.stride(-1), 1)

        out_perm = sageattn(q_perm, k_perm, v_perm, tensor_layout="HND")
        self.assertEqual(out_perm.shape, (4, 2, 128, 64))
        self.assertFalse(torch.isnan(out_perm).any())

        # 3. Asymmetric sequence lengths (Sq != Sk in non-causal)
        q_asym = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        k_asym = torch.randn(1, 4, 96, 64, dtype=torch.float16)
        v_asym = torch.randn(1, 4, 96, 64, dtype=torch.float16)

        out_asym = sageattn(q_asym, k_asym, v_asym, is_causal=False)
        self.assertEqual(out_asym.shape, (1, 4, 32, 64))

    # -------------------------------------------------------------------------
    # TC-04: 归一化对数因子导出与 GQA 校正 (LSE return, shape [B, Hq, Sq])
    # -------------------------------------------------------------------------
    def test_tc04_lse_export_and_gqa_broadcast(self):
        """TC-04: Verify LSE output tuple, shape [B, Hq, Sq], and GQA broadcast correction."""
        b, hq, hk, sq, sk, d = 2, 8, 2, 64, 64, 64

        q = torch.randn(b, hq, sq, d, dtype=torch.float16)
        k = torch.randn(b, hk, sk, d, dtype=torch.float16)
        v = torch.randn(b, hk, sk, d, dtype=torch.float16)

        # 1. Standard LSE return with GQA (Hq=8, Hk=2, G=4)
        out, lse = sageattn(q, k, v, tensor_layout="HND", return_lse=True)
        self.assertEqual(out.shape, (b, hq, sq, d))
        self.assertEqual(lse.shape, (b, hq, sq))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertFalse(torch.isnan(lse).any())
        self.assertFalse(torch.isinf(lse).any())

        # 2. LSE with explicit interface and smooth_k=False
        out_no_smooth, lse_no_smooth = sageattn_qk_int8_pv_fp8_asc(
            q, k, v, tensor_layout="HND", smooth_k=False, return_lse=True
        )
        self.assertEqual(lse_no_smooth.shape, (b, hq, sq))
        self.assertFalse(torch.isnan(lse_no_smooth).any())

        # 3. LSE in NHD layout
        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()

        out_nhd, lse_nhd = sageattn(
            q_nhd, k_nhd, v_nhd, tensor_layout="NHD", return_lse=True
        )
        self.assertEqual(out_nhd.shape, (b, sq, hq, d))
        self.assertEqual(lse_nhd.shape, (b, hq, sq))
        self.assertEqual(lse_nhd.dtype, torch.float32)

    # -------------------------------------------------------------------------
    # TC-01 补充：sageattn 固定分发到 FP8 PV 路径（pv_accum_dtype 默认下沉值）
    # -------------------------------------------------------------------------
    def test_tc01_dispatch_targets_fp8_pv(self):
        """TC-01: sageattn must dispatch to the FP8 PV path with fp32+fp16 by default."""
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

    # -------------------------------------------------------------------------
    # TC-04 补充：L-B 精度门限（vs 开源 FP8 路径数值仿真的误差比）+ LSE 数值
    # -------------------------------------------------------------------------
    @staticmethod
    def _open_source_pv_fp8(q, k, v, sm_scale, is_causal, q_group=4, k_group=16):
        """Open-source SageAttention2 sm89 FP8-path numeric emulation (CPU).

        Mirrors thu-ml/SageAttention per_thread granularity: per-4-token Q scales
        and per-16-token K scales over the full head dim after smooth_k,
        per-channel FP8 E4M3 V over the whole sequence, ln(448)-shifted online
        softmax with FP16 scores, and fp32+fp16 two-level accumulation. Groups
        are configurable so per_warp (32/64) is compared under its own config.
        """
        b, hq, sq, d = q.shape
        hk, sk = k.shape[1:3]

        def group_int8(x, group):
            pad = (-x.shape[-2]) % group
            xp = torch.nn.functional.pad(x.float(), (0, 0, 0, pad))
            g = xp.reshape(*x.shape[:-2], -1, group, x.shape[-1])
            amax = g.abs().amax(dim=(-2, -1), keepdim=True)
            scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
            q8 = (g / scale).round().clamp(-127, 127)
            row_scale = (
                scale.expand(-1, -1, -1, group, -1)
                .reshape(*x.shape[:-2], -1)[..., : x.shape[-2], None]
            )
            return q8.reshape(*x.shape[:-2], -1, x.shape[-1])[..., : x.shape[-2], :], row_scale

        mean_k = k.float().mean(-2, keepdim=True)
        q8, qs = group_int8(q, q_group)
        k8, ks = group_int8(k.float() - mean_k, k_group)
        vam = v.float().abs().amax(-2, keepdim=True)
        vscale = torch.where(vam > 0, vam / 448.0, torch.ones_like(vam))
        v_deq = (v.float() / vscale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * vscale

        out = torch.zeros(b, hq, sq, d, dtype=torch.float32)
        m = torch.full((b, hq, sq, 1), -torch.inf)
        den = torch.zeros_like(m)
        log448 = math.log(448.0)
        for start in range(0, sk, 128):
            s_int = q8.float() @ k8[:, :, start : start + 128].float().transpose(-2, -1)
            scores = (s_int * (qs * ks[:, :, start : start + 128].transpose(-2, -1)) * sm_scale).half()
            if is_causal:
                mask = torch.arange(start, min(start + 128, sk))[None, :] > torch.arange(sq)[:, None]
                scores = scores.masked_fill(mask, -torch.inf)
            bm = scores.float().amax(-1, keepdim=True) - log448
            nm = torch.maximum(bm, m)
            alpha = torch.where(torch.isfinite(m - nm), (m - nm).exp(), torch.zeros_like(m - nm))
            prob = (scores.float() - nm).exp()
            p8 = prob.clamp(max=448.0).to(torch.float8_e4m3fn).float()
            den = den * alpha + p8.sum(-1, keepdim=True)
            partial = (p8 @ v_deq[:, :, start : start + 128]).half().float()
            out = out * alpha + partial
            m = nm
        result = (out / den).to(q.dtype).float()
        lse = (m.squeeze(-1) + den.squeeze(-1).log()) + (q.float() * mean_k).sum(-1) * sm_scale
        return result, lse

    def test_tc04b_lb_precision_gates(self):
        """TC-04 (L-B layer): error ratios vs FP32 SDPA within max<=2 / mean<=1.2 / RMSE<=1.2.

        Compares the CPU golden datapath (the numerics registered as the CPU
        backend) against an independent open-source FP8-path emulation, both
        measured against the FP32 SDPA ground truth, per task spec "per_thread
        vs per_warp are compared under their own configuration". The formal
        L-A layer (GPU *_cuda golden via AscendOpTest) runs on real hardware
        and is out of scope for this host-side suite.
        """
        try:
            from sageattention2.reference import attention as cpu_golden
        except ImportError:
            self.skipTest("reference golden unavailable")
        cases = [
            # (heads, seq, dim, dtype, causal, gran groups)
            (2, 512, 64, torch.float16, False, (4, 16)),
            (2, 1024, 128, torch.float16, True, (4, 16)),
            (1, 512, 128, torch.bfloat16, False, (4, 16)),
            (2, 512, 128, torch.float16, False, (32, 64)),  # per_warp 同配置对标
        ]
        for heads, seq, dim, dtype, causal, (qg, kg) in cases:
            with self.subTest(heads=heads, seq=seq, dim=dim, dtype=dtype, causal=causal, groups=(qg, kg)):
                q = torch.randn(1, heads, seq, dim, dtype=dtype)
                k = torch.randn(1, heads, seq, dim, dtype=dtype)
                v = torch.randn(1, heads, seq, dim, dtype=dtype)
                scale = 1.0 / math.sqrt(dim)

                npu_out, npu_lse = cpu_golden(
                    q, k, v, sm_scale=scale, is_causal=causal,
                    qk_quant_gran="per_thread" if qg == 4 else "per_warp",
                    return_lse=True,
                )
                gpu_out, gpu_lse = self._open_source_pv_fp8(q, k, v, scale, causal, qg, kg)

                truth = F.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(), is_causal=causal, scale=scale
                )
                err = (npu_out.float() - truth).abs()
                berr = (gpu_out - truth).abs()
                ratio_max = (err.amax() / berr.amax().clamp_min(1e-9)).item()
                ratio_mean = (err.mean() / berr.mean().clamp_min(1e-12)).item()
                ratio_rmse = (
                    err.pow(2).mean().sqrt() / berr.pow(2).mean().sqrt().clamp_min(1e-12)
                ).item()
                self.assertLessEqual(ratio_max, 2.0, f"L-B max ratio {ratio_max:.3f}")
                self.assertLessEqual(ratio_mean, 1.2, f"L-B mean ratio {ratio_mean:.3f}")
                self.assertLessEqual(ratio_rmse, 1.2, f"L-B rmse ratio {ratio_rmse:.3f}")

                # LSE：两条量化路径对同一契约量的偏差（log 域，FP16 噪声量级）
                lse_rel = ((npu_lse - gpu_lse).abs() / npu_lse.abs().clamp_min(1.0)).amax().item()
                self.assertLessEqual(lse_rel, 5e-2, f"LSE relative deviation {lse_rel:.4f}")

    # -------------------------------------------------------------------------
    # TC-05: 特征维度自动补齐与裁剪 (Head dimension padding to 64 or 128)
    # -------------------------------------------------------------------------
    def test_tc05_head_dim_auto_pad_and_crop(self):
        """TC-05: Verify auto-padding for D <= 64 -> 64, 64 < D <= 128 -> 128, and cropping."""
        b, h, s = 1, 4, 64

        # D = 48 (should pad to 64, output cropped to 48)
        q48 = torch.randn(b, h, s, 48, dtype=torch.float16)
        k48 = torch.randn(b, h, s, 48, dtype=torch.float16)
        v48 = torch.randn(b, h, s, 48, dtype=torch.float16)

        out48 = sageattn(q48, k48, v48)
        self.assertEqual(out48.shape, (b, h, s, 48))
        self.assertFalse(torch.isnan(out48).any())

        # D = 96 (should pad to 128, output cropped to 96)
        q96 = torch.randn(b, h, s, 96, dtype=torch.float16)
        k96 = torch.randn(b, h, s, 96, dtype=torch.float16)
        v96 = torch.randn(b, h, s, 96, dtype=torch.float16)

        out96 = sageattn(q96, k96, v96)
        self.assertEqual(out96.shape, (b, h, s, 96))
        self.assertFalse(torch.isnan(out96).any())

        # D = 32 (should pad to 64, output cropped to 32)
        q32 = torch.randn(b, h, s, 32, dtype=torch.float16)
        k32 = torch.randn(b, h, s, 32, dtype=torch.float16)
        v32 = torch.randn(b, h, s, 32, dtype=torch.float16)

        out32 = sageattn(q32, k32, v32)
        self.assertEqual(out32.shape, (b, h, s, 32))

        # D = 64 & D = 128 (exact boundaries)
        q64 = torch.randn(b, h, s, 64, dtype=torch.float16)
        out64 = sageattn(q64, q64, q64)
        self.assertEqual(out64.shape, (b, h, s, 64))

        q128 = torch.randn(b, h, s, 128, dtype=torch.float16)
        out128 = sageattn(q128, q128, q128)
        self.assertEqual(out128.shape, (b, h, s, 128))

        # D > 128 (must raise ValueError("Only head_dim <= 128 is supported"))
        q144 = torch.randn(b, h, s, 144, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "Only head_dim <= 128 is supported"):
            sageattn(q144, q144, q144)

    # -------------------------------------------------------------------------
    # TC-06: GQA + 异构序列长度 (GQA Hq % Hk == 0, Sq != Sk)
    # -------------------------------------------------------------------------
    def test_tc06_gqa_and_asymmetric_sequence_lengths(self):
        """TC-06: Verify GQA (Hq % Hk == 0) with asymmetric sequence lengths (Sq != Sk)."""
        b, hq, hk, sq, sk, d = 2, 8, 2, 32, 96, 64
        q = torch.randn(b, hq, sq, d, dtype=torch.float16)
        k = torch.randn(b, hk, sk, d, dtype=torch.float16)
        v = torch.randn(b, hk, sk, d, dtype=torch.float16)

        # Non-causal allows Sq != Sk
        out_hnd = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        self.assertEqual(out_hnd.shape, (b, hq, sq, d))

        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()
        out_nhd = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD", is_causal=False)
        self.assertEqual(out_nhd.shape, (b, sq, hq, d))

    # -------------------------------------------------------------------------
    # TC-07: 因果注意力掩码特性 (Causal Attention Masking)
    # -------------------------------------------------------------------------
    def test_tc07_causal_attention(self):
        """TC-07: Verify causal attention masking (future key positions masked)."""
        b, h, d = 1, 2, 64
        for s in [64, 128, 256]:
            with self.subTest(seq_len=s):
                q = torch.randn(b, h, s, d, dtype=torch.float16)
                k = torch.randn(b, h, s, d, dtype=torch.float16)
                v = torch.randn(b, h, s, d, dtype=torch.float16)

                out_causal = sageattn(q, k, v, is_causal=True)
                self.assertEqual(out_causal.shape, (b, h, s, d))
                self.assertFalse(torch.isnan(out_causal).any())

                # Modify future tokens of k and v, output at pos 0 must remain identical within quant noise
                k_modified = k.clone()
                v_modified = v.clone()
                k_modified[:, :, 1:, :] = torch.randn_like(k_modified[:, :, 1:, :])
                v_modified[:, :, 1:, :] = torch.randn_like(v_modified[:, :, 1:, :])
                out_modified = sageattn(q, k_modified, v_modified, is_causal=True)
                diff = (out_causal[:, :, 0, :] - out_modified[:, :, 0, :]).abs().max()
                self.assertLess(diff.item(), 0.05)

                # Grouped INT8 K scales span 16 tokens and MXFP8 V scales span 32,
                # so a group's scale sees the tokens sharing that group. Exact
                # independence therefore holds across whole scale groups: without
                # the sequence-wide meanK, rows below a group boundary are
                # bit-identical when only later groups change.
                boundary = 32
                k_future = k.clone()
                v_future = v.clone()
                k_future[:, :, boundary:, :] = torch.randn_like(k_future[:, :, boundary:, :])
                v_future[:, :, boundary:, :] = torch.randn_like(v_future[:, :, boundary:, :])
                out_exact = sageattn(q, k, v, is_causal=True, smooth_k=False)
                out_exact_mod = sageattn(q, k_future, v_future, is_causal=True, smooth_k=False)
                diff_exact = (out_exact[:, :, :boundary, :] - out_exact_mod[:, :, :boundary, :]).abs().max()
                self.assertEqual(diff_exact.item(), 0.0)

                # Midpoint causality at an aligned boundary without smooth_k
                mid = (s // 2) // boundary * boundary
                self.assertGreaterEqual(mid, boundary)
                k_mid = k.clone()
                v_mid = v.clone()
                k_mid[:, :, mid:, :] = torch.randn_like(k_mid[:, :, mid:, :])
                v_mid[:, :, mid:, :] = torch.randn_like(v_mid[:, :, mid:, :])
                out_mid = sageattn(q, k_mid, v_mid, is_causal=True, smooth_k=False)
                diff_mid = (out_exact[:, :, :mid, :] - out_mid[:, :, :mid, :]).abs().max()
                self.assertEqual(diff_mid.item(), 0.0)

    # -------------------------------------------------------------------------
    # TC-08: 严格异常拦截机制 (Input validation & 4-tier exception checks)
    # -------------------------------------------------------------------------
    def test_tc08_strict_exception_interception(self):
        """TC-08: Verify strict error checking for invalid arguments, layouts, and shapes."""
        q = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        k = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 64, 64, dtype=torch.float16)

        # 1. Non-contiguous last dimension (stride(-1) != 1)
        q_bad_stride = q.transpose(-2, -1)  # shape: [1, 4, 64, 64], stride(-1) == 64 != 1
        with self.assertRaises(ValueError):
            sageattn(q_bad_stride, k, v)

        # 2. Invalid tensor_layout
        with self.assertRaises(ValueError):
            sageattn(q, k, v, tensor_layout="BSND")

        # 3. Invalid qk_quant_gran (only per_thread and per_warp are accepted)
        for gran in ("per_head", "per_token", "per_block"):
            with self.subTest(qk_quant_gran=gran):
                with self.assertRaises(ValueError):
                    sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran=gran)

        # 4. Invalid pv_accum_dtype
        with self.assertRaises(ValueError):
            sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="int32")

        # 5. Causal requires Sq == Sk
        q_causal = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        k_causal = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q_causal, k_causal, k_causal, is_causal=True)

        # 6. attn_mask is unsupported
        dummy_mask = torch.zeros(1, 1, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q, k, v, attn_mask=dummy_mask)

        # 7. GQA mismatch (Hq % Hk != 0)
        q_gqa = torch.randn(1, 7, 64, 64, dtype=torch.float16)
        k_gqa = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q_gqa, k_gqa, k_gqa)

        # 8. Rank != 4
        q_3d = torch.randn(4, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            sageattn(q_3d, q_3d, q_3d)

        # 9. Dtype mismatch (q is FP16, k is BF16)
        k_bf16 = k.to(torch.bfloat16)
        with self.assertRaises(ValueError):
            sageattn(q, k_bf16, v)

        # 10. Unsupported input dtype (FP32)
        q_fp32 = q.float()
        with self.assertRaises(ValueError):
            sageattn(q_fp32, q_fp32, q_fp32)

        # 11. smooth_v=True warning on fp32+fp16/fp32+fp32
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sageattn_qk_int8_pv_fp8_asc(
                q, k, v, smooth_v=True, pv_accum_dtype="fp32+fp16"
            )
            self.assertTrue(any("smooth_v is ignored" in str(item.message) for item in w))

    # -------------------------------------------------------------------------
    # TC-09: 长序列处理 (S >= 4096 / S >= 8192)
    # -------------------------------------------------------------------------
    def test_tc09_long_sequence_handling(self):
        """TC-09: Verify long sequence execution (S>=8K; 16K tier on NPU).

        The CPU fallback covers reduced sizes as a smoke test only; the task
        spec's 16K tier is asserted on NPU where TC-09 is formally accepted.
        """
        is_npu = hasattr(torch, "npu") and torch.npu.is_available()
        device = "npu" if is_npu else "cpu"
        seq_lens = [4096, 8192, 16384] if is_npu else [512, 1024]
        heads = 4 if is_npu else 2
        for s in seq_lens:
            with self.subTest(seq_len=s):
                q = torch.randn(1, heads, s, 128, dtype=torch.float16, device=device)
                k = torch.randn(1, heads, s, 128, dtype=torch.float16, device=device)
                v = torch.randn(1, heads, s, 128, dtype=torch.float16, device=device)
                out = sageattn(q, k, v, tensor_layout="HND")
                self.assertEqual(out.shape, (1, heads, s, 128))
                self.assertFalse(torch.isnan(out).any())

    # -------------------------------------------------------------------------
    # TC-10: 官方接口无缝平替冒烟 (torch.nn.functional.scaled_dot_product_attention)
    # -------------------------------------------------------------------------
    def test_tc10_sdpa_seamless_substitution(self):
        """TC-10: Verify replace_sdpa monkey-patching and restoring F.scaled_dot_product_attention."""
        b, h, s, d = 1, 4, 64, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)

        orig_sdpa = F.scaled_dot_product_attention

        try:
            # Enable monkey-patch
            replace_sdpa(enable=True)
            self.assertNotEqual(F.scaled_dot_product_attention, orig_sdpa)

            # 1. Forward through SDPA interface (standard non-causal)
            out_sdpa = F.scaled_dot_product_attention(q, k, v)
            self.assertEqual(out_sdpa.shape, (b, h, s, d))
            self.assertEqual(out_sdpa.dtype, torch.float16)
            self.assertFalse(torch.isnan(out_sdpa).any())

            # 2. Forward with causal=True
            out_sdpa_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            self.assertEqual(out_sdpa_causal.shape, (b, h, s, d))

            # 3. Forward with custom scale
            custom_scale = 0.125
            out_sdpa_scale = F.scaled_dot_product_attention(q, k, v, scale=custom_scale)
            self.assertEqual(out_sdpa_scale.shape, (b, h, s, d))

            # 4. Graceful fallback on unsupported features (attn_mask or dropout_p > 0)
            dummy_mask = torch.zeros(b, h, s, s, dtype=torch.float16)
            out_fallback = F.scaled_dot_product_attention(q, k, v, attn_mask=dummy_mask)
            self.assertEqual(out_fallback.shape, (b, h, s, d))

            # 5. In a simulated Attention Module
            class MockMultiheadAttention(torch.nn.Module):
                def forward(self, q, k, v):
                    return F.scaled_dot_product_attention(q, k, v)

            model = MockMultiheadAttention()
            model_out = model(q, k, v)
            self.assertEqual(model_out.shape, (b, h, s, d))

        finally:
            # Disable monkey-patch and restore
            replace_sdpa(enable=False)
            self.assertEqual(F.scaled_dot_product_attention, orig_sdpa)

        # Verify restored behavior
        restored_out = F.scaled_dot_product_attention(q, k, v)
        self.assertEqual(restored_out.shape, (b, h, s, d))

    # -------------------------------------------------------------------------
    # TC-11: 快速通路 (Fast-Path) 与预校验缓存加速验证
    # -------------------------------------------------------------------------
    def test_fast_path_and_prevalidation_optimization(self):
        """TC-11: Verify fast-path static caching, inference loop consistency, and error safety."""
        b, h, s, d = 2, 4, 64, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)

        # 1. Warm-up and repeated execution (simulating transformer inference loop)
        out_first = sageattn(q, k, v, tensor_layout="HND")
        for _ in range(10):
            out_fast = sageattn(q, k, v, tensor_layout="HND")
            self.assertTrue(torch.equal(out_first, out_fast))

        # 2. Dynamic shape transition (e.g. sequence length changes in decode loop)
        q_dec = torch.randn(b, h, 1, d, dtype=torch.float16)
        k_dec = torch.randn(b, h, 65, d, dtype=torch.float16)
        v_dec = torch.randn(b, h, 65, d, dtype=torch.float16)
        out_dec = sageattn(q_dec, k_dec, v_dec, is_causal=False)
        self.assertEqual(out_dec.shape, (b, h, 1, d))

        # 3. Switching back to previous cached shape succeeds and hits cache
        out_rewarm = sageattn(q, k, v, tensor_layout="HND")
        self.assertTrue(torch.equal(out_first, out_rewarm))

        # 4. Strict error interception remains active even when fast-path is warmed up
        # Non-contiguous stride(-1) != 1 on same shape
        q_bad = q.transpose(-2, -1)
        with self.assertRaises(ValueError):
            sageattn(q_bad, k, v)

        # Autograd requires_grad on same shape
        q_grad = q.clone().requires_grad_(True)
        with self.assertRaises(ValueError):
            sageattn(q_grad, k, v)

        # 5. Fast-path with GQA LSE return
        hq, hk = 8, 2
        q_gqa = torch.randn(b, hq, s, d, dtype=torch.float16)
        k_gqa = torch.randn(b, hk, s, d, dtype=torch.float16)
        v_gqa = torch.randn(b, hk, s, d, dtype=torch.float16)

        out_gqa1, lse_gqa1 = sageattn(q_gqa, k_gqa, v_gqa, return_lse=True)
        for _ in range(5):
            out_gqa_fast, lse_gqa_fast = sageattn(q_gqa, k_gqa, v_gqa, return_lse=True)
            self.assertTrue(torch.equal(out_gqa1, out_gqa_fast))
            self.assertTrue(torch.equal(lse_gqa1, lse_gqa_fast))


def main():
    print("=" * 80)
    print("  [SageAttention2 Python API Acceptance Test Suite (TC-00 ~ TC-10, 含 L-B 精度门限)]")
    print("=" * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSageAttention2PythonAPI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\n✓ ALL PYTHON API ACCEPTANCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
