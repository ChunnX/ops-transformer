#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Data-type × layout coverage for `cann_ops_transformer.ops.block_sparse_attention`.

This complements test_attn.py / test_lse.py (which fix dtype=bf16, layout=BNSD)
by exercising the full support matrix the kernel advertises:

                 │ out dtype        │ SoC          │ layouts
  ───────────────┼──────────────────┼──────────────┼──────────────────
  float16        │ float16          │ all          │ TND, BNSD, BSND*
  bfloat16       │ bfloat16         │ all          │ TND, BNSD, BSND*
  float8_e4m3fn  │ float16/bfloat16 │ 950 (arch35) │ TND, BNSD, BSND

  *BSND requires an Ascend 950 (arch35).

What runs where
───────────────
• FP16/BF16 + TND/BNSD  → run on every supported chip (incl. Atlas A2/A3).
  These are validated *numerically* by layout-equivalence: the same data fed as
  TND must match the BNSD result (the kernel does identical math, only the
  memory layout differs). No external reference needed.
• FP8 and BSND          → require arch35; SKIPPED on A2/A3 with a clear reason.
  On arch35 they run as contract tests (shape / dtype / finiteness) plus, for
  BSND, layout-equivalence vs BNSD.
• The SoC guard and attention_out_dtype validation are checked on every chip — those are
  pure front-end logic and must reject bad combinations *before* the kernel.

Run with:
    pytest test_dtypes_layouts.py -v
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import cann_ops_transformer
from cann_ops_transformer.ops.block_sparse_attention import (
    _is_arch220_soc,
    _resolve_out_dtype,
)

SEED = 42
DEVICE = "npu:0"
BLOCK_Q = 128
BLOCK_KV = 128
torch.npu.set_device(DEVICE)


# ── SoC capability detection (mirrors the front-end guard) ───────────────────

def _device_name():
    try:
        return torch_npu.npu.get_device_name(torch_npu.npu.current_device())
    except Exception:  # pylint: disable=broad-except
        return "unknown"


_DEVICE_NAME = _device_name()
_IS_ARCH35 = not _is_arch220_soc(_DEVICE_NAME)
_arch35_only = pytest.mark.skipif(
    not _IS_ARCH35,
    reason=f"FP8 / BSND require an Ascend 950 (arch35); current device is "
           f"'{_DEVICE_NAME}' (Atlas A2/A3, arch v220).",
)
# inner_precise is SoC-gated by the kernel (block_sparse_attention_tiling.cpp):
# 950 → 4 (low/high mixed softmax); 910/910_93 → 0 (fp32 softmax). Picking it from
# the runtime SoC lets the identical test run on both chips.
_INNER_PRECISE = 4 if _IS_ARCH35 else 0

# Tolerances for layout-equivalence (identical math, only layout differs).
_EQ_ATOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
_EQ_RTOL = 1e-3


def _make_fp8(*shape, device=DEVICE):
    """Construct a float8_e4m3fn tensor on `device`.

    The cast to fp8 is done on CPU first, then moved with `.npu()` — casting to
    fp8 *on* an NPU tensor (`.to(float8_e4m3fn)` on a device tensor) is not
    supported on Atlas A2/A3 (fails in aclnnInplaceCopy). This mirrors how
    float8_crash_reproduce.py builds its inputs.
    """
    return torch.randn(*shape).to(torch.float8_e4m3fn).to(device)


# ── Shared helpers ───────────────────────────────────────────────────────────

def _dense_mask(b, n, s_q, s_kv, device=DEVICE, block_q=BLOCK_Q, block_kv=BLOCK_KV):
    q_blocks = (s_q + block_q - 1) // block_q
    kv_blocks = (s_kv + block_kv - 1) // block_kv
    return torch.ones(b, n, q_blocks, kv_blocks, dtype=torch.int8, device=device)


def _run_bnsd(q_bnsd, k_bnsd, v_bnsd, n, scale, lse=False, attention_out_dtype=None,
              dequant=None, block_kv=BLOCK_KV):
    """q_*_bnsd: [B, N, S, D]. Returns (out[B,N,S,D], lse_or_none).

    block_kv overrides the KV block size — FP8 requires it to be a multiple of 256
    (block_sparse_attention_tiling.cpp CheckBlockShapeQuantConstraint); FP16/BF16 use 128.
    """
    b, _, s_q, _ = q_bnsd.shape
    s_kv = k_bnsd.shape[2]
    qds, kds, vds = (dequant or (None, None, None))
    out, lse_t = cann_ops_transformer.ops.block_sparse_attention(
        q_bnsd, k_bnsd, v_bnsd,
        block_sparse_mask=_dense_mask(b, n, s_q, s_kv, block_kv=block_kv),
        block_shape=[BLOCK_Q, block_kv],
        actual_seq_lengths=[s_q] * b, actual_seq_lengths_kv=[s_kv] * b,
        num_heads=n, num_key_value_heads=n,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        mask_type=0, scale_value=float(scale),
        inner_precise=_INNER_PRECISE, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=lse,
        q_dequant_scale=qds, k_dequant_scale=kds, v_dequant_scale=vds,
        attention_out_dtype=attention_out_dtype,
    )
    return out, (lse_t if lse else None)


def _run_tnd(q_tnd, k_tnd, v_tnd, n, s_q, s_kv, scale, lse=False, attention_out_dtype=None):
    """q_*_tnd: [T, N, D] with a single batch (T == S). Returns (out, lse_or_none)."""
    out, lse_t = cann_ops_transformer.ops.block_sparse_attention(
        q_tnd, k_tnd, v_tnd,
        block_sparse_mask=_dense_mask(1, n, s_q, s_kv),
        block_shape=[BLOCK_Q, BLOCK_KV],
        actual_seq_lengths=[s_q], actual_seq_lengths_kv=[s_kv],
        num_heads=n, num_key_value_heads=n,
        q_input_layout="TND", kv_input_layout="TND",
        mask_type=0, scale_value=float(scale),
        inner_precise=_INNER_PRECISE, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=lse,
        attention_out_dtype=attention_out_dtype,
    )
    return out, (lse_t if lse else None)


def _run_bsnd(q_bsnd, k_bsnd, v_bsnd, n, scale, lse=False, attention_out_dtype=None):
    """q_*_bsnd: [B, S, N, D]. Returns (out, lse_or_none)."""
    b, s_q = q_bsnd.shape[0], q_bsnd.shape[1]
    s_kv = k_bsnd.shape[1]
    out, lse_t = cann_ops_transformer.ops.block_sparse_attention(
        q_bsnd, k_bsnd, v_bsnd,
        block_sparse_mask=_dense_mask(b, n, s_q, s_kv),
        block_shape=[BLOCK_Q, BLOCK_KV],
        actual_seq_lengths=[s_q] * b, actual_seq_lengths_kv=[s_kv] * b,
        num_heads=n, num_key_value_heads=n,
        q_input_layout="BSND", kv_input_layout="BSND",
        mask_type=0, scale_value=float(scale),
        inner_precise=_INNER_PRECISE, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=lse,
        attention_out_dtype=attention_out_dtype,
    )
    return out, (lse_t if lse else None)


# ════════════════════════════════════════════════════════════════════════════
#  Front-end logic (runs on every chip — pure validation, no kernel binary)
# ════════════════════════════════════════════════════════════════════════════

class TestOutDtypeResolution:
    """`_resolve_out_dtype` is the single source of truth for the (in→out) dtype
    pair that selects the kernel binary."""

    def test_fp16_defaults_to_fp16(self):
        assert _resolve_out_dtype(torch.float16, None) == torch.float16

    def test_bf16_defaults_to_bf16(self):
        assert _resolve_out_dtype(torch.bfloat16, None) == torch.bfloat16

    def test_fp8_defaults_to_bf16(self):
        assert _resolve_out_dtype(torch.float8_e4m3fn, None) == torch.bfloat16

    def test_fp8_accepts_fp16(self):
        assert _resolve_out_dtype(torch.float8_e4m3fn, torch.float16) == torch.float16

    def test_fp16_rejects_mismatched_out_dtype(self):
        with pytest.raises(ValueError, match="must equal the input dtype"):
            _resolve_out_dtype(torch.float16, torch.bfloat16)

    def test_fp8_rejects_fp8_out(self):
        with pytest.raises(ValueError, match="float16 or torch.bfloat16"):
            _resolve_out_dtype(torch.float8_e4m3fn, torch.float8_e4m3fn)

    def test_unsupported_input_dtype(self):
        with pytest.raises(ValueError, match="must be float16, bfloat16 or float8"):
            _resolve_out_dtype(torch.float32, None)


def test_fp16_out_dtype_mismatch_raises_in_call():
    """fp16 input with attention_out_dtype=bfloat16 must be rejected by the front-end."""
    torch.manual_seed(SEED)
    q = torch.randn(1, 1, 256, 128, dtype=torch.float16, device=DEVICE)
    with pytest.raises(ValueError, match="must equal the input dtype"):
        _run_bnsd(q, q, q, 1, 1 / math.sqrt(128), attention_out_dtype=torch.bfloat16)


# ════════════════════════════════════════════════════════════════════════════
#  SoC guard — FP8 / BSND must fail *clearly* on arch v220 (NOT crash deep in
#  the kernel with "Cannot find binary for op BlockSparseAttention").
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(_IS_ARCH35, reason="guard only fires on arch v220 (A2/A3)")
def test_fp8_on_v220_raises_clear_error():
    torch.manual_seed(SEED)
    b, n, s, d = 1, 1, 256, 128
    q = _make_fp8(b, n, s, d)
    ceil_q = (s + 127) // 128
    ceil_kv = (s + 255) // 256
    qds = torch.rand(b, n, ceil_q, 1, dtype=torch.float32, device=DEVICE) + 0.5
    kds = torch.rand(b, n, ceil_kv, 1, dtype=torch.float32, device=DEVICE) + 0.5
    with pytest.raises(RuntimeError, match="Ascend 950"):
        cann_ops_transformer.ops.block_sparse_attention(
            q, q, q,
            block_sparse_mask=_dense_mask(b, n, s, s),
            block_shape=[BLOCK_Q, BLOCK_KV],
            actual_seq_lengths=[s], actual_seq_lengths_kv=[s],
            num_heads=n, num_key_value_heads=n,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            mask_type=0, scale_value=1 / math.sqrt(d),
            inner_precise=_INNER_PRECISE, block_size=0,
            pre_tokens=2147483647, next_tokens=2147483647,
            softmax_lse_flag=False,
            q_dequant_scale=qds, k_dequant_scale=kds, v_dequant_scale=kds,
            attention_out_dtype=torch.bfloat16,
        )


@pytest.mark.skipif(_IS_ARCH35, reason="guard only fires on arch v220 (A2/A3)")
def test_bsnd_on_v220_raises_clear_error():
    torch.manual_seed(SEED)
    b, s, n, d = 1, 256, 1, 128
    q = torch.randn(b, s, n, d, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError, match="Ascend 950"):
        _run_bsnd(q, q, q, n, 1 / math.sqrt(d))


# ════════════════════════════════════════════════════════════════════════════
#  FP16 / BF16 — TND vs BNSD layout equivalence (runs on every chip)
# ════════════════════════════════════════════════════════════════════════════

_EQ_SHAPES = [(1, 256, 128), (2, 512, 128), (4, 1024, 128)]  # (N, S, D)
_EQ_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", _EQ_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", _EQ_SHAPES, ids=lambda s: f"n{s[0]}-s{s[1]}-d{s[2]}")
def test_tnd_matches_bnsd(shape, dtype):
    """Same data as TND ([S,N,D]) must produce the same attention as BNSD
    ([1,N,S,D]) — only the layout differs, the math is identical."""
    torch.manual_seed(SEED)
    n, s, d = shape
    scale = 1 / math.sqrt(d)

    q_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)
    k_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)
    v_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)

    # BNSD [1,N,S,D] → TND [S,N,D]
    q_tnd = q_bnsd[0].transpose(0, 1).contiguous()
    k_tnd = k_bnsd[0].transpose(0, 1).contiguous()
    v_tnd = v_bnsd[0].transpose(0, 1).contiguous()

    out_bnsd, _ = _run_bnsd(q_bnsd, k_bnsd, v_bnsd, n, scale)
    out_tnd, _ = _run_tnd(q_tnd, k_tnd, v_tnd, n, s, s, scale)

    assert out_tnd.shape == (s, n, d)
    assert out_tnd.dtype == dtype
    # TND [S,N,D] → BNSD [1,N,S,D] for comparison
    out_tnd_as_bnsd = out_tnd.transpose(0, 1).unsqueeze(0)
    atol = _EQ_ATOL[dtype]
    if not torch.allclose(out_tnd_as_bnsd.cpu(), out_bnsd.cpu(), rtol=_EQ_RTOL, atol=atol):
        max_diff = (out_tnd_as_bnsd.float() - out_bnsd.float()).abs().max().item()
        pytest.fail(f"TND vs BNSD mismatch shape={shape} dtype={dtype}: max_diff={max_diff:.6f}")


@pytest.mark.parametrize("dtype", _EQ_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", _EQ_SHAPES, ids=lambda s: f"n{s[0]}-s{s[1]}-d{s[2]}")
def test_tnd_lse_shape_and_matches_bnsd(shape, dtype):
    """TND LSE is [T, N, 1]; its values must match the BNSD LSE [1, N, S, 1]."""
    torch.manual_seed(SEED)
    n, s, d = shape
    scale = 1 / math.sqrt(d)

    q_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)
    q_tnd = q_bnsd[0].transpose(0, 1).contiguous()

    _, lse_bnsd = _run_bnsd(q_bnsd, q_bnsd, q_bnsd, n, scale, lse=True)
    _, lse_tnd = _run_tnd(q_tnd, q_tnd, q_tnd, n, s, s, scale, lse=True)

    assert lse_bnsd.shape == (1, n, s, 1)
    assert lse_tnd.shape == (s, n, 1)
    assert lse_tnd.dtype == torch.float32

    # BNSD [1,N,S,1]→[N,S]→[S,N];  TND [S,N,1]→[S,N]
    lse_bnsd_sn = lse_bnsd[0].squeeze(-1).transpose(0, 1)
    lse_tnd_sn = lse_tnd.squeeze(-1)
    if not torch.allclose(lse_tnd_sn.cpu(), lse_bnsd_sn.cpu(), rtol=1e-3, atol=1e-3):
        max_diff = (lse_tnd_sn.cpu() - lse_bnsd_sn.cpu()).abs().max().item()
        pytest.fail(f"TND vs BNSD LSE mismatch shape={shape} dtype={dtype}: max_diff={max_diff:.6f}")


# ════════════════════════════════════════════════════════════════════════════
#  BSND layout (arch35 only) — equivalence vs BNSD
# ════════════════════════════════════════════════════════════════════════════

@_arch35_only
@pytest.mark.parametrize("dtype", _EQ_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", _EQ_SHAPES, ids=lambda s: f"n{s[0]}-s{s[1]}-d{s[2]}")
def test_bsnd_matches_bnsd(shape, dtype):
    """BSND ([B,S,N,D]) must match BNSD ([B,N,S,D]) on the same data."""
    torch.manual_seed(SEED)
    n, s, d = shape
    scale = 1 / math.sqrt(d)

    q_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)
    k_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)
    v_bnsd = torch.randn(1, n, s, d, dtype=dtype, device=DEVICE)

    # BNSD [1,N,S,D] → BSND [1,S,N,D]
    q_bsnd = q_bnsd.transpose(1, 2).contiguous()
    k_bsnd = k_bnsd.transpose(1, 2).contiguous()
    v_bsnd = v_bnsd.transpose(1, 2).contiguous()

    out_bnsd, _ = _run_bnsd(q_bnsd, k_bnsd, v_bnsd, n, scale)
    out_bsnd, _ = _run_bsnd(q_bsnd, k_bsnd, v_bsnd, n, scale)

    assert out_bsnd.shape == (1, s, n, d)
    out_bsnd_as_bnsd = out_bsnd.transpose(1, 2)
    atol = _EQ_ATOL[dtype]
    if not torch.allclose(out_bsnd_as_bnsd.cpu(), out_bnsd.cpu(), rtol=_EQ_RTOL, atol=atol):
        max_diff = (out_bsnd_as_bnsd.float() - out_bnsd.float()).abs().max().item()
        pytest.fail(f"BSND vs BNSD mismatch shape={shape} dtype={dtype}: max_diff={max_diff:.6f}")


# ════════════════════════════════════════════════════════════════════════════
#  FP8 (float8_e4m3fn) — arch35 only. Contract test: the op must run and return
#  a finite tensor of the requested out dtype with query's shape, for each
#  attention_out_dtype and each layout. (Numerical correctness needs a bit-exact dequant
#  reference; this gates the API contract and that a binary exists.)
# ════════════════════════════════════════════════════════════════════════════

def _fp8_dequant_scales(b, n, kv_n, s_q, s_kv, block_q=128, block_kv=256, device=DEVICE):
    # The per-block reverse-quant factor counts must match the block grid the kernel
    # derives from block_sparse_mask (ValidateGenericDequantScale,
    # block_sparse_attention_tiling.cpp): q -> ceil(maxQ / blockShapeX),
    # k/v -> ceil(maxKV / blockShapeY). For FP8 the kernel mandates blockShapeX == 128
    # and blockShapeY a multiple of 256 (CheckBlockShapeQuantConstraint) — hence the
    # V2 doc's "ceil(maxKV/256)" for k/v. (My first cut wrongly used a 128 KV block,
    # which the tiling rejected: "kDequantScale ... expected (1,2,2,1) ... got (1,2,1,1)".)
    ceil_q = (s_q + block_q - 1) // block_q
    ceil_kv = (s_kv + block_kv - 1) // block_kv
    qds = torch.rand(b, n, ceil_q, 1, dtype=torch.float32, device=device) + 0.5
    kds = torch.rand(b, kv_n, ceil_kv, 1, dtype=torch.float32, device=device) + 0.5
    vds = torch.rand(b, kv_n, ceil_kv, 1, dtype=torch.float32, device=device) + 0.5
    return qds, kds, vds


# FP8 mandates blockShapeX==128 and blockShapeY a multiple of 256 on arch35
# (block_sparse_attention_tiling.cpp). s=512 with FP8_BLOCK_KV=256 → 2 KV blocks.
FP8_BLOCK_KV = 256


@_arch35_only
@pytest.mark.parametrize("attention_out_dtype", _EQ_DTYPES, ids=lambda d: "out_" + str(d).split(".")[-1])
def test_fp8_bnsd_contract(attention_out_dtype):
    torch.manual_seed(SEED)
    b, n, s, d = 1, 2, 512, 128
    scale = 1 / math.sqrt(d)
    q = _make_fp8(b, n, s, d)
    qds, kds, vds = _fp8_dequant_scales(b, n, n, s, s, block_kv=FP8_BLOCK_KV)

    out, _ = _run_bnsd(q, q, q, n, scale, attention_out_dtype=attention_out_dtype,
                       dequant=(qds, kds, vds), block_kv=FP8_BLOCK_KV)

    assert out.shape == (b, n, s, d)
    assert out.dtype == attention_out_dtype
    assert torch.isfinite(out.float()).all(), "FP8 attention produced non-finite values"


@_arch35_only
def test_fp8_requires_dequant_scales():
    """FP8 inputs without dequant scales must be rejected (not silently run)."""
    torch.manual_seed(SEED)
    b, n, s, d = 1, 1, 256, 128
    q = _make_fp8(b, n, s, d)
    with pytest.raises((RuntimeError, ValueError)):
        _run_bnsd(q, q, q, n, 1 / math.sqrt(d), attention_out_dtype=torch.bfloat16)


# ── __main__ smoke ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    log.info("device=%s  arch35=%s", _DEVICE_NAME, _IS_ARCH35)

    TestOutDtypeResolution().test_fp8_defaults_to_bf16()
    log.info("attention_out_dtype resolution OK")

    test_tnd_matches_bnsd((2, 512, 128), torch.bfloat16)
    log.info("TND == BNSD (bf16) OK")

    test_tnd_lse_shape_and_matches_bnsd((2, 512, 128), torch.float16)
    log.info("TND LSE == BNSD LSE (fp16) OK")

    if not _IS_ARCH35:
        test_fp8_on_v220_raises_clear_error()
        test_bsnd_on_v220_raises_clear_error()
        log.info("SoC guard raises clean errors for FP8/BSND on v220 OK")

    log.info("Smoke tests passed.")
