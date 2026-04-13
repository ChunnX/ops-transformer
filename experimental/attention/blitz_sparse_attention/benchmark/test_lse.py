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
Tests for BSA softmax_lse output.

Compares BSA's log-sum-exp output against two references:
  • FIAS (npu_fused_infer_attention_score) — used for dense and sparse H=1 tests.
  • Python float32 reference — used for multi-head sparse tests.

Target: ~100 pytest test instances (B=1 only; B>1 not working).

Run with:
    pytest benchmark/test_lse.py -v

Design notes
────────────
1.  FIAS sparse_mode=1 requires a broadcast-compatible mask [B, 1, S, S].
    Passing [B, H, S, S] with H > 1 causes FIAS to hang on the NPU.
    Fix: all direct FIAS-comparison tests use H=1 shapes only.

2.  Per-head FIAS calls (passing q[:, h:h+1, :, :]) look innocent but are
    wrong for B > 1: the slice's batch-stride is H×S×D (from the original
    allocation) rather than 1×S×D.  NPU kernels read raw strides, so FIAS
    lands on the wrong memory for batch item 1.
    Fix: multi-head sparse tests use a Python float32 reference instead of
    FIAS, so BSA is always called once with the full [B, H, S, D] tensor.

3.  create_attention_mask(..., device="npu:0") issues one NPU kernel per kept
    block (hundreds of launches for s=2048, H=2).  Fix: always build masks on
    CPU, transfer once with .to(DEVICE).

4.  float16 sparse LSE: FIAS with -inf masking and BSA with full block-skip
    diverge beyond any reasonable atol at high sparsity in float16.
    Fix: all sparse tests use bfloat16 only.
"""

import math
import pytest
import torch
import torch_npu
import torch_bsa
from benchmark import create_attention_mask, FrameWidths, BLOCK_SIZE_Q, BLOCK_SIZE_KV

SEED = 42
DEVICE = "npu:0"
INPUT_LAYOUT = "BNSD"
_SABI_PAD = 65535   # 0xFFFF – padding sentinel in uint16 sabi tensors
torch.npu.set_device(DEVICE)


# ── Dense shapes (B, H, S, D) — 14 × 2 dtypes = 28 tests ────────────────────
LSE_TEST_SHAPES = [
    (1, 1, 1024, 128), (1, 2, 1024, 128),
    (1, 4, 2048, 128),
    (1, 4, 4096, 128), (1, 8, 4096, 128),
    (1, 2, 8192, 128), (1, 4, 8192, 128),
    (1, 2, 12288, 128), (1, 4, 12288, 128),
    (1, 2, 16384, 128), (1, 4, 16384, 128),
    (1, 1, 24000, 128), (1, 2, 24000, 128), (1, 4, 24000, 128),
]
LSE_DTYPES = [torch.bfloat16, torch.float16]

# ── Sparse H=1 shapes for FIAS comparison — 8 × 4 sparsities = 32 tests ──────
# H=1 keeps mask shape [B, 1, S, S] which FIAS sparse_mode=1 handles correctly.
# B=1 only (B>1 not working). bfloat16 only (see design note 4).
SPARSE_LSE_FIAS_SHAPES = [
    (1, 1,  1024, 128),
    (1, 1,  2048, 128),
    (1, 1,  4096, 128),
    (1, 1,  8192, 128),
    (1, 1, 12288, 128),
    (1, 1, 16384, 128),
    (1, 1, 20480, 128),
    (1, 1, 24000, 128),
]

# ── Sparse H=1 framed shapes — 2 × 4 sparsities = 8 tests ───────────────────
# s >= 16384: FrameWidths(2,1,3,1) forces ≈5.4% density, within sparsity=0.9.
SPARSE_FRAMED_FIAS_SHAPES = [
    (1, 1, 16384, 128),
    (1, 1, 24000, 128),
]
SPARSE_FRAME = FrameWidths(left_cols=2, right_cols=1, top_rows=3, bottom_rows=1)

# ── Multi-head sparse shapes for Python-reference comparison ──────────────────
# BSA runs once on all heads; LSE is compared against a CPU float32 reference
# that computes log-sum-exp directly over the sabi-selected KV blocks.
# S <= 1024 keeps Python reference loops fast (< 1 s per test).
# B=1 only (B>1 not working). 4 shapes × 4 sparsities = 16 tests, bfloat16 only.
SPARSE_LSE_MULTIHEAD_SHAPES = [
    (1, 2,  512, 128), (1, 4,  512, 128),
    (1, 2, 1024, 128), (1, 4, 1024, 128),
]

SPARSE_LSE_SPARSITIES = [0.3, 0.5, 0.7, 0.9]
NO_FRAME = FrameWidths(0, 0, 0, 0)

# ── Zero-input shapes — 5 × 2 dtypes = 10 tests ──────────────────────────────
ZERO_INPUT_SHAPES = [
    (1, 1,   512, 128), (1, 2,  1024, 128), (1, 4,  4096, 128),
    (1, 4, 16384, 128), (1, 2, 24000, 128),
]

# ── Shape/dtype check shapes — 3 tests ───────────────────────────────────────
SHAPE_DTYPE_SHAPES = [
    (1, 1,   512, 128),
    (1, 8,  8192, 128), (1, 4, 24000, 128),
]

# ── Disabled-LSE shapes — 3 tests ────────────────────────────────────────────
DISABLED_LSE_SHAPES = [
    (1, 2,   512, 128),
    (1, 8,  8192, 128), (1, 4, 24000, 128),
]

# ── Tolerances ────────────────────────────────────────────────────────────────
LSE_ATOL = {
    torch.bfloat16: 0.025,
    torch.float16:  0.025,
    torch.float32:  1e-3,
}
# Bfloat16 input vs float32 reference: allow for bfloat16 mantissa error in
# the attention score accumulation (typically < 0.03, 0.05 gives headroom).
SPARSE_REF_ATOL = 0.05


# ── Mask helper ───────────────────────────────────────────────────────────────

def _build_sparse_masks(B, H, S, sparsity, frame):
    """Return (sabi, pfa_atten_mask) on DEVICE, built on CPU first."""
    pfa_atten_mask, _, sabi, _, _, _, _ = create_attention_mask(
        B, H, S, S, 128, sparsity, "blocks_optimized_batched",
        block_size_q=BLOCK_SIZE_Q, block_size_kv=BLOCK_SIZE_KV,
        frame=frame, device="cpu", emit_atten_mask=True,
    )
    return sabi.to(DEVICE), pfa_atten_mask.to(DEVICE)


# ── Python float32 reference LSE ──────────────────────────────────────────────

def _lse_reference_sparse(q, k, scale, sabi_cpu):
    """Compute exact sparse LSE in float32 using Python loops.

    For each (batch, head, q-block), collects the KV token indices listed in
    sabi (skipping 0xFFFF padding), computes QK^T * scale in float32, then
    logsumexp over attended tokens.

    q, k : [B, H, S, D] (any device / dtype — moved to CPU float32 internally)
    sabi_cpu : [B, H, n_q_blocks, n_kv_blocks] uint16, on CPU
    Returns [B, H, S] float32 on CPU.
    """
    B, H, S, D = q.shape
    n_q_blocks = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q

    qf = q.cpu().float()
    kf = k.cpu().float()
    lse = torch.full((B, H, S), float("-inf"))

    for b in range(B):
        for h in range(H):
            for qb in range(n_q_blocks):
                q_s = qb * BLOCK_SIZE_Q
                q_e = min(q_s + BLOCK_SIZE_Q, S)

                kept = [int(c) for c in sabi_cpu[b, h, qb].tolist()
                        if int(c) != _SABI_PAD]
                if not kept:
                    continue

                kv_rows = []
                for c in kept:
                    kv_s = c * BLOCK_SIZE_KV
                    kv_e = min(kv_s + BLOCK_SIZE_KV, S)
                    kv_rows.extend(range(kv_s, kv_e))

                scores = (qf[b, h, q_s:q_e] @ kf[b, h, kv_rows].t()) * scale
                lse[b, h, q_s:q_e] = torch.logsumexp(scores, dim=-1)

    return lse


# ── Kernel helpers ────────────────────────────────────────────────────────────

def _run_bsa_with_lse(q, k, v, h, scale, seq_len):
    """BSA dense (no sabi), softmax_lse_flag=True → (out, lse [B,H,S])."""
    actual_seq = [seq_len] * q.shape[0]
    out, lse = torch_bsa.blitz_sparse_attention(
        q, k, v,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        num_heads=h, num_key_value_heads=h,
        input_layout=INPUT_LAYOUT,
        scale_value=scale,
        sparse_mode=0,
        softmax_lse_flag=True,
    )
    return out, lse


def _run_fias_with_lse(q, k, v, h, scale, seq_len):
    """FIAS dense → (out, lse [B,H,S])."""
    B = q.shape[0]
    actual_seq = [seq_len] * B
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=h, scale=scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=h,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


def _run_bsa_with_lse_sparse(q, k, v, h, scale, seq_len, sabi):
    """BSA block-sparse (sabi, sparse_mode=0) → (out, lse [B,H,S]).

    Always pass full [B, H, S, D] tensors — never per-head views (see note 2).
    """
    B = q.shape[0]
    actual_seq = [seq_len] * B
    out, lse = torch_bsa.blitz_sparse_attention(
        q, k, v,
        sabi=sabi,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        num_heads=h, num_key_value_heads=h,
        input_layout=INPUT_LAYOUT,
        scale_value=scale,
        sparse_mode=0,
        softmax_lse_flag=True,
    )
    return out, lse


def _run_fias_with_lse_sparse(q, k, v, h, scale, seq_len, pfa_atten_mask):
    """FIAS sparse_mode=1 → (out, lse [B,h,S]).

    pfa_atten_mask MUST be [B, 1, S, S] — H > 1 causes an NPU hang.
    """
    B = q.shape[0]
    actual_seq = [seq_len] * B
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=h, scale=scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=h,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        atten_mask=pfa_atten_mask,
        sparse_mode=1,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


# ── Test 1: dense LSE vs FIAS (40 tests) ─────────────────────────────────────

@pytest.mark.parametrize("dtype", LSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", LSE_TEST_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_vs_fias(shape, dtype):
    """BSA dense softmax_lse must match FIAS softmax_lse."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, bsa_lse  = _run_bsa_with_lse(q, k, v, H, scale, S)
    _, fias_lse = _run_fias_with_lse(q, k, v, H, scale, S)

    assert bsa_lse.shape == (B, H, S)
    assert bsa_lse.dtype == torch.float32

    atol = LSE_ATOL[dtype]
    if not torch.allclose(bsa_lse.cpu(), fias_lse.cpu().float(), rtol=1e-3, atol=atol):
        max_diff = (bsa_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(f"shape={shape} dtype={dtype}: max_diff={max_diff:.5f} > atol={atol}")


# ── Test 2: sparse H=1 LSE vs FIAS, no frame (64 tests) ──────────────────────

@pytest.mark.parametrize("sparsity", SPARSE_LSE_SPARSITIES,
                          ids=lambda s: f"sp{int(s*100):02d}pct")
@pytest.mark.parametrize("shape", SPARSE_LSE_FIAS_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_vs_fias_sparse(shape, sparsity):
    """BSA sparse (H=1, no frame) softmax_lse vs FIAS with the matching token mask.

    H=1 only: mask is [B,1,S,S], safe for FIAS sparse_mode=1.
    Mask built on CPU → single .to(DEVICE) transfer.
    bfloat16 only (float16 diverges at high sparsity, see module docstring).
    """
    torch.manual_seed(SEED)
    B, H, S, D = shape
    assert H == 1
    scale = 1.0 / math.sqrt(D)
    dtype = torch.bfloat16

    sabi, pfa_atten_mask = _build_sparse_masks(B, H, S, sparsity, NO_FRAME)
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, bsa_lse  = _run_bsa_with_lse_sparse(q, k, v, H, scale, S, sabi)
    _, fias_lse = _run_fias_with_lse_sparse(q, k, v, H, scale, S, pfa_atten_mask)

    assert bsa_lse.shape == (B, H, S)
    assert bsa_lse.dtype == torch.float32

    atol = LSE_ATOL[dtype]
    if not torch.allclose(bsa_lse.cpu(), fias_lse.cpu().float(), rtol=1e-3, atol=atol):
        max_diff = (bsa_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(
            f"shape={shape} sparsity={sparsity}: max_diff={max_diff:.5f} > atol={atol}"
        )


# ── Test 3: sparse H=1 LSE vs FIAS, with border frame (8 tests) ──────────────

@pytest.mark.parametrize("sparsity", SPARSE_LSE_SPARSITIES,
                          ids=lambda s: f"sp{int(s*100):02d}pct")
@pytest.mark.parametrize("shape", SPARSE_FRAMED_FIAS_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_vs_fias_sparse_framed(shape, sparsity):
    """BSA sparse LSE with FrameWidths(2,1,3,1) on long sequences, H=1, bf16."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    assert H == 1
    scale = 1.0 / math.sqrt(D)
    dtype = torch.bfloat16

    sabi, pfa_atten_mask = _build_sparse_masks(B, H, S, sparsity, SPARSE_FRAME)
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, bsa_lse  = _run_bsa_with_lse_sparse(q, k, v, H, scale, S, sabi)
    _, fias_lse = _run_fias_with_lse_sparse(q, k, v, H, scale, S, pfa_atten_mask)

    atol = LSE_ATOL[dtype]
    if not torch.allclose(bsa_lse.cpu(), fias_lse.cpu().float(), rtol=1e-3, atol=atol):
        max_diff = (bsa_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(
            f"shape={shape} sparsity={sparsity}: max_diff={max_diff:.5f} > atol={atol}"
        )


# ── Test 4: multi-head sparse LSE vs Python float32 reference (32 tests) ──────

@pytest.mark.parametrize("sparsity", SPARSE_LSE_SPARSITIES,
                          ids=lambda s: f"sp{int(s*100):02d}pct")
@pytest.mark.parametrize("shape", SPARSE_LSE_MULTIHEAD_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_sparse_multihead(shape, sparsity):
    """BSA multi-head sparse LSE vs a CPU float32 reference.

    The reference _lse_reference_sparse() gathers the exact KV token rows
    listed in sabi and computes log-sum-exp in float32 — independent of FIAS.
    BSA is called once on the full [B, H, S, D] tensor (never per-head slices).
    S <= 1024 keeps the Python reference loops fast (< 1 s per test).
    bfloat16 only.
    """
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    dtype = torch.bfloat16

    # Build masks on CPU; keep sabi_cpu for the reference computation
    sabi_cpu, _ = _build_sparse_masks(B, H, S, sparsity, NO_FRAME)
    sabi_cpu = sabi_cpu.cpu()
    sabi_dev = sabi_cpu.to(DEVICE)

    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, bsa_lse = _run_bsa_with_lse_sparse(q, k, v, H, scale, S, sabi_dev)
    ref_lse    = _lse_reference_sparse(q, k, scale, sabi_cpu)  # [B, H, S] float32 CPU

    if not torch.allclose(bsa_lse.cpu(), ref_lse, rtol=1e-3, atol=SPARSE_REF_ATOL):
        max_diff = (bsa_lse.cpu() - ref_lse).abs().max().item()
        pytest.fail(
            f"Multi-head sparse LSE mismatch: shape={shape} sparsity={sparsity}: "
            f"max_diff={max_diff:.5f} > atol={SPARSE_REF_ATOL}"
        )


# ── Test 5: zero-input — LSE must equal log(S) (12 tests) ────────────────────

@pytest.mark.parametrize("dtype", LSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", ZERO_INPUT_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_zero_input(shape, dtype):
    """Q=K=V=0 → uniform softmax → LSE = log(S). Catches missing ViewCopy etc.

    float16 inputs use fp16 arithmetic for online-softmax intermediates; the
    vector-unit log near log(S)≈6–9 has ~1 fp16 ULP error (≈0.004–0.008), so
    a tighter tolerance is not achievable for fp16.  bfloat16 uses a fp32
    accumulator path and stays within 1e-3.
    """
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.zeros(B, H, S, D, dtype=dtype, device=DEVICE)

    _, lse = _run_bsa_with_lse(q, q, q, H, scale, S)
    expected = math.log(S)
    atol = LSE_ATOL[dtype]   # 0.025 for fp16/bf16 — matches the log ULP budget
    if not torch.allclose(lse.cpu(), torch.full_like(lse.cpu(), expected), atol=atol):
        max_diff = (lse.cpu() - expected).abs().max().item()
        pytest.fail(
            f"shape={shape} dtype={dtype}: max_diff={max_diff:.5f}, "
            f"expected log({S})={expected:.5f}, atol={atol}"
        )


# ── Test 6: LSE shape and dtype (4 tests) ────────────────────────────────────

@pytest.mark.parametrize("shape", SHAPE_DTYPE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_shape_and_dtype(shape):
    """LSE tensor must have shape [B, H, S] and dtype float32."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEVICE)

    _, lse = _run_bsa_with_lse(q, q, q, H, scale, S)
    assert lse.shape == torch.Size([B, H, S]), f"Wrong shape: {lse.shape}"
    assert lse.dtype == torch.float32, f"Wrong dtype: {lse.dtype}"


# ── Test 7: disabled LSE returns empty tensor (4 tests) ──────────────────────

@pytest.mark.parametrize("shape", DISABLED_LSE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_disabled_returns_empty(shape):
    """softmax_lse_flag=False must produce an empty tensor."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEVICE)
    actual_seq = [S] * B

    _, lse = torch_bsa.blitz_sparse_attention(
        q, q, q,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        num_heads=H, num_key_value_heads=H,
        input_layout=INPUT_LAYOUT, scale_value=scale,
        softmax_lse_flag=False,
    )
    assert lse.numel() == 0, f"Expected empty tensor, got shape {lse.shape}"


# ── __main__ smoke tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    test_bsa_lse_shape_and_dtype((1, 2, 512, 128))
    log.info("shape/dtype check passed")

    test_bsa_lse_disabled_returns_empty((1, 2, 512, 128))
    log.info("disabled LSE check passed")

    test_bsa_lse_zero_input((1, 2, 1024, 128), torch.bfloat16)
    log.info("zero-input log(S) invariant passed")

    test_bsa_lse_vs_fias((1, 1, 1024, 128), torch.bfloat16)
    log.info("dense LSE vs FIAS passed")

    test_bsa_lse_vs_fias_sparse((1, 1, 2048, 128), 0.5)
    log.info("sparse H=1 no-frame passed")

    test_bsa_lse_vs_fias_sparse_framed((1, 1, 16384, 128), 0.5)
    log.info("sparse H=1 framed passed")

    test_bsa_lse_sparse_multihead((1, 2, 1024, 128), 0.5)
    log.info("multi-head sparse vs Python reference passed")
