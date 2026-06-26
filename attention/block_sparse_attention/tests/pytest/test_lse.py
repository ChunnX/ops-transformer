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
Tests for `block_sparse_attention` softmax_lse output.

Compares LSE against two references:
  • FIAS (`npu_fused_infer_attention_score`) — used for dense tests.
  • Python float32 reference — used for sparse multi-head tests.

LSE shape note
──────────────
`block_sparse_attention` returns LSE shaped `[B, H, S, 1]` (the kernel
infershape hardcodes `LSE_DIM_D=1`). The python interface returns this
as-is; tests squeeze the trailing dim before comparing against FIAS's
`[B, H, S, 1]` (also squeezed via `.squeeze(-1)` in the launcher).

Sparse vs FIAS divergence
─────────────────────────
Empirically, `block_sparse_attention`'s LSE under non-zero sparsity diverges
from FIAS's `sparse_mode=1 + atten_mask` LSE by ~0.06 (sp 0.5), ~0.12
(sp 0.8), ~0.16 (sp 0.9). Likely a definitional difference in which
positions are summed over. The Python float32 reference computes
log-sum-exp directly over the sabi-selected KV blocks — if `block_sparse_attention`
matches THAT, the FIAS divergence is FIAS-specific. The dedicated multi-head
sparse test uses the Python reference. The FIAS-comparison sparse test
uses a relaxed atol to document the divergence rather than gate on it.

Run with:
    pytest test_lse.py -v
"""

import math
import os
import sys
import pytest
import torch
import torch_npu
import cann_ops_transformer

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark")))
from util import (  # noqa: E402
    create_attention_mask, FrameWidths, WorkloadShape,
    sabi_to_block_sparse_mask, SABI_PAD,
    bsa_inner_precise, is_arch35_soc,
)

# inner_precise is SoC-gated: 950 → 4 (low/high mixed), 910/910_93 → 0 (fp32 softmax).
INNER_PRECISE = bsa_inner_precise()
_IS_ARCH35 = is_arch35_soc()

# On A5 (arch35) the kernel is forced to inner_precise=4 — the online softmax runs in
# the input dtype, not fp32. For bf16 that drifts from the fp32-accurate reference by
# ~1% over long reductions (grows with S; fp16 stays tight). The reference stays the
# fp32 ground truth; we validate to within bf16's achievable precision with a relative
# band instead of pretending bf16 can hit 1e-4. Applied to bf16 on A5 only — fp16/fp32
# and the entire 910 path keep their tight tolerances.
LSE_BF16_A5_RTOL = 0.015  # ~1.5%, covers the ~0.097 (≈1%) max drift seen at S=24000

SEED = 42
DEVICE = "npu:0"
INPUT_LAYOUT = "BNSD"
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 128
_SABI_PAD = SABI_PAD  # 0xFFFF padding sentinel — single source of truth in util
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

# ── Sparse H=1 shapes for FIAS comparison — bfloat16 only (see design notes) ─
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

# ── Sparse H=1 framed shapes ─────────────────────────────────────────────────
SPARSE_FRAMED_FIAS_SHAPES = [
    (1, 1, 16384, 128),
    (1, 1, 24000, 128),
]
SPARSE_FRAME = FrameWidths(left_cols=2, right_cols=1, top_rows=3, bottom_rows=1)

# ── Multi-head sparse shapes for Python-reference comparison ─────────────────
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
    torch.bfloat16: 1e-4,
    torch.float16:  1e-4,
    torch.float32:  1e-3,
}
# block_sparse_attention sparse-vs-FIAS LSE diverges by ~0.06–0.16 (definitional);
# this atol documents it. Tests fail only if numerically non-finite or the gap
# grows past this band.
SPARSE_FIAS_LSE_ATOL = 0.25
# Bfloat16 input vs Python float32 reference: ~0.05 headroom is enough for the
# bfloat16 score-accumulation mantissa error.
SPARSE_REF_ATOL = 0.05


# ── Mask helper ───────────────────────────────────────────────────────────────

def _build_sparse_masks(B, H, S, sparsity, frame):
    """Return (sabi, pfa_atten_mask) on DEVICE, built on CPU first."""
    workload = WorkloadShape(B, H, S, S, 128, BLOCK_SIZE_Q, BLOCK_SIZE_KV)
    pfa_atten_mask, sabi, _ = create_attention_mask(
        workload, sparsity, frame, "cpu", emit_atten_mask=True,
    )
    return sabi.to(DEVICE), pfa_atten_mask.to(DEVICE)


def _sabi_to_int8_mask(sabi, B, H, S):
    """Thin wrapper over util.sabi_to_block_sparse_mask: derive the (square) test
    block grid from S and delegate to the shared converter, so the sabi → int8
    logic lives in exactly one place."""
    q_blocks = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    return sabi_to_block_sparse_mask(
        sabi, (B, H, q_blocks, kv_blocks), device=str(sabi.device)
    )


# ── Python float32 reference LSE ──────────────────────────────────────────────

def _lse_reference_sparse(q, k, scale, sabi_cpu):
    """Compute exact sparse LSE in float32 using Python loops.

    For each (batch, head, q-block), gathers KV token indices listed in sabi
    (skipping 0xFFFF padding), computes QK^T * scale in float32, then logsumexp
    over attended tokens.

    q, k     : [B, H, S, D] (any device / dtype — moved to CPU float32 internally)
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

def _run_kernel_with_lse_dense(q, k, v, h, scale, seq_len):
    """block_sparse_attention dense (all-ones mask) with softmax_lse_flag=True.

    Returns (out, lse_3d) where lse_3d is the [B, H, S] view of the
    kernel's [B, H, S, 1] output (trailing-1 squeezed).
    """
    B = q.shape[0]
    actual_seq = [seq_len] * B
    q_blocks = (seq_len + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (seq_len + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    mask = torch.ones(B, h, q_blocks, kv_blocks, dtype=torch.int8, device=q.device)

    out, lse = cann_ops_transformer.ops.block_sparse_attention(
        q, k, v,
        block_sparse_mask=mask,
        block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        num_heads=h, num_key_value_heads=h,
        q_input_layout=INPUT_LAYOUT, kv_input_layout=INPUT_LAYOUT,
        mask_type=0, scale_value=float(scale),
        inner_precise=INNER_PRECISE,
        block_size=0,  # paged cache not yet supported by kernel; must be 0
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


def _run_kernel_with_lse_sparse(q, k, v, h, scale, seq_len, sabi):
    """block_sparse_attention sparse (sabi → int8 mask), softmax_lse_flag=True."""
    B = q.shape[0]
    actual_seq = [seq_len] * B
    mask = _sabi_to_int8_mask(sabi, B, h, seq_len)

    out, lse = cann_ops_transformer.ops.block_sparse_attention(
        q, k, v,
        block_sparse_mask=mask,
        block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        num_heads=h, num_key_value_heads=h,
        q_input_layout=INPUT_LAYOUT, kv_input_layout=INPUT_LAYOUT,
        mask_type=0, scale_value=float(scale),
        inner_precise=INNER_PRECISE,
        block_size=0,  # paged cache not yet supported by kernel; must be 0
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


def _run_fias_with_lse_dense(q, k, v, h, scale, seq_len):
    """FIAS dense → (out, lse [B, H, S])."""
    B = q.shape[0]
    actual_seq = [seq_len] * B
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=h, scale=scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=h,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


def _run_fias_with_lse_sparse(q, k, v, h, scale, seq_len, pfa_atten_mask):
    """FIAS sparse_mode=1 → (out, lse [B, H, S]).
    pfa_atten_mask MUST be [B, 1, S, S] — H>1 causes an NPU hang."""
    B = q.shape[0]
    actual_seq = [seq_len] * B
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=h, scale=scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=h,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        atten_mask=pfa_atten_mask, sparse_mode=1,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


# ── Test 1: dense LSE vs FIAS (28 tests) ─────────────────────────────────────

@pytest.mark.parametrize("dtype", LSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", LSE_TEST_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_vs_fias_dense(shape, dtype):
    """block_sparse_attention dense LSE must match FIAS LSE."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, our_lse  = _run_kernel_with_lse_dense(q, k, v, H, scale, S)
    _, fias_lse = _run_fias_with_lse_dense(q, k, v, H, scale, S)

    assert our_lse.shape == (B, H, S)
    assert our_lse.dtype == torch.float32

    atol = LSE_ATOL[dtype]
    rtol = LSE_BF16_A5_RTOL if (_IS_ARCH35 and dtype == torch.bfloat16) else 1e-3
    if not torch.allclose(our_lse.cpu(), fias_lse.cpu().float(), rtol=rtol, atol=atol):
        max_diff = (our_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(f"shape={shape} dtype={dtype}: max_diff={max_diff:.5f} > rtol={rtol} atol={atol}")


# ── Test 2: sparse H=1 LSE vs FIAS (no frame) ────────────────────────────────

@pytest.mark.parametrize("sparsity", SPARSE_LSE_SPARSITIES,
                          ids=lambda s: f"sp{int(s*100):02d}pct")
@pytest.mark.parametrize("shape", SPARSE_LSE_FIAS_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_vs_fias_sparse(shape, sparsity):
    """Sparse (H=1, no frame) LSE vs FIAS.

    FIAS uses a different LSE definition for `sparse_mode=1 + atten_mask` than
    `block_sparse_attention` does for `block_sparse_mask` — they diverge by
    ~0.06–0.16 at sparsity 0.5–0.9. The atol here (`SPARSE_FIAS_LSE_ATOL`)
    is set to document this band, not to gate on FIAS-exact agreement.
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

    _, our_lse  = _run_kernel_with_lse_sparse(q, k, v, H, scale, S, sabi)
    _, fias_lse = _run_fias_with_lse_sparse(q, k, v, H, scale, S, pfa_atten_mask)

    assert our_lse.shape == (B, H, S)
    assert our_lse.dtype == torch.float32

    if not torch.allclose(our_lse.cpu(), fias_lse.cpu().float(),
                          rtol=1e-3, atol=SPARSE_FIAS_LSE_ATOL):
        max_diff = (our_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(
            f"shape={shape} sparsity={sparsity}: "
            f"max_diff={max_diff:.5f} > atol={SPARSE_FIAS_LSE_ATOL} (definitional bound)"
        )


# ── Test 3: sparse H=1 LSE vs FIAS (with border frame) ───────────────────────

@pytest.mark.parametrize("sparsity", SPARSE_LSE_SPARSITIES,
                          ids=lambda s: f"sp{int(s*100):02d}pct")
@pytest.mark.parametrize("shape", SPARSE_FRAMED_FIAS_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_vs_fias_sparse_framed(shape, sparsity):
    """Sparse LSE with FrameWidths(2,1,3,1) on long sequences, H=1, bf16.
    Same FIAS-divergence band as test_lse_vs_fias_sparse."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    assert H == 1
    scale = 1.0 / math.sqrt(D)
    dtype = torch.bfloat16

    sabi, pfa_atten_mask = _build_sparse_masks(B, H, S, sparsity, SPARSE_FRAME)
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, our_lse  = _run_kernel_with_lse_sparse(q, k, v, H, scale, S, sabi)
    _, fias_lse = _run_fias_with_lse_sparse(q, k, v, H, scale, S, pfa_atten_mask)

    if not torch.allclose(our_lse.cpu(), fias_lse.cpu().float(),
                          rtol=1e-3, atol=SPARSE_FIAS_LSE_ATOL):
        max_diff = (our_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(
            f"shape={shape} sparsity={sparsity}: "
            f"max_diff={max_diff:.5f} > atol={SPARSE_FIAS_LSE_ATOL} (definitional bound)"
        )


# ── Test 4: multi-head sparse LSE vs Python float32 reference ────────────────

@pytest.mark.parametrize("sparsity", SPARSE_LSE_SPARSITIES,
                          ids=lambda s: f"sp{int(s*100):02d}pct")
@pytest.mark.parametrize("shape", SPARSE_LSE_MULTIHEAD_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_sparse_multihead_vs_python_ref(shape, sparsity):
    """Multi-head sparse LSE vs a CPU float32 reference (independent of FIAS).
    Passing here demonstrates the kernel's LSE is correct in its own
    definition; if this passes AND test_lse_vs_fias_sparse fails its tight
    bound, the FIAS divergence is FIAS-side or definitional, not a bug here."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    dtype = torch.bfloat16

    sabi_cpu, _ = _build_sparse_masks(B, H, S, sparsity, NO_FRAME)
    sabi_cpu = sabi_cpu.cpu()
    sabi_dev = sabi_cpu.to(DEVICE)

    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    _, our_lse = _run_kernel_with_lse_sparse(q, k, v, H, scale, S, sabi_dev)
    ref_lse    = _lse_reference_sparse(q, k, scale, sabi_cpu)

    if not torch.allclose(our_lse.cpu(), ref_lse, rtol=1e-3, atol=SPARSE_REF_ATOL):
        max_diff = (our_lse.cpu() - ref_lse).abs().max().item()
        pytest.fail(
            f"Multi-head sparse LSE mismatch vs python ref: "
            f"shape={shape} sparsity={sparsity}: max_diff={max_diff:.5f} > atol={SPARSE_REF_ATOL}"
        )


# ── Test 5: zero-input — LSE must equal log(S) ───────────────────────────────

@pytest.mark.parametrize("dtype", LSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", ZERO_INPUT_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_zero_input(shape, dtype):
    """Q=K=V=0 → uniform softmax → LSE = log(S)."""
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.zeros(B, H, S, D, dtype=dtype, device=DEVICE)

    _, lse = _run_kernel_with_lse_dense(q, q, q, H, scale, S)
    expected = math.log(S)
    atol = LSE_ATOL[dtype]
    if not torch.allclose(lse.cpu(), torch.full_like(lse.cpu(), expected), atol=atol):
        max_diff = (lse.cpu() - expected).abs().max().item()
        pytest.fail(
            f"shape={shape} dtype={dtype}: max_diff={max_diff:.5f}, "
            f"expected log({S})={expected:.5f}, atol={atol}"
        )


# ── Test 6: LSE shape and dtype ───────────────────────────────────────────────

@pytest.mark.parametrize("shape", SHAPE_DTYPE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_shape_and_dtype(shape):
    """LSE tensor must be 4-D `[B, H, S, 1]` float32 (kernel's infershape spec).
    Squeezed to `[B, H, S]` in `_run_kernel_with_lse_*` for comparison."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEVICE)

    # Call the kernel directly to inspect the 4-D LSE shape (not squeezed).
    q_blocks = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    mask = torch.ones(B, H, q_blocks, kv_blocks, dtype=torch.int8, device=DEVICE)
    _, lse_raw = cann_ops_transformer.ops.block_sparse_attention(
        q, q, q,
        block_sparse_mask=mask,
        block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
        actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
        num_heads=H, num_key_value_heads=H,
        q_input_layout=INPUT_LAYOUT, kv_input_layout=INPUT_LAYOUT,
        mask_type=0, scale_value=float(scale),
        inner_precise=INNER_PRECISE,
        block_size=0,  # paged cache not yet supported by kernel; must be 0
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=True,
    )
    assert lse_raw.shape == torch.Size([B, H, S, 1]), f"Wrong shape: {lse_raw.shape}"
    assert lse_raw.dtype == torch.float32, f"Wrong dtype: {lse_raw.dtype}"


# ── Test 7: disabled LSE returns empty tensor ────────────────────────────────

@pytest.mark.parametrize("shape", DISABLED_LSE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_lse_disabled_returns_empty(shape):
    """`softmax_lse_flag=False` must produce an empty tensor (`numel()==0`)."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=DEVICE)

    q_blocks = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    mask = torch.ones(B, H, q_blocks, kv_blocks, dtype=torch.int8, device=DEVICE)
    _, lse = cann_ops_transformer.ops.block_sparse_attention(
        q, q, q,
        block_sparse_mask=mask,
        block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
        actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
        num_heads=H, num_key_value_heads=H,
        q_input_layout=INPUT_LAYOUT, kv_input_layout=INPUT_LAYOUT,
        mask_type=0, scale_value=float(scale),
        inner_precise=INNER_PRECISE,
        block_size=0,  # paged cache not yet supported by kernel; must be 0
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False,
    )
    assert lse.numel() == 0, f"Expected empty tensor, got shape {lse.shape}"


# ── __main__ smoke tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    test_lse_shape_and_dtype((1, 2, 512, 128))
    log.info("shape/dtype check passed")

    test_lse_disabled_returns_empty((1, 2, 512, 128))
    log.info("disabled LSE check passed")

    test_lse_zero_input((1, 2, 1024, 128), torch.bfloat16)
    log.info("zero-input log(S) invariant passed")

    test_lse_vs_fias_dense((1, 1, 1024, 128), torch.bfloat16)
    log.info("dense LSE vs FIAS passed")

    test_lse_vs_fias_sparse((1, 1, 2048, 128), 0.5)
    log.info("sparse H=1 no-frame passed (with definitional atol)")

    test_lse_sparse_multihead_vs_python_ref((1, 2, 1024, 128), 0.5)
    log.info("multi-head sparse vs python reference passed")
