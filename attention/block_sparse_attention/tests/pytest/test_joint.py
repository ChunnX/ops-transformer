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
test_joint.py — verifies attention_out and softmax_lse from the *same*
block_sparse_attention call.

Every test makes a single `cann_ops_transformer.ops.block_sparse_attention(..., softmax_lse_flag=True)`
call and asserts both outputs simultaneously:

  • dense  (all-ones mask)             any H, bfloat16 + float16
        — attn_out  vs FIAS (tight)
        — softmax_lse vs FIAS (tight)
  • sparse (block-aligned sabi)         H=1 bfloat16
        — attn_out  vs PFA  sparse_mode=1 (tight)
        — softmax_lse vs Python float32 ref (definitionally correct;
          FIAS sparse_mode=1 diverges by ~0.06–0.16 — see test_lse.py)

Sparse tests are limited to H=1 because FIAS sparse_mode=1 requires a
[B,1,S,S] mask (H>1 hangs the NPU), and we want the *output* check to
use FIAS / PFA where they're known-correct.

Run with:
    pytest test_joint.py -v
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

SEED = 42
DEVICE = "npu:0"
INPUT_LAYOUT = "BNSD"
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 128
_SABI_PAD = SABI_PAD  # 0xFFFF padding sentinel — single source of truth in util
torch.npu.set_device(DEVICE)
# inner_precise is SoC-gated: 950 → 4 (low/high mixed), 910/910_93 → 0 (fp32 softmax).
INNER_PRECISE = bsa_inner_precise()
_IS_ARCH35 = is_arch35_soc()

# 910 baselines (inner_precise=0, full fp32 softmax → matches the references tightly).
OUT_ATOL        = 1e-3   # dense attention_out vs FIAS
SPARSE_OUT_ATOL = 1e-3   # sparse attention_out vs PFA — block-boundary headroom
LSE_ATOL        = 1e-3   # dense softmax_lse vs FIAS (tight)
SPARSE_LSE_REF_ATOL = 0.05  # sparse softmax_lse vs python float32 ref (bf16 score-acc headroom)

# A5 (arch35) is forced to inner_precise=4 (online softmax in the input dtype, not fp32),
# so bf16 deviates from the fp32-accurate references. The references stay ground truth;
# only the bf16-on-A5 band widens (fp16/fp32 and the whole 910 path keep the baselines):
#   • attention_out → ~1 bf16 ULP (≤ ~0.0625 over this sweep): match test_attn (0.01/0.08).
#   • softmax_lse   → ~1% drift over long reductions: a 1.5% relative band.
ARCH35_OUT_RTOL, ARCH35_OUT_ATOL = 0.01, 0.08
LSE_BF16_A5_RTOL = 0.015

NO_FRAME = FrameWidths(0, 0, 0, 0)

# Dense: varied H (1,2,4,8) and S (1024–16384); both dtypes → 14 tests
DENSE_SHAPES = [
    (1, 1,  1024, 128),
    (1, 2,  2048, 128),
    (1, 4,  4096, 128),
    (1, 8,  4096, 128),     # high head count
    (1, 4,  8192, 128),     # longer sequence, moderate H
    (1, 2, 12288, 128),
    (1, 8, 16384, 128),
]
DENSE_DTYPES = [torch.bfloat16, torch.float16]

# Sparse: H=1, bf16. Explicit (shape, sparsity) pairs chosen so that
# kept_kv_blocks >= 15 — at fewer kept blocks the softmax denominator becomes
# tiny and block-boundary rounding exceeds SPARSE_OUT_ATOL.
SPARSE_CASES = [
    ((1, 1,  4096, 128), 0.3),
    ((1, 1,  4096, 128), 0.5),
    ((1, 1,  8192, 128), 0.3),
    ((1, 1,  8192, 128), 0.5),
    ((1, 1,  8192, 128), 0.7),
    ((1, 1, 16384, 128), 0.7),
]


# ── Kernel helpers ────────────────────────────────────────────────────────────

def _all_ones_block_mask(B, H, S, device):
    q_blocks = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    return torch.ones(B, H, q_blocks, kv_blocks, dtype=torch.int8, device=device)


def _sabi_to_int8_mask(sabi, B, H, S):
    """Thin wrapper over util.sabi_to_block_sparse_mask: derive the (square) test
    block grid from S and delegate to the shared converter, so the sabi → int8
    logic lives in exactly one place."""
    q_blocks = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    return sabi_to_block_sparse_mask(
        sabi, (B, H, q_blocks, kv_blocks), device=str(sabi.device)
    )


def _run_kernel(q, k, v, H, scale, S, block_mask):
    B = q.shape[0]
    actual_seq = [S] * B
    out, lse = cann_ops_transformer.ops.block_sparse_attention(
        q, k, v,
        block_sparse_mask=block_mask,
        block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        num_heads=H, num_key_value_heads=H,
        q_input_layout=INPUT_LAYOUT, kv_input_layout=INPUT_LAYOUT,
        mask_type=0, scale_value=float(scale),
        inner_precise=INNER_PRECISE,
        block_size=0,  # paged cache not yet supported by kernel; must be 0
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)   # [B,H,S,1] → [B,H,S]


def _run_fias_dense(q, k, v, H, scale, S):
    B = q.shape[0]
    actual_seq = [S] * B
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=H, scale=scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=H,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


def _run_pfa_sparse(q, k, v, H, scale, pfa_atten_mask):
    """PFA sparse_mode=1 — used as the attention_out reference (block-aligned)."""
    return torch_npu.npu_fusion_attention(
        q, k, v,
        head_num=H, input_layout=INPUT_LAYOUT, scale=scale,
        pre_tockens=0, next_tockens=0,
        atten_mask=pfa_atten_mask, sparse_mode=1,
    )[0]


def _lse_reference_sparse(q, k, scale, sabi_cpu):
    """Python float32 LSE reference — gathers KV rows listed in sabi and
    logsumexp's over them in float32. Used because FIAS sparse_mode=1 LSE
    diverges from `block_sparse_attention`'s LSE by ~0.06–0.16
    (see test_lse.py)."""
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


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", DENSE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_joint_dense(shape, dtype):
    """Dense: attention_out and softmax_lse both match FIAS in one call."""
    torch.manual_seed(SEED)
    B, H, S, D = shape
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    mask = _all_ones_block_mask(B, H, S, DEVICE)
    our_out, our_lse = _run_kernel(q, k, v, H, scale, S, mask)
    ref_out, ref_lse = _run_fias_dense(q, k, v, H, scale, S)

    assert our_lse.shape == torch.Size([B, H, S]), f"lse shape wrong: {our_lse.shape}"
    assert our_lse.dtype == torch.float32,         f"lse dtype wrong: {our_lse.dtype}"

    bf16_a5 = _IS_ARCH35 and dtype == torch.bfloat16
    out_rtol, out_atol = (ARCH35_OUT_RTOL, ARCH35_OUT_ATOL) if bf16_a5 else (0.005, OUT_ATOL)
    lse_rtol = LSE_BF16_A5_RTOL if bf16_a5 else 1e-3
    if not torch.allclose(our_out.cpu(), ref_out.cpu(), rtol=out_rtol, atol=out_atol):
        max_diff = (our_out.cpu() - ref_out.cpu()).abs().max().item()
        pytest.fail(f"attention_out: shape={shape} dtype={dtype} "
                    f"max_diff={max_diff:.5f} > rtol={out_rtol} atol={out_atol}")

    if not torch.allclose(our_lse.cpu(), ref_lse.cpu().float(), rtol=lse_rtol, atol=LSE_ATOL):
        max_diff = (our_lse.cpu() - ref_lse.cpu().float()).abs().max().item()
        pytest.fail(f"softmax_lse:  shape={shape} dtype={dtype} "
                    f"max_diff={max_diff:.5f} > rtol={lse_rtol} atol={LSE_ATOL}")


@pytest.mark.parametrize("shape,sparsity", SPARSE_CASES,
                          ids=[f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}-sp{int(sp*100):02d}pct"
                               for s, sp in SPARSE_CASES])
def test_joint_sparse(shape, sparsity):
    """Sparse (H=1, bf16): attention_out vs PFA, softmax_lse vs python float32 ref.

    PFA sparse_mode=1 + pfa_atten_mask gives a block-aligned attention reference
    that matches `block_sparse_attention`'s `block_sparse_mask` semantics exactly.

    Python float32 ref is used for LSE because FIAS sparse_mode=1 LSE
    definitionally diverges from the per-block logsumexp (~0.06–0.16). The
    python ref computes exactly what `block_sparse_attention` reports.
    """
    torch.manual_seed(SEED)
    B, H, S, D = shape
    assert H == 1
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(D)

    workload = WorkloadShape(B, H, S, S, D, BLOCK_SIZE_Q, BLOCK_SIZE_KV)
    pfa_atten_mask, sabi, _ = create_attention_mask(
        workload, sparsity, NO_FRAME, "cpu", emit_atten_mask=True,
    )
    sabi_cpu = sabi
    sabi_dev = sabi.to(DEVICE)
    pfa_atten_mask = pfa_atten_mask.to(DEVICE)

    q = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=dtype, device=DEVICE)

    mask = _sabi_to_int8_mask(sabi_dev, B, H, S)
    our_out, our_lse = _run_kernel(q, k, v, H, scale, S, mask)
    pfa_out          = _run_pfa_sparse(q, k, v, H, scale, pfa_atten_mask)
    ref_lse          = _lse_reference_sparse(q, k, scale, sabi_cpu)

    assert our_lse.shape == torch.Size([B, H, S]), f"lse shape wrong: {our_lse.shape}"
    assert our_lse.dtype == torch.float32,         f"lse dtype wrong: {our_lse.dtype}"

    # H==1 bf16 throughout this test; widen only on A5 (see module header).
    bf16_a5 = _IS_ARCH35
    out_rtol, out_atol = (ARCH35_OUT_RTOL, ARCH35_OUT_ATOL) if bf16_a5 else (0.005, SPARSE_OUT_ATOL)
    lse_rtol = LSE_BF16_A5_RTOL if bf16_a5 else 1e-3
    if not torch.allclose(our_out.cpu(), pfa_out.cpu(), rtol=out_rtol, atol=out_atol):
        max_diff = (our_out.cpu() - pfa_out.cpu()).abs().max().item()
        pytest.fail(f"attention_out: shape={shape} sparsity={sparsity} "
                    f"max_diff={max_diff:.5f} > rtol={out_rtol} atol={out_atol}")

    if not torch.allclose(our_lse.cpu(), ref_lse, rtol=lse_rtol, atol=SPARSE_LSE_REF_ATOL):
        max_diff = (our_lse.cpu() - ref_lse).abs().max().item()
        pytest.fail(f"softmax_lse:  shape={shape} sparsity={sparsity} "
                    f"max_diff={max_diff:.5f} > rtol={lse_rtol} atol={SPARSE_LSE_REF_ATOL}")


# ── __main__ smoke tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    test_joint_dense((1, 2, 2048, 128), torch.bfloat16)
    log.info("dense joint bf16 passed")

    test_joint_dense((1, 2, 2048, 128), torch.float16)
    log.info("dense joint fp16 passed")

    test_joint_sparse((1, 1, 4096, 128), 0.5)
    log.info("sparse joint H=1 bf16 sp=50% passed")
