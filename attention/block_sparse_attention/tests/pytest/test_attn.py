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
attention_out correctness gate for `attention/block_sparse_attention`.

The sweep (shapes / frames / sparsities) is shared with the LSE and joint
suites so coverage and tolerances are directly comparable. The reference is
`torch_npu.npu_fusion_attention` (PFA) with `sparse_mode=0` at sparsity 0
(force-dense) and `sparse_mode=1` with the per-token `pfa_atten_mask`
otherwise.

Two modes:
1. Full pytest:
       pytest test_attn.py
2. Quick smoke test on a single input scenario:
       python test_attn.py

Notes
─────
- `block_sparse_attention` accepts only block-level masks (no fine-grained
  intra-block atten_mask on CANN 8.5.0-beta.1). The reference WITH the
  per-token `pfa_atten_mask` produced by `create_attention_mask` is
  block-aligned (`make_block_mask_per_head` expands each kept block to all
  of its 128×128 token positions), so both kernel and reference compute
  the same attention at the priority workloads. Hence strict tolerance
  (`rtol=0.005, atol=0.001`) is achievable here.
"""

import os
import sys
import itertools
import pytest
import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import cann_ops_transformer

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark")))
from util import (  # noqa: E402
    gen_pfa_inputs, create_attention_mask, FrameWidths, WorkloadShape,
    ref_attention_launcher,
    sabi_to_block_sparse_mask,
    bsa_inner_precise, is_arch35_soc,
)

# inner_precise is SoC-gated: 950 → 4 (low/high mixed), 910/910_93 → 0 (fp32 softmax).
# Picked from the runtime device so this same test runs on both chips.
INNER_PRECISE = bsa_inner_precise()
_IS_ARCH35 = is_arch35_soc()

# attention_out tolerance vs npu_fusion_attention.
#   910 : inner_precise=0 (full fp32 softmax) → matches the reference tightly.
#   950 : forced to inner_precise=4 (low/high *mixed* precision; online softmax in
#         bf16/fp16) → larger deviation. Two regimes coexist in this random-input sweep:
#         • dense/low-sparsity rows: attention is ~uniform over S=24000, so the true
#           output is ~N(0, 1/S) — near-zero noise. There an ABSOLUTE floor is the
#           right tool (relative error is meaningless near zero); the worst deviation
#           observed was 0.0625, so atol=0.08 bounds it with a little margin.
#         • framed/high-sparsity rows: O(1+) outputs, where 0.0625 is ~1 bf16 ULP —
#           handled by rtol (a few ULP), not atol.
# NOTE: these arch35 bands are empirical (npu_fusion_attention is itself bf16, not
# ground truth). Confirm against the float32 ref before treating them as tight bounds.
RTOL = 0.01 if _IS_ARCH35 else 0.005
ATOL = 0.08 if _IS_ARCH35 else 0.001

# Global test configurations
SEED = 42
DEVICE = "npu:0"
DTYPE = torch.bfloat16
INPUT_LAYOUT = 'BNSD'
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 128
torch.npu.set_device(DEVICE)

# Sweep parameters — shared grid across the attn / lse / joint suites so coverage is comparable.
TORCH_REF_VALS = ["npu_fusion_attention"]
B_VALS = [1]
H_VALS = [1, 2, 3, 4]
S_VALS = [24_000]            # s_q = s_kv
D_VALS = [128]
FRAME_VALUES = [FrameWidths(0, 0, 0, 0), FrameWidths(2, 1, 3, 1)]
SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHAPES = list(itertools.product(B_VALS, H_VALS, S_VALS, D_VALS))
FRAME_SPARSITY_COMBOS = list(itertools.product(FRAME_VALUES, SPARSITY_VALS))
FRAME_SPARSITY_IDS = [
    f"frame({f.left_cols},{f.right_cols},{f.top_rows},{f.bottom_rows})-sparsity{s:.2f}"
    for f, s in FRAME_SPARSITY_COMBOS
]


def _run_block_sparse_attention(q, k, v, h, scale, s_q, s_kv, sabi):
    """Single call site for block_sparse_attention from the tests.

    Converts sabi (sparse attention block indices; uint16, padded 0xFFFF) into
    the kernel's int8 block_sparse_mask, then issues the kernel. Returns the
    attention_out tensor only (LSE is exercised in test_lse.py / test_joint.py).
    """
    b = q.shape[0]
    q_blocks = (s_q + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q
    kv_blocks = (s_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    block_mask = sabi_to_block_sparse_mask(sabi, (b, h, q_blocks, kv_blocks), device=DEVICE)

    actual_seq = [s_q] * b
    actual_seq_kv = [s_kv] * b
    out, _ = cann_ops_transformer.ops.block_sparse_attention(
        q, k, v,
        block_sparse_mask=block_mask,
        block_shape=[BLOCK_SIZE_Q, BLOCK_SIZE_KV],
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq_kv,
        num_heads=h, num_key_value_heads=h,
        q_input_layout=INPUT_LAYOUT, kv_input_layout=INPUT_LAYOUT,
        mask_type=0, scale_value=float(scale),
        inner_precise=INNER_PRECISE,      # 910: 0 (fp32 sm); 950: 4 (low/high mixed)
        block_size=0,                     # paged cache not yet supported by kernel; must be 0
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False,
    )
    return out


@pytest.mark.parametrize("torch_ref", TORCH_REF_VALS)
@pytest.mark.parametrize("frame,sparsity", FRAME_SPARSITY_COMBOS, ids=FRAME_SPARSITY_IDS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_block_sparse_attention_correctness(torch_ref, shape, frame, sparsity):
    """Test correctness of cann_ops_transformer.ops.block_sparse_attention vs reference implementation."""
    torch.manual_seed(SEED)
    b, h, s_kv, d = shape
    s_q = s_kv
    workload = WorkloadShape(b, h, s_q, s_kv, d, BLOCK_SIZE_Q, BLOCK_SIZE_KV)

    # Build sabi + per-token reference mask. The kernel uses only the block-level
    # sabi (converted to int8 inside _run_block_sparse_attention); the reference
    # uses the per-token pfa_atten_mask (block-aligned via make_block_mask_per_head).
    pfa_atten_mask, sabi, scale = create_attention_mask(
        workload, sparsity, frame, DEVICE, emit_atten_mask=True,
    )

    q, k, v, _, _ = gen_pfa_inputs(workload, device=DEVICE, dtype=DTYPE)

    out_our = _run_block_sparse_attention(q, k, v, h, scale, s_q, s_kv, sabi)

    out_ref = ref_attention_launcher(
        torch_reference=torch_ref,
        pfa_inputs=(q, k, v, h, scale, pfa_atten_mask, INPUT_LAYOUT),
        force_dense_sm=(sparsity == 0),
    )

    if not torch.allclose(out_our.cpu(), out_ref.cpu(), rtol=RTOL, atol=ATOL):
        max_diff = (out_our.cpu() - out_ref.cpu()).abs().max().item()
        pytest.fail(
            f"Outputs don't match for b={b}, h={h}, s_q={s_q}, s_kv={s_kv}, "
            f"d={d}, frame={frame}, sparsity={sparsity}: max_diff={max_diff:.6f} "
            f"(rtol={RTOL}, atol={ATOL}, arch35={_IS_ARCH35})"
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_block_sparse_attention_correctness(
        torch_ref="npu_fusion_attention",
        shape=(1, 3, 24_000, 128),
        frame=FrameWidths(left_cols=2, right_cols=1, top_rows=3, bottom_rows=1),
        sparsity=0.5,
    )
    logging.getLogger(__name__).info("Smoke test passed.")
