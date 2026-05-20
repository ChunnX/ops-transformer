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
Test suite. Has two modes:
1. Full-blown pytesting:
pytest test_attn.py 

2. Quick smoke test on a single input scenario:
python test_attn.py
"""

import itertools
import pytest
import torch
import torch_npu
import torch_bsa
from benchmark import gen_pfa_inputs, create_attention_mask, FrameWidths  # data gen
from benchmark import ref_blitz_sparse_attention_launcher  # baseline kernel
from benchmark import is_block_sparse_pattern_feasible

# Global test configurations
SEED = 42
DEVICE = "npu:0"
DTYPE = torch.bfloat16
INPUT_LAYOUT = 'BNSD'
# block_shape pairs exercised by the suite. The kernel accepts any pair from
# {128, 256, 512, 1024} × {128, 256, 512, 1024} at runtime via the `block_shape`
# op-attr; all 16 combinations share the same kernel binary.
BLOCK_SHAPES = [(bsq, bskv)
                for bsq  in (128, 256, 512, 1024)
                for bskv in (128, 256, 512, 1024)]
torch.npu.set_device(DEVICE)

# Sweeped test parameters
# Reference kernel under test. One of:
#   "custom"                          - our pythonic ref_blitz_sparse_attention_fp32
#   "npu_fusion_attention"            - torch_npu.npu_fusion_attention (PFA)
#   "npu_fused_infer_attention_score" - torch_npu.npu_fused_infer_attention_score (FIAS)
TORCH_REF_VALS = ["npu_fusion_attention"]
B_VALS = [1]
H_VALS = [1, 2, 3, 4]
S_VALS = [24_000, 36000]   # s_q = s_kv (B=1 only; B>1 currently produces wrong outputs)
D_VALS = [128]   # head dimension
# `frame_kind` is a label; the actual FrameWidths is resolved per block_shape
# in the test body. `left_cols` is in BLOCK_SIZE_KV-token units and `top_rows`
# is in BLOCK_SIZE_Q-token units, so we scale both by 128 / size (rounded up,
# min 1) to keep the forced-token footprint comparable across granularities.
FRAME_KINDS = ["no_frame", "small_frame"]
_SMALL_FRAME_LEFT_BY_KV = {128: 2, 256: 1, 512: 1, 1024: 1}
_SMALL_FRAME_TOP_BY_Q   = {128: 3, 256: 2, 512: 1, 1024: 1}
FRAMES_BY_KIND_BLOCK_SHAPE = {}
for _bsq in (128, 256, 512, 1024):
    for _bskv in (128, 256, 512, 1024):
        FRAMES_BY_KIND_BLOCK_SHAPE[("no_frame", (_bsq, _bskv))] = FrameWidths(0, 0, 0, 0)
        FRAMES_BY_KIND_BLOCK_SHAPE[("small_frame", (_bsq, _bskv))] = FrameWidths(
            left_cols=_SMALL_FRAME_LEFT_BY_KV[_bskv],
            right_cols=1,
            top_rows=_SMALL_FRAME_TOP_BY_Q[_bsq],
            bottom_rows=1,
        )
SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHAPES = list(itertools.product(B_VALS, H_VALS, S_VALS, D_VALS))

# Pre-filter the (shape, frame_kind, sparsity, block_shape, torch_ref) cross
# product so that infeasible cells are NEVER collected by pytest — no skip
# noise in the run log and no risk of leaking BSA's empty-Q-row LSE sentinel
# into a downstream torch_npu op that aborts the device with an aicore
# exception (see is_block_sparse_pattern_feasible doc-string).
def _attn_param_id(p):
    shape, frame_kind, sparsity, block_shape, torch_ref = p
    return (f"b{shape[0]}-h{shape[1]}-s{shape[2]}-d{shape[3]}"
            f"-{frame_kind}-sparsity{sparsity:.2f}"
            f"-bs{block_shape[0]}x{block_shape[1]}"
            f"-{torch_ref}")

ATTN_PARAMS = [
    (shape, frame_kind, sparsity, block_shape, torch_ref)
    for shape in SHAPES
    for frame_kind in FRAME_KINDS
    for sparsity in SPARSITY_VALS
    for block_shape in BLOCK_SHAPES
    for torch_ref in TORCH_REF_VALS
    if is_block_sparse_pattern_feasible(
        s_q=shape[2], s_kv=shape[2],
        block_size_q=block_shape[0], block_size_kv=block_shape[1],
        frame=FRAMES_BY_KIND_BLOCK_SHAPE[(frame_kind, block_shape)],
        sparsity=sparsity,
    )
]
ATTN_PARAM_IDS = [_attn_param_id(p) for p in ATTN_PARAMS]


@pytest.mark.parametrize(
    "shape,frame_kind,sparsity,block_shape,torch_ref",
    ATTN_PARAMS, ids=ATTN_PARAM_IDS,
)
def test_blitz_sparse_attention_correctness(shape, frame_kind, sparsity, block_shape, torch_ref):
    """Test correctness of torch_bsa.blitz_sparse_attention vs reference implementation"""
    # Set random seed for reproducible test inputs
    torch.manual_seed(SEED)

    # Unpack input shapes
    b, h, s_kv, d = shape
    s_q = s_kv
    BLOCK_SIZE_Q, BLOCK_SIZE_KV = block_shape
    # Resolve the per-shape frame so the token-footprint stays comparable
    # across block_shape values (left_cols is in BLOCK_SIZE_KV-token units,
    # top_rows is in BLOCK_SIZE_Q-token units).
    frame = FRAMES_BY_KIND_BLOCK_SHAPE[(frame_kind, block_shape)]

    pfa_atten_mask, bsa_atten_mask, sabi, sm, scale, pre_tok, post_tok = create_attention_mask(
        b, h, s_q, s_kv, d, sparsity, frame=frame,
        block_size_q=BLOCK_SIZE_Q, block_size_kv=BLOCK_SIZE_KV,
        device=DEVICE, emit_atten_mask=True,
    )

    # Generate input tensors
    q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
        b, h, s_q, s_kv, d, device=DEVICE, dtype=DTYPE
    )

    # Run our implementation (returns tuple: attention_out, softmax_lse)
    out_our, _ = torch_bsa.blitz_sparse_attention(q, k, v,
        sabi=sabi,
        actual_seq_lengths=actseqlen, actual_seq_lengths_kv=actseqlenkv,
        num_heads=h, num_key_value_heads=h, input_layout=INPUT_LAYOUT,
        scale_value=scale, atten_mask=bsa_atten_mask, sparse_mode=sm,
        pre_tokens=pre_tok, next_tokens=post_tok,
        block_shape=list(block_shape),
    )
    
    # Run reference implementation
    out_ref = ref_blitz_sparse_attention_launcher(torch_reference=torch_ref, 
                                                  pfa_inputs=(q, k, v, h, scale, pfa_atten_mask, INPUT_LAYOUT), 
                                                  force_dense_sm=(sparsity == 0))   
    
    # Check correctness with reasonable tolerances (moved to CPU for comparison)
    a, b_ref = out_our.cpu(), out_ref.cpu()
    diff = (a - b_ref).abs()
    max_diff = diff.max().item()
    # Opt-in diff logging: BSA_LOG_DIFFS=1 prints `[DIFF]` lines for every
    # comparison site (useful when calibrating atol). Kept opt-in because the
    # extra CPU-side print can perturb cube scheduling and expose intermittent
    # bf16 allocator-slack noise.
    import os as _os
    if _os.environ.get("BSA_LOG_DIFFS"):
        print(
            f"[DIFF] attn_vs_pfa {DTYPE} b={b} h={h} s={s_q} d={d} sp={sparsity} "
            f"bs={block_shape} frame={frame_kind}: max_diff={max_diff:.6g}",
            flush=True,
        )
    if not torch.allclose(a, b_ref, rtol=0.01, atol=0.001):
        pytest.fail(
            f"Outputs don't match for b={b}, h={h}, s_q={s_q}, s_kv={s_kv}, "
            f"d={d}, sparsity={sparsity}, bs={block_shape}: max_diff={max_diff:.5f}"
        )


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test_blitz_sparse_attention_correctness(torch_ref="npu_fusion_attention",
                                            shape=(1, 3, 10_000, 128),
                                            frame_kind="small_frame",
                                            sparsity=0.5,
                                            block_shape=(128, 128))
    logging.getLogger(__name__).info("Smoke test passed.")
