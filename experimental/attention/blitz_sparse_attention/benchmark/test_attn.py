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

# Global test configurations
SEED = 42
DEVICE = "npu:0"
DTYPE = torch.bfloat16
INPUT_LAYOUT = 'BNSD'
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 128
torch.npu.set_device(DEVICE)

# Sweeped test parameters
TORCH_REF_VALS = [False]  # True=our custom reference model; False = torch_npu official kernel
B_VALS = [1]
H_VALS = [1, 2, 3, 4]
S_VALS = [
    24_000,
    # TODO: uncomment once sub-block gather is implemented for partial 512-chunks
    # 36_000,  # ceil(36000/128)=282, not a multiple of 4
]  # s_q = s_kv
D_VALS = [128]   # head dimension
FRAME_VALUES = [FrameWidths(0,0,0,0), FrameWidths(2,1,3,1)]
SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHAPES = list(itertools.product(B_VALS, H_VALS, S_VALS, D_VALS))
FRAME_SPARSITY_COMBOS = list(itertools.product(FRAME_VALUES, SPARSITY_VALS))
FRAME_SPARSITY_IDS = [f"frame({f.left_cols},{f.right_cols},{f.top_rows},{f.bottom_rows})-sparsity{s:.2f}"
                      for f, s in FRAME_SPARSITY_COMBOS]

@pytest.mark.parametrize("torch_ref", TORCH_REF_VALS)
@pytest.mark.parametrize("frame,sparsity", FRAME_SPARSITY_COMBOS, ids=FRAME_SPARSITY_IDS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_blitz_sparse_attention_correctness(torch_ref, shape, frame, sparsity):
    """Test correctness of torch_bsa.blitz_sparse_attention vs reference implementation"""
    # Set random seed for reproducible test inputs
    torch.manual_seed(SEED)

    # Unpack input shapes
    b, h, s_kv, d = shape
    s_q = s_kv
    

    # Generate attention mask and parameters
    pfa_atten_mask, bsa_atten_mask, sabi, sm, scale, pre_tok, post_tok = create_attention_mask(
        b, h, s_q, s_kv, d, sparsity, "blocks_optimized_batched", frame=frame, 
        block_size_q=BLOCK_SIZE_Q, block_size_kv=BLOCK_SIZE_KV, 
        device=DEVICE, emit_atten_mask=True
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
    )
    
    # Run reference implementation
    out_ref = ref_blitz_sparse_attention_launcher(torch_reference=torch_ref, 
                                                  pfa_inputs=(q, k, v, h, scale, pfa_atten_mask, INPUT_LAYOUT), 
                                                  force_dense_sm=(sparsity == 0))   
    
    # Check correctness with reasonable tolerances (moved to CPU for comparison)
    if not torch.allclose(out_our.cpu(), out_ref.cpu(), rtol=0.01, atol=0.01):
        pytest.fail(
            f"Outputs don't match for b={b}, h={h}, s_q={s_q}, s_kv={s_kv}, d={d}, sparsity={sparsity}"
        )


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test_blitz_sparse_attention_correctness(torch_ref=False,
                                            shape=(1, 3, 10_000, 128), 
                                            frame=FrameWidths(left_cols=2, right_cols=1, top_rows=2, bottom_rows=1), 
                                            sparsity=0.5)
    logging.getLogger(__name__).info("Smoke test passed.")
