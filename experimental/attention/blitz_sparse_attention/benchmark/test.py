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
pytest test.py 

2. Quick smoke test on a single input scenario:
python test.py
"""

import itertools
from typing import Tuple
import pytest
import torch
import torch_npu
import torch_bsa
from benchmark import gen_pfa_inputs, create_attention_mask  # data gen
from benchmark import ref_blitz_sparse_attention_launcher  # baseline kernel

# Global test configurations
SEED = 42
DEVICE = "npu:0"
DTYPE = torch.bfloat16
INPUT_LAYOUT = 'BNSD'
torch.npu.set_device(DEVICE)

# Sweeped test parameters
TORCH_REF_VALS = [True, False]  # True=our custom reference model; False = torch_npu official kernel
A_VALS = ["blocks_optimized_batched", "blocks_optimized", "sparse_block_all_same", "lower_triangular", "band", "custom"]
B_VALS = [1]
H_VALS = [1, 2, 3, 4]
S_VALS = [10_000, 20_000, 30_000]  # s_q = s_kv
D_VALS = [128]   # head dimension
SPARSITY_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHAPES = list(itertools.product(B_VALS, H_VALS, S_VALS, D_VALS))


@pytest.mark.parametrize("torch_ref", TORCH_REF_VALS)
@pytest.mark.parametrize("a", A_VALS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
@pytest.mark.parametrize("sparsity", SPARSITY_VALS)
def test_blitz_sparse_attention_correctness(torch_ref, a, shape, sparsity):
    """Test correctness of torch_bsa.blitz_sparse_attention vs reference implementation"""
    b, h, s_kv, d = shape

    # Set random seed for reproducible test inputs
    torch.manual_seed(SEED)

    # Skip test if sparsity is not compatible with current configuration
    if a in ["blocks_optimized", "blocks_optimized_batched"] and sparsity > 0.9:
        pytest.skip("Skipping high sparsity for block optimized modes")

    s_q = s_kv
    
    # Generate attention mask and parameters
    pfa_atten_mask, bsa_atten_mask, sabi, sm, scale, pre_tok, post_tok = create_attention_mask(
        b, h, s_q, s_kv, d, sparsity, a, device=DEVICE, emit_atten_mask=True
    )
    
    # Generate input tensors
    q, k, v, actseqlen, actseqlenkv = gen_pfa_inputs(
        b, h, s_q, s_kv, d, device=DEVICE, dtype=DTYPE
    )
    
    # Run our implementation
    out_our = torch_bsa.blitz_sparse_attention(q, k, v,
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
    
    # Compare results (moved to CPU for comparison)
    out_our_cpu = out_our.cpu()
    out_ref_cpu = out_ref.cpu()
    
    # Check correctness with reasonable tolerances
    if not torch.allclose(out_our_cpu, out_ref_cpu, rtol=0.05, atol=0.05):
        pytest.fail(
            f"Outputs don't match for b={b}, h={h}, s_q={s_q}, s_kv={s_kv}, d={d}, sparsity={sparsity}"
        )


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test_blitz_sparse_attention_correctness(torch_ref=False, a="blocks_optimized_batched", 
                                            shape=(1, 3, 10_000, 128), sparsity=0.5)
    logging.getLogger(__name__).info("Smoke test passed.")
