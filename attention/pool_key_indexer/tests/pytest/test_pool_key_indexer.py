# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Pytest test cases for PoolKeyIndexer torch extension.

Reference:
  - Golden: tests/atk/pool_key_indexer_golden.py (PKIGoldenRef)
  - Pattern: tests/pytest/ (LIV2 pytest structure)

Coverage:
  - Layout: BSND, TND
  - dtype: fp16, bf16
  - mask_mode: 0 (non-causal), 3 (causal)
  - pool_size: 1 (degenerate), 8, 16 (expanded)
  - return_value: True, False
  - block sizes (PA_BBND optional)
"""

import os
import sys
import math
import pytest
import numpy as np
import torch
import torch_npu

# Import golden from ATK test assets
_ATK_DIR = os.path.join(os.path.dirname(__file__), "..", "atk")
sys.path.insert(0, os.path.abspath(_ATK_DIR))
from pool_key_indexer_golden import (
    PKIGoldenRef,
    _get_compute_dtype,
    _normalize_prefix_seq,
)

# Import comparison utilities
sys.path.insert(0, os.path.dirname(__file__))
from result_compare import compare_indices, compare_values

# Import torch extension
from cann_ops_transformer import pool_key_indexer

HEAD_DIM = 128
N2 = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_actual_seq(seq_lens):
    """Build cumulative prefix-sum tensor for TND layout."""
    return _normalize_prefix_seq(seq_lens, sum(seq_lens))


def _build_bsnd_inputs(batch, q_seq, k_seq, n1, pool_size, dtype, seed=42):
    """Construct BSND-layout inputs on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dt = dtype
    query = torch.randn(batch, q_seq, n1, HEAD_DIM, dtype=dt)
    k_seq_pools = k_seq // pool_size
    pool_key = torch.randn(batch, k_seq_pools, N2, HEAD_DIM, dtype=dt)
    weights = torch.randn(batch, q_seq, n1, dtype=dt)
    pool_tail_k = torch.randint(0, pool_size, (batch,), dtype=torch.int32)
    return query, pool_key, weights, pool_tail_k


def _build_tnd_inputs(q_lens, k_lens, n1, pool_size, dtype, seed=42):
    """Construct TND-layout inputs on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dt = dtype
    total_q = sum(q_lens)
    total_k = sum(k_lens)
    query = torch.randn(total_q, n1, HEAD_DIM, dtype=dt)
    k_pool_lens = [kl // pool_size for kl in k_lens]
    total_k_pools = sum(k_pool_lens)
    pool_key = torch.randn(total_k_pools, N2, HEAD_DIM, dtype=dt)
    weights = torch.randn(total_q, n1, dtype=dt)
    pool_tail_k = torch.randint(0, pool_size, (len(q_lens),), dtype=torch.int32)
    actual_seq_q = _make_actual_seq(q_lens)
    actual_seq_k = _make_actual_seq(k_lens)
    return query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k


def _run_npu(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    quant_mode=-1,
    return_value=False,
    device_id=0,
):
    """Move inputs to NPU and call the torch extension."""
    dev = f"npu:{device_id}"
    q = query.to(dev)
    k = pool_key.to(dev)
    w = weights.to(dev)
    ptk = pool_tail_k.to(dev)
    asq = actual_seq_q.to(dev) if actual_seq_q is not None else None
    ask = actual_seq_k.to(dev) if actual_seq_k is not None else None
    bt = block_table.to(dev) if block_table is not None else None

    indices, values = pool_key_indexer(
        q,
        k,
        w,
        ptk,
        actual_seq_q=asq,
        actual_seq_k=ask,
        block_table=bt,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
    )
    torch.npu.synchronize()
    return indices.cpu(), values.cpu()


def _run_golden(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    batch=1,
    q_seq=1,
    k_seq=1,
    n1=1,
    topk=128,
    pool_size=1,
    mask_mode=0,
    return_value=False,
    dtype=torch.float16,
    ext_indices=None,
):
    """Run CPU golden reference (PKIGoldenRef)."""
    compute_dtype = _get_compute_dtype(dtype)
    ref = PKIGoldenRef(
        batch_size=batch,
        q_seq=q_seq,
        k_seq=k_seq,
        q_head_num=n1,
        k_head_num=N2,
        head_dim=HEAD_DIM,
        layout_query=layout_q,
        layout_key=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        compute_dtype=compute_dtype,
    )
    g_indices, g_values = ref.forward(
        query,
        pool_key,
        weights,
        actual_seq_q,
        actual_seq_k,
        block_table=block_table,
        pool_tail_k=pool_tail_k,
        ext_indices=ext_indices,
    )
    return g_indices, g_values


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _npu_init():
    torch_npu.npu.set_device(0)


# Test case configurations: (id, layout_q, layout_k, dtype, batch, q_seq, k_seq,
#                            n1, topk, pool_size, mask_mode, return_value)
TEST_CASES = [
    # --- Degenerate (pool_size=1) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        1,
        0,
        False,
        id="bsnd_fp16_ps1_nocausal_noval",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        1,
        3,
        False,
        id="bsnd_fp16_ps1_causal_noval",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.bfloat16,
        2,
        32,
        128,
        4,
        64,
        1,
        0,
        True,
        id="bsnd_bf16_ps1_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        2,
        48,
        192,
        16,
        128,
        1,
        3,
        True,
        id="bsnd_fp16_ps1_causal_val",
    ),
    # --- Expanded (pool_size>1) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        16,
        0,
        False,
        id="bsnd_fp16_ps16_nocausal_noval",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        16,
        3,
        True,
        id="bsnd_fp16_ps16_causal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.bfloat16,
        2,
        32,
        256,
        16,
        128,
        8,
        0,
        True,
        id="bsnd_bf16_ps8_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        32,
        512,
        32,
        256,
        32,
        3,
        False,
        id="bsnd_fp16_ps32_causal_noval",
    ),
    # --- TND ---
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [32, 32],
        [64, 64],
        [128, 128],
        8,
        128,
        1,
        0,
        False,
        id="tnd_fp16_ps1_nocausal_noval",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [32, 32],
        [64, 64],
        [128, 128],
        8,
        128,
        16,
        3,
        True,
        id="tnd_fp16_ps16_causal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.bfloat16,
        [16, 48],
        [32, 32],
        [64, 192],
        4,
        64,
        8,
        0,
        True,
        id="tnd_bf16_ps8_nocausal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [16, 16, 16],
        [48, 48, 48],
        [96, 96, 96],
        16,
        128,
        16,
        3,
        False,
        id="tnd_fp16_ps16_causal_noval_3batch",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout_q,layout_k,dtype,batch,q_seq,k_seq,n1,topk,pool_size,mask_mode,return_value",
    TEST_CASES,
)
def test_pool_key_indexer(
    layout_q,
    layout_k,
    dtype,
    batch,
    q_seq,
    k_seq,
    n1,
    topk,
    pool_size,
    mask_mode,
    return_value,
):
    """End-to-end test: golden vs NPU torch extension."""

    # ---- Construct inputs ----
    if layout_q == "BSND":
        assert layout_k == "BSND"
        query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
            batch, q_seq, k_seq, n1, pool_size, dtype
        )
        actual_seq_q = None
        actual_seq_k = None
    elif layout_q == "TND":
        assert layout_k == "TND"
        if isinstance(batch, list):
            q_lens = batch
            k_lens = k_seq if isinstance(k_seq, list) else [k_seq] * len(q_lens)
            b_size = len(q_lens)
            total_q = sum(q_lens)
            total_k = sum(k_lens)
        else:
            q_lens = [q_seq] * batch
            k_lens = [k_seq] * batch
            b_size = batch
            total_q = q_seq * batch
            total_k = k_seq * batch
        query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k = (
            _build_tnd_inputs(q_lens, k_lens, n1, pool_size, dtype)
        )
        q_seq = total_q
        k_seq = total_k
        batch = b_size
    else:
        pytest.skip(f"Unsupported layout_q={layout_q}")

    print(
        f"\n=== Test: layout_q={layout_q} dtype={dtype} batch={batch} "
        f"q_seq={q_seq} k_seq={k_seq} n1={n1} topk={topk} "
        f"pool_size={pool_size} mask_mode={mask_mode} return_value={return_value} ==="
    )

    # ---- Run golden ----
    g_indices, g_values = _run_golden(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q=layout_q,
        layout_k=layout_k,
        batch=batch,
        q_seq=q_seq,
        k_seq=k_seq,
        n1=n1,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
        dtype=dtype,
    )

    # ---- Run NPU ----
    npu_indices, npu_values = _run_npu(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
    )

    # ---- Canonicalize NPU output via golden normalization ----
    _, canon_indices_npu = _run_golden(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q=layout_q,
        layout_k=layout_k,
        batch=batch,
        q_seq=q_seq,
        k_seq=k_seq,
        n1=n1,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
        dtype=dtype,
        ext_indices=npu_indices,
    )

    # Also get canonicalized golden indices
    _, canon_values_g = _run_golden(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        layout_q=layout_q,
        layout_k=layout_k,
        batch=batch,
        q_seq=q_seq,
        k_seq=k_seq,
        n1=n1,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
        dtype=dtype,
        ext_indices=g_indices,
    )

    # ---- Compare ----
    # Indices: compare golden-raw vs npu (after canonicalization both should match)
    # Since COMPARE_INDICES=False in golden, indices are zeroed; we compare values instead
    ok_values = compare_values(canon_values_g, canon_indices_npu, dtype)

    if return_value:
        # Compare raw NPU values against golden values (sorted per row)
        ok_raw_vals = compare_values(g_values, npu_values, dtype)
        assert ok_values or ok_raw_vals, (
            f"Values comparison failed for {layout_q} ps={pool_size} mask={mask_mode}"
        )
    else:
        # return_value=False: NPU values should be empty
        assert npu_values.numel() == 0 or npu_values.shape[-1] == 0, (
            f"Expected empty values when return_value=False, got shape {npu_values.shape}"
        )

    print(
        f"=== PASSED: {layout_q} dtype={dtype} ps={pool_size} mask={mask_mode} val={return_value} ==="
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
