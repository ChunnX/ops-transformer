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
    _construct_block_table,
)

# Import comparison utilities
sys.path.insert(0, os.path.dirname(__file__))
from result_compare import compare_indices, compare_values

# Force golden to emit real indices (default False zeroes them out,
# which defeats index set comparison)
import pool_key_indexer_golden as _pki_golden_mod

_pki_golden_mod.COMPARE_INDICES = True

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
    pool_tail_k = torch.randint(0, pool_size, (batch,), dtype=torch.int64)
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
    pool_tail_k = torch.randint(0, pool_size, (len(q_lens),), dtype=torch.int64)
    actual_seq_q = _make_actual_seq(q_lens)
    # actual_seq_k 语义为累计 pool 数（与 pool_key 的 T 维一致，host 校验
    # last == key T2；golden 内部 L_orig = S2*pool_size + tail 同样按 pool 数解释）
    actual_seq_k = _make_actual_seq(k_pool_lens)
    return query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k


def _make_actual_seq_noncum(pool_lens):
    """PA_BBND: actual_seq_k 为每 batch pool 数(非前缀和)。"""
    return torch.tensor(pool_lens, dtype=torch.int64)


def _pack_pa_key(pool_key_bsnd, k_pool_lens, pool_size):
    """将逻辑 (T2_pool, N2, D) 打包为 PA 物理块布局 (blockNum, blockSize=16, N2, D)。

    与 LIV2 语义一致: PA key 块内每行 = 1 个池(key 不做 token 展开)。
    每 batch 的池独立连续填块(与 _build_block_table 的块数口径一致),
    物理块随机排布, 尾块不足补零。返回 (key_pa, block_size, block_num)。
    """
    block_size = 16
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_bsnd.shape[1], pool_key_bsnd.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_bsnd.dtype)
    # 逐 batch 填块
    pool_start = 0
    blk_start = 0
    for b, nb in enumerate(blocks_per_batch):
        pools = k_pool_lens[b]
        seg = pool_key_bsnd[pool_start : pool_start + pools]  # (pools, n2, d)
        pad = nb * block_size - pools
        if pad:
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start : blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)
    return blocks[perm].contiguous(), block_size, block_num


def _build_block_table(batch, k_pool_lens, pool_size):
    """构造 (B, maxBlocks) int32 物理块映射, 空槽 -1。

    PA key 块内每行=1池(LIV2 语义): 每块容纳 16 个池, 块数 = ceil(pools/16)。
    返回 (block_table, 物理块总数)。
    """
    block_size = 16
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    max_blocks = max(blocks_per_batch)
    total_blocks = sum(blocks_per_batch)
    perm = torch.randperm(total_blocks)
    bt = torch.full((batch, max_blocks), -1, dtype=torch.int32)
    cur = 0
    for b, nb in enumerate(blocks_per_batch):
        for i in range(nb):
            bt[b, i] = perm[cur]
            cur += 1
    return bt, total_blocks


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
    # --- PA (layout_k=PA_BBND) ---
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        2,
        32,
        256,
        8,
        128,
        8,
        0,
        True,
        id="bsnd_pa_fp16_ps8_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        2,
        32,
        256,
        8,
        128,
        16,
        3,
        True,
        id="bsnd_pa_fp16_ps16_causal_val",
    ),
    pytest.param(
        "TND",
        "PA_BBND",
        torch.float16,
        [16, 16],
        32,
        [128, 128],
        8,
        128,
        8,
        0,
        False,
        id="tnd_pa_fp16_ps8_nocausal_noval",
    ),
    pytest.param(
        "TND",
        "PA_BBND",
        torch.bfloat16,
        [16, 16],
        32,
        [128, 128],
        8,
        128,
        16,
        3,
        True,
        id="tnd_pa_bf16_ps16_causal_val",
    ),
    # --- sparseCount=1 边界(topk == pool_size): 尾区越写/清理缺陷回归 ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        48,
        800,
        8,
        16,
        16,
        3,
        True,
        id="bsnd_fp16_ps16_topk16_causal_sparse1",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        48,
        48,
        8,
        16,
        16,
        3,
        True,
        id="bsnd_fp16_ps16_topk16_causal_L48",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [16, 32],
        48,
        48,
        8,
        16,
        16,
        3,
        True,
        id="tnd_fp16_ps16_topk16_causal_L48",
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
    block_table = None
    if layout_q == "BSND" and layout_k == "PA_BBND":
        # PA + BSND: actual_seq_q 必须为 None; actual_seq_k 为每 batch pool 数(非前缀和)
        query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
            batch, q_seq, k_seq, n1, pool_size, dtype
        )
        actual_seq_q = None
        k_pool_lens = [k_seq // pool_size] * batch
        actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
        # pool_key: (B, pools, N2, D) -> (B*pools, N2, D) 再打包为物理块布局
        pool_key = pool_key.reshape(-1, pool_key.shape[-2], pool_key.shape[-1])
        pool_key, pa_block_size, pa_block_num = _pack_pa_key(
            pool_key, k_pool_lens, pool_size
        )
        block_table, _ = _build_block_table(batch, k_pool_lens, pool_size)
    elif layout_q == "TND" and layout_k == "PA_BBND":
        # PA + TND: actual_seq_q 前缀和; actual_seq_k 为每 batch pool 数(非前缀和)
        q_lens = batch if isinstance(batch, list) else [q_seq] * batch
        k_lens = k_seq if isinstance(k_seq, list) else [k_seq] * len(q_lens)
        query, _, weights, pool_tail_k, actual_seq_q, _ = _build_tnd_inputs(
            q_lens, k_lens, n1, pool_size, dtype
        )
        k_pool_lens = [kl // pool_size for kl in k_lens]
        actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
        pool_key, pa_block_size, pa_block_num = _pack_pa_key(
            torch.randn(sum(k_pool_lens), N2, HEAD_DIM, dtype=dtype),
            k_pool_lens,
            pool_size,
        )
        q_seq = sum(q_lens)
        k_seq = sum(k_lens)
        batch = len(q_lens)
        block_table, _ = _build_block_table(batch, k_pool_lens, pool_size)
    elif layout_q == "BSND":
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
        block_table=block_table,
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
        block_table=block_table,
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
        block_table=block_table,
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
        block_table=block_table,
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
    # 1) Index set comparison (mandatory, all cases): golden-raw vs NPU-raw.
    #    Row-level sorted-set equality with 0.995 pass-rate threshold.
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"Indices set comparison failed for {layout_q} ps={pool_size} mask={mask_mode}"
    )

    # 2) Values comparison (return_value=True): normalized channel must pass
    #    on its own; raw channel is informational only (no OR fallback).
    if return_value:
        ok_values = compare_values(canon_values_g, canon_indices_npu, dtype)
        ok_raw_vals = compare_values(g_values, npu_values, dtype)
        assert ok_values, (
            f"Normalized values comparison failed for {layout_q} ps={pool_size} "
            f"mask={mask_mode} (raw channel: {'PASS' if ok_raw_vals else 'FAIL'})"
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
