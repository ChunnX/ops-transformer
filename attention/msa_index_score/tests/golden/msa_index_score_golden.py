#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MsaIndexScore CPU 参考实现。

数值路径（与 kernel 对齐）：
  score = Maxpool[ (scale ·) Q @ Kᵀ + atten_mask ] + local_mask

- sparse_mode=0：无因果，仅 kv_len 截断
- sparse_mode=3：rightDownCausal，visible_end = kv_len - q_len + t_off + 1
- local_mask：由 start_loc（query 所在逻辑 block）+ init/local_blocks 强制高分
"""

import numpy as np

NEG_INF = np.float32(-3.4028234663852886e38)
LOCAL_SCORE_INIT = np.float32(1.0e30)
LOCAL_SCORE_LOCAL = np.float32(1.0e29)
SCORE_STRIDE_ALIGN = 16
SPARSE_MODE_DEFAULT = 0
SPARSE_MODE_RIGHT_DOWN = 3
DEFAULT_INIT_BLOCKS = 0
DEFAULT_LOCAL_BLOCKS = 1


def round_up(value, align):
    return (value + align - 1) // align * align


def _visible_key_end(sparse_mode, kv_len, q_len, t_off):
    if sparse_mode == SPARSE_MODE_RIGHT_DOWN:
        return int(np.clip(kv_len - q_len + t_off + 1, 0, kv_len))
    return int(kv_len)


def msa_index_score_golden(query, key, block_table, cu_seq_q, kv_len, start_loc,
                           sparse_mode=SPARSE_MODE_RIGHT_DOWN, block_size=128, scale=None,
                           init_blocks=DEFAULT_INIT_BLOCKS, local_blocks=DEFAULT_LOCAL_BLOCKS):
    """计算 MSA index block score。

    Args:
        query:       [T1, N1, D]
        key:         [NP, P, N2, D] BBND；非量化与 query 同 dtype；量化为 int8
        block_table: [B, MB]
        cu_seq_q:    [B+1]
        kv_len:      [B]
        start_loc:   [B]，当前 query 所在逻辑 block 索引（local_mask）
        sparse_mode: 0 或 3
        scale:       int8 时 [NP, N2, P]；非量化为 None
    """
    total_q, num_q_heads, head_dim = query.shape
    batch = len(kv_len)
    max_blocks = block_table.shape[1]
    score_stride = round_up(max_blocks, SCORE_STRIDE_ALIGN)
    num_kv_heads = key.shape[2]
    is_quant = np.issubdtype(key.dtype, np.integer)

    if is_quant:
        if scale is None:
            raise ValueError("int8 key requires dequant scale of shape [NP, N_kv, P]")
        if scale.shape != (key.shape[0], num_kv_heads, block_size):
            raise ValueError(
                f"dequant scale shape must be {(key.shape[0], num_kv_heads, block_size)}, got {scale.shape}")
    elif scale is not None:
        raise ValueError("non-quant path must not pass scale")

    q = query.astype(np.float32)
    k = key.astype(np.float32)[:, :, 0, :]
    deq = None
    if is_quant:
        deq = scale.astype(np.float32)[:, 0, :]

    out = np.full((num_q_heads, total_q, score_stride), NEG_INF, dtype=np.float32)

    for b in range(batch):
        q_begin, q_end = int(cu_seq_q[b]), int(cu_seq_q[b + 1])
        q_len = q_end - q_begin
        kv = int(kv_len[b])
        num_blocks = (kv + block_size - 1) // block_size
        q_block = int(start_loc[b])
        local_start = max(0, q_block + 1 - int(local_blocks))

        for t in range(q_begin, q_end):
            t_off = t - q_begin
            visible_key_end = _visible_key_end(sparse_mode, kv, q_len, t_off)
            for blk in range(num_blocks):
                key_lo = blk * block_size
                valid = min(visible_key_end - key_lo, block_size)
                if valid <= 0:
                    continue
                page = int(block_table[b, blk])
                k_page = k[page, :valid, :]
                if deq is not None:
                    k_page = k_page * deq[page, :valid, None]
                s = q[t] @ k_page.T
                out[:, t, blk] = s.max(axis=1)

            # local_mask：强制选中 block 写高分（覆盖 maxpool 结果）
            for blk in range(num_blocks):
                boost = None
                if blk < int(init_blocks):
                    boost = LOCAL_SCORE_INIT
                if local_start <= blk <= q_block:
                    boost = LOCAL_SCORE_LOCAL
                if boost is not None:
                    out[:, t, blk] = boost

    return out


def compare(actual, golden, atol=1e-3, rtol=1e-3, error_ratio_threshold=1e-3):
    """比对 actual 与 golden。填充位必须同为 -inf；强制高分位允许同为 +inf/1e29/1e30。"""
    actual = np.asarray(actual, dtype=np.float32)
    golden = np.asarray(golden, dtype=np.float32)
    if actual.shape != golden.shape:
        return False, f"shape mismatch: actual={actual.shape} golden={golden.shape}"

    fill_a = np.isneginf(actual)
    fill_g = np.isneginf(golden)
    if not np.array_equal(fill_a, fill_g):
        return False, "fill(-inf) mask mismatch"

    # 强制高分：两侧都极大即可
    boost_g = golden >= 1.0e28
    boost_a = actual >= 1.0e28
    if not np.array_equal(boost_a, boost_g):
        return False, "local_mask boost mask mismatch"

    valid = (~fill_g) & (~boost_g)
    if not np.any(valid):
        return True, "ok (all fill/boost)"

    diff = np.abs(actual[valid] - golden[valid])
    denom = np.maximum(np.abs(golden[valid]), 1.0)
    bad = diff > (atol + rtol * denom)
    ratio = float(np.mean(bad)) if bad.size else 0.0
    if ratio > error_ratio_threshold:
        max_diff = float(np.max(diff)) if diff.size else 0.0
        return False, f"error_ratio={ratio:.6f} max_abs_diff={max_diff}"
    return True, "ok"
