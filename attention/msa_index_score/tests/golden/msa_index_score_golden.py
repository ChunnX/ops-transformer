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

数值路径：
  非量化：fp/bf16 key → fp32 点乘 → mask → 逐 block max（score 侧不乘 scale）。
  int8 前融合：cast(int8)*scale[page,h,token] → fp32 点乘 → mask → max。
"""

import numpy as np

NEG_INF = np.float32(-3.4028234663852886e38)
SCORE_STRIDE_ALIGN = 16


def round_up(value, align):
    return (value + align - 1) // align * align


def msa_index_score_golden(query, key_cache, block_table, cu_seq_q, kv_len, start_loc,
                           block_size=128, scale=None, fill_invalid_tail=True):
    """计算 MSA index block score。

    Args:
        query:       [total_q, num_q_heads, head_dim]，fp16 / bf16 / fp32
        key_cache:   [num_pages, block_size, num_kv_heads, head_dim]
                     非量化：与 query 同 dtype；量化：int8
        block_table: [batch, max_blocks_per_batch] int32，逻辑 block -> 物理 page
        cu_seq_q:    [batch + 1] int32，query 长度前缀和
        kv_len:      [batch] int32，每个请求可见的 kv 长度
        start_loc:   [batch] int32，每个请求首个 query token 在自身 kv 序列中的位置
        block_size:  一个 sparse block / page 的 token 数
        scale:       非量化必须为 None；int8 路径为 [num_pages, num_kv_heads, block_size] fp32 反量化系数
        fill_invalid_tail: 是否把不可见 block 写成 -inf

    Returns:
        [num_q_heads, total_q, round_up(max_blocks_per_batch, 16)] fp32
    """
    total_q, num_q_heads, head_dim = query.shape
    batch = len(kv_len)
    max_blocks = block_table.shape[1]
    score_stride = round_up(max_blocks, SCORE_STRIDE_ALIGN)
    num_kv_heads = key_cache.shape[2]
    is_quant = np.issubdtype(key_cache.dtype, np.integer)

    if is_quant:
        if scale is None:
            raise ValueError("int8 key_cache requires dequant scale of shape [NP, N_kv, P]")
        if scale.shape != (key_cache.shape[0], num_kv_heads, block_size):
            raise ValueError(
                f"dequant scale shape must be {(key_cache.shape[0], num_kv_heads, block_size)}, "
                f"got {scale.shape}")
    elif scale is not None:
        raise ValueError("non-quant path must not pass scale (score-side scale is not supported)")

    q = query.astype(np.float32)
    # Hkv==1 时取 head0；若 Hkv>1，按 GQA 取对应 head（P0 仅 Hkv=1）。
    k = key_cache.astype(np.float32)[:, :, 0, :]  # [num_pages, block_size, head_dim]
    deq = None
    if is_quant:
        deq = scale.astype(np.float32)[:, 0, :]  # [num_pages, block_size]

    out = np.full((num_q_heads, total_q, score_stride), NEG_INF, dtype=np.float32)

    for b in range(batch):
        q_begin, q_end = int(cu_seq_q[b]), int(cu_seq_q[b + 1])
        num_blocks = (int(kv_len[b]) + block_size - 1) // block_size
        for t in range(q_begin, q_end):
            q_pos = int(start_loc[b]) + (t - q_begin)
            visible_key_end = min(q_pos + 1, int(kv_len[b]))
            for blk in range(num_blocks):
                key_lo = blk * block_size
                valid = min(visible_key_end - key_lo, block_size)
                if valid <= 0:
                    continue  # 全不可见，保持 -inf
                page = int(block_table[b, blk])
                k_page = k[page, :valid, :]
                if deq is not None:
                    k_page = k_page * deq[page, :valid, None]
                # 点乘 -> mask（只取可见列）-> max，顺序不可交换
                s = q[t] @ k_page.T  # [num_q_heads, valid]
                out[:, t, blk] = s.max(axis=1)

    if not fill_invalid_tail:
        return out
    return out


def compare(actual, golden, atol=1e-3, rtol=1e-3, error_ratio_threshold=1e-3):
    """比对 actual 与 golden。填充位必须同为 -inf；有效位按 atol/rtol。"""
    actual = np.asarray(actual, dtype=np.float32)
    golden = np.asarray(golden, dtype=np.float32)
    if actual.shape != golden.shape:
        return False, f"shape mismatch: actual={actual.shape} golden={golden.shape}"

    fill_a = np.isneginf(actual)
    fill_g = np.isneginf(golden)
    if not np.array_equal(fill_a, fill_g):
        return False, "fill(-inf) mask mismatch"

    valid = ~fill_g
    if not np.any(valid):
        return True, "ok (all fill)"

    diff = np.abs(actual[valid] - golden[valid])
    denom = np.maximum(np.abs(golden[valid]), 1.0)
    bad = diff > (atol + rtol * denom)
    ratio = float(np.mean(bad)) if bad.size else 0.0
    if ratio > error_ratio_threshold:
        max_diff = float(np.max(diff)) if diff.size else 0.0
        return False, f"error_ratio={ratio:.6f} max_abs_diff={max_diff}"
    return True, "ok"
