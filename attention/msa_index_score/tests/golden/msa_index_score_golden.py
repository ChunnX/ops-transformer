#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MsaIndexScore CPU 参考实现。

必须镜像 kernel 的数值路径：低精度输入 -> fp32 累加 -> 逐 block 取 max -> 乘 per-token/per-head scale。
语义与上游 Triton 参考实现（vllm-ascend minimax_m3 的 _prefill_index_score_kernel /
_decode_index_score_kernel）一致：点乘 -> mask -> max，mask 必须先于 max 施加。
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
        key_cache:   [num_pages, block_size, num_kv_heads, head_dim]，与 query 同 dtype
        block_table: [batch, max_blocks_per_batch] int32，逻辑 block -> 物理 page
        cu_seq_q:    [batch + 1] int32，query 长度前缀和
        kv_len:      [batch] int32，每个请求可见的 kv 长度
        start_loc:   [batch] int32，每个请求首个 query token 在自身 kv 序列中的位置
        block_size:  一个 sparse block / page 的 token 数
        scale:       [total_q, num_q_heads] fp32 正值，None 表示不施加
        fill_invalid_tail: 是否把不可见 block 写成 -inf

    Returns:
        [num_q_heads, total_q, round_up(max_blocks_per_batch, 16)] fp32
    """
    total_q, num_q_heads, head_dim = query.shape
    batch = len(kv_len)
    max_blocks = block_table.shape[1]
    score_stride = round_up(max_blocks, SCORE_STRIDE_ALIGN)

    # 累加恒为 fp32；key_cache 的 kv head 维在 MSA 中恒为 1（index K 全头共享）。
    q = query.astype(np.float32)
    k = key_cache.astype(np.float32)[:, :, 0, :]  # [num_pages, block_size, head_dim]

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
                # 点乘 -> mask（只取可见列）-> max，顺序不可交换
                s = q[t] @ k[page, :valid, :].T   # [num_q_heads, valid]
                out[:, t, blk] = s.max(axis=1)
            if scale is not None:
                out[:, t, :num_blocks] *= scale[t].astype(np.float32)[:, None]

    if not fill_invalid_tail:
        return out
    return out


def compare(actual, golden, atol=1e-3, rtol=1e-3, error_ratio_threshold=1e-3):
    """双标准比对：填充位必须同为 -inf，其余按 atol/rtol/error_ratio 判定。"""
    actual = np.asarray(actual, dtype=np.float32)
    golden = np.asarray(golden, dtype=np.float32)
    if actual.shape != golden.shape:
        return False, f"shape mismatch: {actual.shape} vs {golden.shape}"

    fill_mask = golden <= NEG_INF * np.float32(0.5)
    fill_bad = int(np.count_nonzero(~(actual[fill_mask] <= NEG_INF * np.float32(0.5))))

    valid = ~fill_mask
    total = int(np.count_nonzero(valid))
    if total == 0:
        return fill_bad == 0, f"no valid element, fill_mismatch={fill_bad}"

    diff = np.abs(actual[valid] - golden[valid])
    bad = int(np.count_nonzero(diff > atol + rtol * np.abs(golden[valid])))
    ratio = bad / total
    msg = (f"valid={total} mismatch={bad} fill_mismatch={fill_bad} "
           f"error_ratio={ratio:.6f} max_abs_diff={float(diff.max()):.6g}")
    return (bad == 0 and fill_bad == 0 and ratio <= error_ratio_threshold), msg
