#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""MLA CPU golden（uniform attention golden 的 MLA 适配）。

数学（m0070 / R1）：
  S = (Q_nope ‖ Q_rope) · (K_nope ‖ K_rope)^T * softmax_scale     （q/k_cache 最后维 576 全量参与 mm1）
  P = softmax(S, dim=-1)                                            （mask_mode=3 时对行施加右下对齐因果 mask）
  O = P · V_nope                                                     （V = k_cache[..., :512]，nope 512 段）
  LSE = log(sum_k exp(S - max_k)) + max_k，形状 (N, T)（flash_attn TND 族约定；FIA 自身为 {t,n,1}，
        以本 golden 为上板对比基准，见 learnings 的 LSE 复核结论）

布局限定 TND/NZ/TND（T11 迁移，out 恒等于 q）：q = [T, N1, 576]，k_cache = [num_blocks, 1, 36, block_size, 16]。
输出 O = [T, N1, 512]（TND），LSE = [N1, T]。
"""

import math
import torch

MLA_D_TOTAL = 576
MLA_HEAD_DIM_V = 512
MLA_NZ_BLK_ELEM = 16


def tsoftmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    res = y.div(x_sum)
    return res, x_max, x_sum


def _right_down_causal_mask(s1, s2):
    """右下对齐因果 mask：行 r 允许 列 c <= r + (s2 - s1)（flash_attn 同款 recipe）。"""
    return torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1).bool()


def _attend(q, k, v, mask, scale):
    q = q.float()
    k = k.float()
    v = v.float()
    # mm1: 576 宽（nope512 + rope64 合并）
    qk = torch.matmul(q, k.transpose(-1, -2)).mul(scale)
    if mask is not None:
        qk = qk.masked_fill(mask, value=torch.tensor(-1.7e38))
    softmax_res, x_max, x_sum = tsoftmax(qk)
    out = torch.matmul(softmax_res, v)
    return out, x_max, x_sum


def mla_forward(
    q,
    k_cache,
    *,
    cu_seqlens_q,
    cache_seqlens,
    block_table,
    block_size=128,
    seqused_q=None,
    mask_mode=0,
    scale=None,
    N1=None,
):
    """全量 MLA 参考计算（fp32 累加）。

    Args:
        q:          [T, N1, 576] TND
        k_cache:    [num_blocks, 1, D_nz_sub, block_size, 16] PA_NZ（D_nz_sub=36）
        cu_seqlens_q:   [B+1] int，含前导 0
        cache_seqlens:  [B] int，每 batch KV cache 长度
        block_table:    [B, max_blocks] int
        seqused_q:      [B] 可选，实际 Q 长度（默认用 cu 差分）
        mask_mode:      0=NO_MASK, 3=CAUSAL/RIGHT_DOWN
        scale:          softmax 缩放（默认 1/sqrt(512)）
        N1:             query head 数（默认 q.shape[1]）

    Returns:
        (out, lse): out = [T, N1, 512] fp16/bf16（与 q 同 dtype），lse = [N1, T] float32
    """
    if scale is None:
        scale = 1.0 / math.sqrt(MLA_HEAD_DIM_V)
    dtype = q.dtype
    n1 = N1 or q.shape[1]
    b = len(cu_seqlens_q) - 1
    t_total = cu_seqlens_q[-1]

    # k_cache PA_NZ → [b, n2=1, s2, 576]（重建 576 宽，nope‖rope 合并）
    k_full = _pa_nz_to_batch(k_cache, block_table, cache_seqlens, block_size, b)

    out = torch.zeros(t_total, n1, MLA_HEAD_DIM_V, dtype=dtype)
    lse = torch.zeros(n1, t_total, dtype=torch.float32)
    for i in range(b):
        q_len = (
            seqused_q[i]
            if seqused_q is not None
            else int(cu_seqlens_q[i + 1]) - int(cu_seqlens_q[i])
        )
        kv_len = int(cache_seqlens[i])
        if q_len == 0 or kv_len == 0:
            continue
        q_start = int(cu_seqlens_q[i])
        qi = q[q_start : q_start + q_len, :, :]  # [q_len, N1, 576]
        ki = k_full[i, 0, :kv_len, :]  # [kv_len, 576]

        mask = _right_down_causal_mask(q_len, kv_len) if mask_mode == 3 else None
        # [q_len, N1, 576] → [N1, q_len, 576]
        outi, x_maxi, x_sumi = _attend(
            qi.permute(1, 0, 2),
            ki.unsqueeze(0),
            ki[..., :MLA_HEAD_DIM_V].unsqueeze(0),
            mask,
            scale,
        )
        # [N1, q_len, D] → [q_len, N1, D]（out 恒 TND 布局）
        out[q_start : q_start + q_len, :, :] = outi.permute(1, 0, 2).to(dtype)
        lse[:, q_start : q_start + q_len] = (torch.log(x_sumi) + x_maxi).squeeze(-1)
    return out, lse


def _pa_nz_to_batch(k_cache, block_table, cache_seqlens, block_size, b):
    """PA_NZ [num_blocks, 1, 36, bs, 16] → [b, 1, s2, 576]（仅取每 batch 的实际 kv 长度）。"""
    n_kv, d_sub, bs, elem = (
        k_cache.shape[1],
        k_cache.shape[2],
        k_cache.shape[3],
        k_cache.shape[4],
    )
    d_total = d_sub * elem
    total = [0] * b
    for i in range(b):
        total[i] = int(cache_seqlens[i])
    result = torch.zeros(b, n_kv, max(total), d_total, dtype=k_cache.dtype)
    bt = block_table.cpu().numpy() if torch.is_tensor(block_table) else block_table
    kc = k_cache.cpu().float()
    for i in range(b):
        slen = total[i]
        nblk = (slen + bs - 1) // bs
        for j in range(nblk):
            blk_id = int(bt[i][j])
            if blk_id < 0 or blk_id >= k_cache.shape[0]:
                continue
            start = j * bs
            end = min(start + bs, slen)
            cnt = end - start
            if cnt <= 0:
                continue
            # [1, d_sub, cnt, 16] → [1, cnt, d_sub*16]
            block_data = (
                kc[blk_id, :, :, :cnt, :].permute(0, 2, 1, 3).reshape(1, cnt, d_total)
            )
            result[i, :, start:end, :] = block_data
    return result
