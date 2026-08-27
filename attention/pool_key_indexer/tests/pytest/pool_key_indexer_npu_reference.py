#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer NPU 组合标杆(性能对比基准)实现。

与 CPU 标杆(pool_key_indexer_reference.py)计算语义一致, 但全程使用 torch 标准
算子在 NPU 上向量化完成(无逐行/逐 batch Python 循环), 用作自定义算子
aclnnPoolKeyIndexer 的性能对比标杆(组合算子基线)。

计算过程(与 CPU 标杆一致):
  1. 池级分数: dot = matmul(q, k^T); score = (w @ relu(dot)) * 1/sqrt(headDim)
  2. 因果可见性(mask_mode=3): 不可见池分数置 -inf
  3. 池级 TopK: k = topk // pool_size, 分数降序
  4. 索引展开: 池 p -> p*pool_size ... p*pool_size+pool_size-1, 无效位 -1
  5. 尾块追加(输出 topk 位置起, 容量 pool_size-1)

布局支持(等长 batch 约束, 覆盖性能对比场景):
  - layout_q: BSND (B,S1,N1,D); TND (T1,N1,D) + actual_seq_q 前缀和(各 batch 等长)
  - layout_k: BSND (B,S2,N2,D); TND (T2,N2,D) + actual_seq_k 前缀和(等长);
              PA_BBND (blockNum,blockSize,N2,D) + block_table + actual_seq_k
              (每 batch 池数, 非前缀和; 通过块表 gather 归一化)

精度: 分数通道全程 fp32(q/k/weights 上转 fp32 计算), 与算子分数管线精度对齐
(算子 cube 侧 fp16 输入 fp32 累加 mm + vector 侧 fp32 加权求和, 与 CPU 标杆
values 误差实测 1e-6 级; fp16 输入的 fp32 matmul 与 fp16-in/fp32-accum mm
数学等价——fp16 乘积可被 fp32 精确表示)。若用 fp16 分数会引入 ~1e-3 舍入,
近并列分数翻转 TopK 选择, 与算子输出不可比。
TopK 并列分数取舍与算子/CPU 标杆可能不同, 由集合比对策略吸收(同既有套件)。

自验(需 NPU): python pool_key_indexer_npu_reference.py
  用小规模配置对 CPU 标杆做 indices 集合比对(逐布局 x ps x mask)。
"""

import torch

__all__ = ["pool_key_indexer_npu_reference"]

_NEG_INF = float("-inf")


def _prefix_to_counts(prefix):
    """前缀和 -> 每 batch 数量(仅在等长断言失败时用于错误信息)。"""
    p = [int(v) for v in prefix.detach().cpu().reshape(-1).tolist()]
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]


def _assert_equal_batch(seq, what, prefix=False):
    """断言各 batch 等长(本标杆的布局归一化约束), 返回公共长度。

    prefix=True 时 seq 为前缀和形式, 先转为每 batch 数量再断言。"""
    vals = _prefix_to_counts(seq) if prefix else [
        int(v) for v in seq.detach().cpu().reshape(-1).tolist()
    ]
    if len(set(vals)) != 1:
        raise ValueError(f"npu reference 仅支持各 batch 等长场景({what}): {vals}")
    return vals[0]


def _topk_desc(scores, k):
    """降序 TopK; NPU topk 不可用时回退到 sort。"""
    try:
        return torch.topk(scores, k, dim=-1, sorted=True)
    except Exception:
        sorted_vals, sorted_idx = torch.sort(scores, dim=-1, descending=True)
        return sorted_vals[..., :k], sorted_idx[..., :k]


def _gather_pa_key(pool_key, block_table, pools):
    """PA_BBND key (blockNum, blockSize, N2, D) -> 逻辑 (B, pools, D)。

    逻辑池 p 属于物理块 block_table[b, p // blockSize] 的第 p % blockSize 行
    (LIV2 语义: 块内每行 = 1 个池)。
    """
    block_num, block_size, n2, d = pool_key.shape
    flat = pool_key[:, :, 0, :].reshape(block_num * block_size, d)
    p_idx = torch.arange(pools, device=pool_key.device)
    blk = block_table[:, p_idx // block_size]
    if int(blk.min()) < 0:
        raise ValueError("block_table 有效槽位不足(存在 -1 被逻辑池引用)")
    rows = blk * block_size + (p_idx % block_size)
    return flat[rows.reshape(-1)].reshape(block_table.shape[0], pools, d)


def pool_key_indexer_npu_reference(
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
    topk=2048,
    pool_size=16,
    mask_mode=3,
    return_value=True,
):
    """PoolKeyIndexer NPU 组合标杆(与 CPU 标杆/CPU 算子语义一致)。

    入参与返回值约定同 pool_key_indexer_reference.pool_key_indexer_reference,
    区别:
      - 输入为 NPU 设备 tensor(输入/输出不跨设备);
      - 仅支持各 batch 等长场景(性能对比场景约束);
      - values 输出 float32(与算子对齐)。
    """
    if layout_q not in ("BSND", "TND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_k not in ("BSND", "TND", "PA_BBND"):
        raise ValueError(f"unsupported layout_k: {layout_k}")
    if mask_mode not in (0, 3):
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if topk % pool_size != 0:
        raise ValueError(f"topk({topk}) must be divisible by pool_size({pool_size})")
    if pool_key.shape[-2] != 1:
        raise ValueError(f"pool_key N2 must be 1, got {pool_key.shape[-2]}")
    device = query.device

    if layout_q == "BSND":
        if actual_seq_q is not None:
            raise ValueError("npu reference 不支持 BSND query 携带 actual_seq_q")
        q, w = query, weights
    else:
        if actual_seq_q is None:
            raise ValueError("layout_q='TND' requires actual_seq_q")
        s1 = _assert_equal_batch(actual_seq_q, "actual_seq_q", prefix=True)
        t1, n1, d = query.shape
        b = actual_seq_q.shape[0]
        if t1 != s1 * b:
            raise ValueError(f"actual_seq_q 总和({s1 * b}) != T1({t1})")
        q = query.reshape(b, s1, n1, d)
        w = weights.reshape(b, s1, n1)
    b, s1, n1, d = q.shape

    if layout_k == "BSND":
        if actual_seq_k is not None:
            raise ValueError("npu reference 不支持 BSND key 携带 actual_seq_k")
        k = pool_key[:, :, 0, :]
    elif layout_k == "TND":
        if actual_seq_k is None:
            raise ValueError("layout_k='TND' requires actual_seq_k")
        pools = _assert_equal_batch(actual_seq_k, "actual_seq_k", prefix=True)
        t2, n2, dk = pool_key.shape
        if t2 != pools * b:
            raise ValueError(f"actual_seq_k 总和({pools * b}) != T2({t2})")
        k = pool_key.reshape(b, pools, n2, dk)[:, :, 0, :]
    else:
        if actual_seq_k is None or block_table is None:
            raise ValueError("layout_k='PA_BBND' requires actual_seq_k and block_table")
        pools = _assert_equal_batch(actual_seq_k, "actual_seq_k")
        k = _gather_pa_key(pool_key, block_table, pools)
    p = k.shape[1]

    sparse = topk // pool_size
    out_len = topk + pool_size - 1
    tail = pool_tail_k.detach().to(device=device, dtype=torch.int64).reshape(b)
    lens = p * pool_size + tail

    dot = torch.relu(torch.matmul(q.float(), k.transpose(-1, -2).unsqueeze(1).float()))
    scores = torch.matmul(w.unsqueeze(2).float(), dot).squeeze(2) * (d ** -0.5)

    row = torch.arange(s1, device=device)
    if mask_mode == 3:
        valid = torch.div(lens[:, None] - s1 + row + 1, pool_size, rounding_mode="floor")
        valid = torch.clamp(valid, 0, p)
        col = torch.arange(p, device=device)
        scores = scores.masked_fill(col[None, None, :] >= valid[:, :, None], _NEG_INF)

    k_sel = min(sparse, p)
    vals, pool_idx = _topk_desc(scores, k_sel)

    if pool_size > 1:
        off = torch.arange(pool_size, device=device)
        expanded = (pool_idx.unsqueeze(-1) * pool_size + off).reshape(b, s1, k_sel * pool_size)
        fin = (
            torch.isfinite(vals)
            .unsqueeze(-1)
            .expand(b, s1, k_sel, pool_size)
            .reshape(b, s1, k_sel * pool_size)
        )
    else:
        expanded = pool_idx
        fin = torch.isfinite(vals)
    token_idx = torch.where(fin, expanded, -1)
    if token_idx.shape[-1] < out_len:
        token_idx = torch.nn.functional.pad(
            token_idx, (0, out_len - token_idx.shape[-1]), value=-1
        )

    if pool_size > 1 and int(tail.max()) > 0:
        t = torch.arange(pool_size - 1, device=device)
        if mask_mode == 3:
            gpos = lens[:, None] - s1 + row[None, :]
            vis = torch.clamp(torch.minimum(tail[:, None], gpos - topk + 1), min=0)
        else:
            vis = tail[:, None].expand(b, s1)
        base = (lens - tail)[:, None, None] + t[None, None, :]
        tail_blk = torch.where(t[None, None, :] < vis[:, :, None], base, -1)
        token_idx[:, :, topk:] = tail_blk

    if layout_q == "TND":
        token_idx = token_idx.reshape(b * s1, out_len)
        if return_value:
            values = vals.reshape(b * s1, k_sel)
        else:
            values = torch.empty(0, dtype=torch.float32, device=device)
    else:
        values = vals if return_value else torch.empty(0, dtype=torch.float32, device=device)
    if return_value:
        values = values.to(torch.float32)
        if k_sel < sparse:
            values = torch.nn.functional.pad(values, (0, sparse - k_sel), value=_NEG_INF)
    return token_idx.to(torch.int32), values


# ---------------------------------------------------------------------------
# 独立运行入口: 小规模自验(与 CPU 标杆比对)
# ---------------------------------------------------------------------------


def _selftest():
    import torch_npu  # noqa: F401
    from pool_key_indexer_reference import pool_key_indexer_reference
    from result_compare import compare_indices, compare_values

    torch_npu.npu.set_device(0)
    dev = "npu:0"
    layouts = (("BSND", "BSND"), ("TND", "TND"), ("BSND", "PA_BBND"), ("TND", "PA_BBND"))
    n_pass = 0
    for ps in (1, 16):
        topk = 128 if ps == 16 else 64
        tail = [ps - 1, min(ps - 1, 7)] if ps > 1 else [0, 0]
        for lq, lk in layouts:
            for mask in (0, 3):
                torch.manual_seed(7)
                b, s1, s2t, n1, d = 2, 64, 128, 8, 128
                pools = s2t // ps
                q = torch.randn(b, s1, n1, d, dtype=torch.float16)
                k = torch.randn(b, pools, 1, d, dtype=torch.float16)
                w = torch.randn(b, s1, n1, dtype=torch.float16)
                ptk = torch.tensor(tail, dtype=torch.int64)
                asq = ask = bt = None
                if lq == "TND":
                    asq = torch.full((b,), s1, dtype=torch.int64).cumsum(0)
                    q = q.reshape(b * s1, n1, d)
                    w = w.reshape(b * s1, n1)
                if lk == "TND":
                    ask = torch.full((b,), pools, dtype=torch.int64).cumsum(0)
                    k = k.reshape(b * pools, 1, d)
                elif lk == "PA_BBND":
                    ask = torch.full((b,), pools, dtype=torch.int64)
                    bs = 16
                    nb = (pools + bs - 1) // bs
                    flat = torch.zeros(b, nb * bs, 1, d, dtype=torch.float16)
                    flat[:, :pools] = k
                    k = flat.reshape(b * nb, bs, 1, d)
                    bt = torch.arange(b * nb, dtype=torch.int32).reshape(b, nb)

                g_idx, g_val = pool_key_indexer_reference(
                    q, k, w, ptk, actual_seq_q=asq, actual_seq_k=ask, block_table=bt,
                    layout_q=lq, layout_k=lk, topk=topk, pool_size=ps,
                    mask_mode=mask, return_value=True,
                )
                n_idx, n_val = pool_key_indexer_npu_reference(
                    q.to(dev), k.to(dev), w.to(dev), ptk.to(dev),
                    actual_seq_q=asq.to(dev) if asq is not None else None,
                    actual_seq_k=ask.to(dev) if ask is not None else None,
                    block_table=bt.to(dev) if bt is not None else None,
                    layout_q=lq, layout_k=lk, topk=topk, pool_size=ps,
                    mask_mode=mask, return_value=True,
                )
                tag = f"ps={ps:<2} {lq}/{lk:<12} mask={mask}"
                ok = compare_indices(g_idx, n_idx)
                compare_values(g_val, n_val, torch.float16, label=f"selftest-{tag}")
                print(f"[selftest] {tag}: {'PASS' if ok else 'FAIL'}")
                assert ok, tag
                n_pass += 1
    print(f"pool_key_indexer_npu_reference selftest passed ({n_pass} cases).")


if __name__ == "__main__":
    _selftest()
