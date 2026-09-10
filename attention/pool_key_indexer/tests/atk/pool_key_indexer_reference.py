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
PoolKeyIndexer 统一标杆实现(CPU 金标 + NPU 组合标杆, 按输入设备自动选择)。

统一入口 ``pool_key_indexer_reference(...)``:
  - 输入 tensor 在 CPU  -> CPU 金标实现(逐 batch 精确计算, 全布局能力, 速度慢,
    作为精度基准);
  - 输入 tensor 在 NPU  -> NPU 组合标杆实现(纯 torch 标准算子向量化, 无 Python
    逐行循环, 作为自定义算子的性能对比基线)。

两种实现接口能力一致(支持 BSND/TND query, BSND/TND/PA_BBND key, 不等长 batch、
0 池/0 长度 batch、PA 0 轴非连续 key、BSND 可选 actual_seq 等), 计算语义一致:

  1. 池级分数计算(逐 batch, float32):
        dot[i, n, j] = dot(query[i, n, :], pool_key[j, :]) / sqrt(headDim)  # 缩放点积
        score[i, j]  = sum_n weights[i, n] * relu(dot[i, n, j])             # 多头加权聚合
     分数通道全程 fp32(q/k/weights 上转 fp32 计算), 与算子 cube 侧 fp16 输入
     fp32 累加 + vector 侧 fp32 加权求和的精度管线对齐; 若用 fp16 分数会引入
     ~1e-3 舍入, 近并列分数翻转 TopK 选择, 导致与算子输出不可比。
  2. 因果可见池数(mask_mode=3, rightDownCausal 右对齐因果):
        L          = 池数 * pool_size + pool_tail_k[b]            # 原始 token 总数
        valid[i]   = clamp(floor((L - S1 + i + 1) / pool_size), 0, 池数)
     即 query 第 i 行只能看到前 valid[i] 个池; mask_mode=0 时所有池可见。
  3. 池级 TopK: 取 k = topk // pool_size 个池, 分数降序选取
     (标杆在同分时按下标升序取; NPU 并列分数取舍可能不同, 由集合比对吸收)。
  4. 索引展开: 选中池 p 展开为 token 索引
        p * pool_size, p * pool_size + 1, ..., p * pool_size + pool_size - 1
  5. 尾块追加(输出 topk 位置起, 容量 pool_size - 1):
        mask_mode=0: visible = pool_tail_k
        mask_mode=3: visible = max(0, min(pool_tail_k, gpos - topk + 1))
                      其中 gpos = L - S1 + i (query 行的全局 token 位置)
        第 t 个可见尾 token 的索引为 L - pool_tail_k + t
  6. 输出:
        indices: (B, S1, topk+pool_size-1) 或 (T1, topk+pool_size-1), int32, 无效 -1
        values:  (B, S1, topk//pool_size)   或 (T1, topk//pool_size),  float32, 无效 -inf

NPU 实现说明:
  - 不等长 batch 通过"归一化到等长稠密布局 + 有效性掩码"向量化完成:
    query 按批 gather 填充到 (B, S1max), pool_key 归一化到 (B, Pmax)
    (TND gather / PA 经 block_table gather), 无效行/列以 -inf 分数掩蔽,
    输出按有效性反填充(unpad);
  - TopK 并列分数取舍与 CPU 标杆可能不同, 由集合比对策略吸收(同既有套件)。

可直接运行本文件做自验(需 NPU):  python pool_key_indexer_reference.py
  1. CPU 实现自验(密集/PA 布局输出 shape/取值范围/交叉一致性);
  2. NPU 实现 vs CPU 实现一致性自验(等长 + 不等长 + 0 池 + PA 非连续 +
     BSND actual_seq 等, indices 集合比对 + values 参考比对)。
"""

import math

import torch

__all__ = ["pool_key_indexer_reference"]

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _as_int_list(x):
    """int tensor / list / 标量 -> list[int]"""
    if x is None:
        return None
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _prefix_to_counts(prefix):
    """前缀和序列 -> 每 batch 数量列表"""
    p = _as_int_list(prefix)
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]


def _topk_desc(scores, k):
    """降序 TopK; NPU topk 不可用时回退到 sort。

    NPU 上 torch.topk 插件内部走 NonZero/AICPU 路径, 在亿级 score 上触发
    aicpu timeout(E30008, 看门狗 ~10min), 且超时为进程级内错无法 try/except
    捕获, 会级联饿死同 worker 的后续用例(实测 k=1024/84M 与 k=256/446M 均
    触发) —— NPU 上一律走 sort(AICore 向量路径, 行为稳定); CPU topk 原生
    快速, 保持不变。
    """
    if getattr(scores, "is_npu", False):
        sorted_vals, sorted_idx = torch.sort(scores, dim=-1, descending=True)
        return sorted_vals[..., :k], sorted_idx[..., :k]
    try:
        return torch.topk(scores, k, dim=-1, sorted=True)
    except Exception:
        sorted_vals, sorted_idx = torch.sort(scores, dim=-1, descending=True)
        return sorted_vals[..., :k], sorted_idx[..., :k]


def _validate_args(query, pool_key, layout_q, layout_k, topk, pool_size, mask_mode):
    """公共参数校验(两种实现共用)。"""
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
    if query.shape[-1] != pool_key.shape[-1]:
        raise ValueError(
            f"headDim mismatch: query {query.shape[-1]} vs pool_key {pool_key.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# CPU 金标实现(逐 batch 精确计算)
# ---------------------------------------------------------------------------


def _split_query(query, weights, layout_q, actual_seq_q):
    """拆分 query/weights 为逐 batch 的 (Sb, N1, D) / (Sb, N1) float32 列表。

    返回 (q_list, w_list, counts, batch, s1_dim):
      - counts: 每 batch 有效 query token 数
      - s1_dim: BSND 输出的 S1 维(query.shape[1], 含无效行)
    """
    if layout_q == "BSND":
        batch, s1_dim, n1, d = query.shape
        if actual_seq_q is not None:
            counts = _as_int_list(actual_seq_q)
        else:
            counts = [s1_dim] * batch
        q_list = [query[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        w_list = [weights[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        return q_list, w_list, counts, batch, s1_dim
    if layout_q == "TND":
        t1, n1, d = query.shape
        if actual_seq_q is None:
            raise ValueError("layout_q='TND' requires actual_seq_q (prefix sums)")
        counts = _prefix_to_counts(actual_seq_q)
        q_list, w_list, start = [], [], 0
        for c in counts:
            q_list.append(query[start : start + c].to(torch.float32))
            w_list.append(weights[start : start + c].to(torch.float32))
            start += c
        return q_list, w_list, counts, len(counts), t1
    raise ValueError(f"unsupported layout_q: {layout_q}")


def _split_key(pool_key, layout_k, actual_seq_k, block_table):
    """拆分 pool_key 为逐 batch 的 (Pb, D) float32 列表。

    - BSND:    pool_key (B, S2, N2, D), actual_seq_k 可选(每 batch 池数, 非前缀和);
    - TND:     pool_key (T2, N2, D),    actual_seq_k 必传(池数前缀和);
    - PA_BBND: pool_key (blockNum, blockSize, N2, D), 块内每行一个池,
               需 block_table(B, maxBlocks) 与 actual_seq_k(每 batch 池数, 非前缀和);
               逻辑池 p 属于物理块 block_table[b][p // blockSize] 的第 p % blockSize 行,
               空槽(-1)跳过; 支持 0 轴非连续(strided view)。
    """
    if layout_k == "BSND":
        batch, s2, n2, d = pool_key.shape
        if actual_seq_k is not None:
            counts = _as_int_list(actual_seq_k)
        else:
            counts = [s2] * batch
        return [pool_key[b, :c, 0, :].to(torch.float32) for b, c in enumerate(counts)]
    if layout_k == "TND":
        t2, n2, d = pool_key.shape
        if actual_seq_k is None:
            raise ValueError("layout_k='TND' requires actual_seq_k (prefix sums)")
        counts = _prefix_to_counts(actual_seq_k)
        keys, start = [], 0
        for c in counts:
            keys.append(pool_key[start : start + c, 0, :].to(torch.float32))
            start += c
        return keys
    if layout_k == "PA_BBND":
        if block_table is None:
            raise ValueError("layout_k='PA_BBND' requires block_table")
        if actual_seq_k is None:
            raise ValueError(
                "layout_k='PA_BBND' requires actual_seq_k (pools per batch)"
            )
        block_num, block_size, n2, d = pool_key.shape
        bt = block_table.detach().cpu()
        batch, max_blocks = bt.shape
        counts = _as_int_list(actual_seq_k)
        keys = []
        for b in range(batch):
            need, rows, pos = counts[b], [], 0
            for t in range(max_blocks):
                if pos >= need:
                    break
                blk = int(bt[b, t])
                if blk < 0:
                    continue
                take = min(block_size, need - pos)
                rows.append(pool_key[blk, :take, 0, :].to(torch.float32))
                pos += take
            if rows:
                keys.append(torch.cat(rows, dim=0))
            else:
                keys.append(pool_key.new_zeros((0, d)).to(torch.float32))
        return keys
    raise ValueError(f"unsupported layout_k: {layout_k}")


def _e8m0_decode(t):
    """E8M0 raw(uint8) -> float32 scale: 2^(raw-127), raw==255 -> NaN。

    与 QLIv2 golden 的 e8m0_raw_to_float 一致(0xFF 约定为 NaN, 不拦截)。
    """
    raw = t.detach().cpu().to(torch.uint8)
    scale = torch.pow(2.0, raw.to(torch.float32) - 127.0)
    return torch.where(raw == 255, torch.full_like(scale, float("nan")), scale)


def _split_q_descale_generic(ds, layout_q, q_counts):
    """通用 q_descale 拆分(任意尾维, 已解码 float) -> 逐 batch (Sb, N1, ...) 列表。"""
    if layout_q == "BSND":
        return [ds[b, :c] for b, c in enumerate(q_counts)]
    out, start = [], 0
    for c in q_counts:
        out.append(ds[start : start + c])
        start += c
    return out


def _split_k_descale_generic(
    ds, layout_k, actual_seq_k, k_counts, block_table=None, block_size=None
):
    """通用 k_descale 拆分(任意尾维, 已解码 float) -> 逐 batch (Pb, ...) 列表。

    支持与 pool_key 相同的三布局(BSND 切批 / TND 前缀切 / PA 经 block_table)。
    """
    if layout_k == "BSND":
        return [ds[b, :c] for b, c in enumerate(k_counts)]
    if layout_k == "TND":
        out, start = [], 0
        for c in k_counts:
            out.append(ds[start : start + c])
            start += c
        return out
    # PA_BBND: (blockNum, blockSize, N2, ...) 经 block_table 重排逻辑池序
    if block_table is None:
        raise ValueError("PA k_descale requires block_table")
    bt = block_table.detach().cpu()
    batch, max_blocks = bt.shape
    out = []
    for b in range(batch):
        need, rows, pos = k_counts[b], [], 0
        for t in range(max_blocks):
            if pos >= need:
                break
            blk = int(bt[b, t])
            if blk < 0:
                continue
            take = min(block_size, need - pos)
            rows.append(ds[blk, :take, 0])
            pos += take
        out.append(
            torch.cat(rows, dim=0) if rows else ds.new_zeros((0,) + ds.shape[3:])
        )
    return out


def _split_q_descale(q_descale, layout_q, actual_seq_q, q_counts):
    """q_descale(mode=0, per-token-head) -> 逐 batch (Sb, N1) float32 列表。"""
    if q_descale is None:
        return None
    ds = q_descale.detach().cpu().to(torch.float32)
    if layout_q == "BSND":
        batch, _, n1 = ds.shape
        return [ds[b, :c] for b, c in enumerate(q_counts)]
    # TND: (T1, N1) 按前缀和切
    out, start = [], 0
    for c in q_counts:
        out.append(ds[start : start + c])
        start += c
    return out


def _split_k_descale(
    k_descale, layout_k, actual_seq_k, k_counts, block_table=None, block_size=None
):
    """k_descale(mode=0) -> 逐 batch (Pb,) float32 列表。

    BSND (B,S2,N2) -> [b, :c, 0]; TND (T2,N2) 前缀切;
    PA (blockNum,blockSize,N2) 与 pool_key 同构: 逻辑池 p 的 scale 在
    物理块 block_table[b][p//bs] 第 p%bs 行。
    """
    if k_descale is None:
        return None
    ds = k_descale.detach().cpu().to(torch.float32)
    if layout_k == "BSND":
        return [ds[b, :c, 0] for b, c in enumerate(k_counts)]
    if layout_k == "TND":
        out, start = [], 0
        for c in k_counts:
            out.append(ds[start : start + c, 0])
            start += c
        return out
    # PA_BBND
    if block_table is None:
        raise ValueError("PA k_descale requires block_table")
    bt = block_table.detach().cpu()
    batch, max_blocks = bt.shape
    out = []
    for b in range(batch):
        need, vals, pos = k_counts[b], [], 0
        for t in range(max_blocks):
            if pos >= need:
                break
            blk = int(bt[b, t])
            if blk < 0:
                continue
            take = min(block_size, need - pos)
            vals.append(ds[blk, :take, 0])
            pos += take
        out.append(torch.cat(vals, dim=0) if vals else ds.new_zeros((0,)))
    return out


def _compute_batch(
    q_b,
    w_b,
    k_b,
    tail_k,
    topk,
    pool_size,
    mask_mode,
    q_scale_b=None,
    k_scale_b=None,
    mx_q_scale_b=None,
    mx_k_scale_b=None,
):
    """单个 batch 的池级 TopK 选择与索引展开。

    入参:
      q_b: (S, N1, D) float32;  w_b: (S, N1) float32;  k_b: (P, D) float32;
      tail_k: 尾部有效 token 数; topk/pool_size/mask_mode: 算子属性;
      q_scale_b/k_scale_b: mode=0 的 per-token-head 反量化系数(float32):
        q_scale_b (S, N1) / None; k_scale_b (P,) / None。
      mx_q_scale_b/mx_k_scale_b: mode=1 的 mxFP8 scale(float32, 已解码):
        mx_q_scale_b (S, N1, D/32) / None; mx_k_scale_b (P, D/32) / None。
        按 per-32-D-group 反量化后计算(参考 QLIv2 cal_atten_per_batch_mx)。

    返回:
      token_idx: (S, topk + pool_size - 1) int32, 无效 -1
      pool_val:  (S, topk // pool_size) float32,   无效 -inf
    """
    s_len, n1, d = q_b.shape
    p_len = k_b.shape[0]
    sparse_count = topk // pool_size
    out_len = topk + pool_size - 1
    l_orig = p_len * pool_size + tail_k  # 原始 token 总数

    token_idx = torch.full((s_len, out_len), -1, dtype=torch.int32)
    pool_val = torch.full((s_len, sparse_count), _NEG_INF, dtype=torch.float32)
    if s_len == 0 or p_len == 0:
        return token_idx, pool_val

    # 步骤 1: 池级分数
    #   mode=0: score[i,j] = (Σ_n relu(Σ_d q·k) · w[i,n] · dq) · dk / sqrt(d)
    #           (q_scale 乘入 weight、k_scale 乘入聚合和, 与 kernel 数值口径一致;
    #            scale 恒正可从 D 维点积提出, 数学等价于先反量化)
    #   mode=1: q/k 先按 per-32-D-group scale 反量化, 再点积(参考 QLIv2 MX golden)
    #   非量化: score[i,j] = Σ_n w[i,n] · relu(q[i,n]·k[j]/sqrt(d))
    scale = 1.0 / math.sqrt(d)
    if mx_q_scale_b is not None or mx_k_scale_b is not None:
        # mxFP8: 展开 group scale 到 D 维后反量化
        if mx_q_scale_b is not None:
            mxq = (
                mx_q_scale_b.reshape(mx_q_scale_b.shape[0], -1)
                if mx_q_scale_b.dim() > 2
                else mx_q_scale_b
            )  # (S, N1*D/32) -> (S, N1, D/32)
            mxq = mxq.reshape(mxq.shape[0], -1, d // 32)
            q_b = q_b * mxq.repeat_interleave(32, dim=-1)
        if mx_k_scale_b is not None:
            mxk = mx_k_scale_b.reshape(mx_k_scale_b.shape[0], -1)  # (P, D/32)
            k_b = k_b * mxk.repeat_interleave(32, dim=-1)
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        scores = torch.einsum("sn,snp->sp", w_b, dot.clamp_min(0.0)) * scale
    else:
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        dot = dot.clamp_min(0.0)
        if q_scale_b is not None:
            # q_scale 乘入 weight(per-token-head, 与 kernel 一致)
            w_eff = w_b * q_scale_b  # (S, N1)
        else:
            w_eff = w_b
        scores = torch.einsum("sn,snp->sp", w_eff, dot)  # (S, P)
        if k_scale_b is not None:
            # k_scale 乘入聚合和(每池 1 个标量; squeeze 吸收 N2=1 维)
            scores = scores * k_scale_b.reshape(1, p_len)
        scores = scores * scale

    arange_pool = torch.arange(pool_size, dtype=torch.int64)
    for i in range(s_len):
        # 步骤 2: 因果可见池数
        if mask_mode == 3:
            valid = (l_orig - s_len + i + 1) // pool_size  # floor 除法
            valid = max(0, min(valid, p_len))
        else:
            valid = p_len
        select = min(valid, sparse_count)
        if select > 0:
            # 步骤 3: 池级 TopK(分数降序, 同分按下标升序)
            order = torch.argsort(scores[i, :valid], descending=True, stable=True)
            picked = order[:select]
            # 步骤 4: 展开为 token 索引(依次填入, 总长 select*pool_size <= topk)
            tokens = picked.to(torch.int64).unsqueeze(1) * pool_size + arange_pool
            flat = tokens.reshape(-1)
            token_idx[i, : flat.numel()] = flat.to(torch.int32)
            pool_val[i, :select] = scores[i, picked]

        # 步骤 5: 尾块追加(输出 topk 位置起, 容量 pool_size - 1)
        if pool_size > 1 and tail_k > 0:
            if mask_mode == 0:
                visible = tail_k
            else:
                gpos = l_orig - s_len + i
                visible = max(0, min(tail_k, gpos - topk + 1))
            if visible > 0:
                tail_tokens = torch.arange(
                    l_orig - tail_k, l_orig - tail_k + visible, dtype=torch.int32
                )
                token_idx[i, topk : topk + visible] = tail_tokens

    return token_idx, pool_val


def _reference_cpu(
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
    return_value=False,
    q_descale=None,
    k_descale=None,
    quant_mode=-1,
):
    """CPU 金标实现(逐 batch Python 循环, 精确语义, 全布局能力)。"""
    # ---- 布局归一化: 逐 batch 拆分 ----
    q_list, w_list, q_counts, batch, s1_dim = _split_query(
        query, weights, layout_q, actual_seq_q
    )
    k_list = _split_key(pool_key, layout_k, actual_seq_k, block_table)
    if len(k_list) != batch:
        raise ValueError(f"batch mismatch: query {batch} vs pool_key {len(k_list)}")
    tail_list = _as_int_list(pool_tail_k) if pool_tail_k is not None else [0] * batch
    if len(tail_list) != batch:
        raise ValueError(f"pool_tail_k length {len(tail_list)} != batch {batch}")

    # ---- 量化 descale 拆分 ----
    # mode=0: q_descale (…,N1) float, dim==N1+1; k_descale (…,N2) float
    # mode=1: q_descale (…,N1,D/64,2) e8m0; k_descale (…,N2,D/64,2) e8m0
    # dispatch 依据: mode=1 的 descale 是 e8m0(uint8/float8_e8m0fnu)或解码后
    # 尾维为 D/32 的 float —— 由 run_golden 统一解码后传入(此时 shape 尾维
    # 为 D/32, dim 比 mode=0 多一维), 按尾维长度判别(D==128 时 D/32=4)
    q_scale_list = None
    k_scale_list = None
    mx_q_scale_list = None
    mx_k_scale_list = None
    is_mx = quant_mode == 1
    if q_descale is not None:
        if not is_mx:
            q_scale_list = _split_q_descale(q_descale, layout_q, actual_seq_q, q_counts)
        else:
            # mxFP8: 已解码 float, 尾维 [D/32](run_golden 展平 [D/64,2] 后)
            mx_q_scale_list = _split_q_descale_generic(q_descale, layout_q, q_counts)
    if k_descale is not None:
        k_counts = [k.shape[0] for k in k_list]
        block_size = pool_key.shape[1] if layout_k == "PA_BBND" else None
        if not is_mx:
            k_scale_list = _split_k_descale(
                k_descale, layout_k, actual_seq_k, k_counts, block_table, block_size
            )
        else:
            mx_k_scale_list = _split_k_descale_generic(
                k_descale, layout_k, actual_seq_k, k_counts, block_table, block_size
            )

    # ---- 逐 batch 计算 ----
    idx_rows, val_rows = [], []
    for b in range(batch):
        token_idx, pool_val = _compute_batch(
            q_list[b],
            w_list[b],
            k_list[b],
            tail_list[b],
            topk,
            pool_size,
            mask_mode,
            q_scale_list[b] if q_scale_list is not None else None,
            k_scale_list[b] if k_scale_list is not None else None,
            mx_q_scale_list[b] if mx_q_scale_list is not None else None,
            mx_k_scale_list[b] if mx_k_scale_list is not None else None,
        )
        idx_rows.append(token_idx)
        val_rows.append(pool_val)

    # ---- 组装输出 ----
    out_len = topk + pool_size - 1
    sparse_count = topk // pool_size
    if layout_q == "BSND":
        indices = torch.full((batch, s1_dim, out_len), -1, dtype=torch.int32)
        values = torch.full(
            (batch, s1_dim, sparse_count), _NEG_INF, dtype=torch.float32
        )
        for b in range(batch):
            c = q_counts[b]
            indices[b, :c] = idx_rows[b]
            values[b, :c] = val_rows[b]
    else:  # TND: 所有行有效, 按序拼接
        indices = (
            torch.cat(idx_rows, dim=0)
            if idx_rows
            else torch.zeros((0, out_len), dtype=torch.int32)
        )
        values = (
            torch.cat(val_rows, dim=0)
            if val_rows
            else torch.zeros((0, sparse_count), dtype=torch.float32)
        )

    if not return_value:
        values = torch.empty(0, dtype=torch.float32)
    return indices, values


# ---------------------------------------------------------------------------
# NPU 组合标杆实现(向量化, 支持不等长 batch)
# ---------------------------------------------------------------------------


def _gather_rows(src, index_2d):
    """按 (B, L) 索引矩阵 gather 首维: src (T, ...) -> (B, L, ...)。"""
    flat = src[index_2d.reshape(-1)]
    return flat.reshape(index_2d.shape[0], index_2d.shape[1], *src.shape[1:])


def _normalize_query_npu(query, weights, actual_seq_q, layout_q, device):
    """query/weights 归一化为等长稠密布局。

    返回 (q, w, q_counts_t, row_valid):
      - q: (B, S1m, N1, D); w: (B, S1m, N1) —— TND 时按批 gather 填充,
        BSND 时原样(仅记录行有效性);
      - q_counts_t: (B,) 每 batch 有效 query 行数(int64, device);
      - row_valid: (B, S1m) 有效行掩码(填充行 False)。
    """
    if layout_q == "BSND":
        b, s1m, n1, d = query.shape
        if actual_seq_q is not None:
            counts = _as_int_list(actual_seq_q)  # 非前缀和
            if len(counts) != b:
                raise ValueError(f"actual_seq_q length {len(counts)} != batch {b}")
        else:
            counts = [s1m] * b
        q_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
        row_valid = torch.arange(s1m, device=device)[None, :] < q_counts_t[:, None]
        return query, weights, q_counts_t, row_valid

    # TND: 前缀和 -> 逐批 gather 填充到 (B, S1m)
    counts = _prefix_to_counts(actual_seq_q)
    b = len(counts)
    t1, n1, d = query.shape
    if t1 != sum(counts):
        raise ValueError(f"actual_seq_q 总和({sum(counts)}) != T1({t1})")
    q_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
    s1m = max(counts) if counts else 0
    if s1m == 0:  # 全空 query: 上层直接短路返回空输出
        return (
            query.new_zeros((b, 0, n1, d)),
            weights.new_zeros((b, 0, n1)),
            q_counts_t,
            torch.zeros((b, 0), dtype=torch.bool, device=device),
        )
    row = torch.arange(s1m, device=device)
    starts = q_counts_t.cumsum(0) - q_counts_t
    gather_idx = starts[:, None] + row[None, :]  # (B, S1m)
    row_valid = row[None, :] < q_counts_t[:, None]
    safe_idx = gather_idx.masked_fill(~row_valid, 0).clamp(0, t1 - 1)
    q = _gather_rows(query, safe_idx)
    w = _gather_rows(weights, safe_idx)
    return q, w, q_counts_t, row_valid


def _normalize_key_npu(pool_key, actual_seq_k, block_table, layout_k, device):
    """pool_key 归一化为等长稠密布局 (B, Pm, D)。

    返回 (k, k_counts_t, pool_valid):
      - k: (B, Pm, D) —— TND 按批 gather 填充, PA 经 block_table gather,
        BSND 原样(仅记录列有效性);
      - k_counts_t: (B,) 每 batch 有效池数(int64, device);
      - pool_valid: (B, Pm) 有效池掩码。
    """
    if layout_k == "BSND":
        b, s2, n2, d = pool_key.shape
        if actual_seq_k is not None:
            counts = _as_int_list(actual_seq_k)  # 非前缀和
            if len(counts) != b:
                raise ValueError(f"actual_seq_k length {len(counts)} != batch {b}")
        else:
            counts = [s2] * b
        k_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
        pool_valid = torch.arange(s2, device=device)[None, :] < k_counts_t[:, None]
        return pool_key[:, :, 0, :], k_counts_t, pool_valid

    if layout_k == "TND":
        counts = _prefix_to_counts(actual_seq_k)
        b = len(counts)
        t2, n2, d = pool_key.shape
        if t2 != sum(counts):
            raise ValueError(f"actual_seq_k 总和({sum(counts)}) != T2({t2})")
        k_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
        pm = max(counts) if counts else 0
        if pm == 0:  # 全部 batch 0 池
            k = pool_key.new_zeros((b, 0, d))
            return k, k_counts_t, torch.zeros((b, 0), dtype=torch.bool, device=device)
        col = torch.arange(pm, device=device)
        starts = k_counts_t.cumsum(0) - k_counts_t
        gather_idx = starts[:, None] + col[None, :]  # (B, Pm)
        pool_valid = col[None, :] < k_counts_t[:, None]
        safe_idx = gather_idx.masked_fill(~pool_valid, 0).clamp(0, t2 - 1)
        k = _gather_rows(pool_key[:, 0, :], safe_idx)  # (B, Pm, D)
        return k, k_counts_t, pool_valid

    # PA_BBND: 经 block_table gather 归一化(块内每行 = 1 个池, LIV2 语义)
    counts = _as_int_list(actual_seq_k)  # 非前缀和(每 batch 池数)
    b = len(counts)
    block_num, block_size, n2, d = pool_key.shape
    k_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
    pm = max(counts) if counts else 0
    if pm == 0 or block_num == 0:  # 全部 batch 0 池
        k = pool_key.new_zeros((b, 0, d))
        return k, k_counts_t, torch.zeros((b, 0), dtype=torch.bool, device=device)
    if block_table is None:
        raise ValueError("layout_k='PA_BBND' requires block_table")
    p_idx = torch.arange(pm, device=device)
    blk_col = (p_idx // block_size).clamp(0, block_table.shape[1] - 1)  # (Pm,)
    # index_select 避免 torch_npu(2.9.0.post2) advanced indexing bug:
    # x[:, idx] 在 NPU 上输出周期性错乱(result[b][j] = x_flat[(b+j)%n]),
    # CPU 正确; index_select/gather 实测正确
    blk = block_table.index_select(1, blk_col).long()  # (B, Pm), 空槽 -1
    pool_valid = p_idx[None, :] < k_counts_t[:, None]
    blk_safe = blk.clamp(min=0)  # 无效槽位指向物理块 0, 数据由掩码丢弃
    rows = blk_safe * block_size + (p_idx % block_size)[None, :]  # (B, Pm)
    # 0 轴非连续 key: reshape 在视图不可用时自动拷贝, 数据语义不变
    flat = pool_key[:, :, 0, :].reshape(block_num * block_size, d)
    k = flat[rows.reshape(-1)].reshape(b, pm, d)
    return k, k_counts_t, pool_valid


def _reference_npu(
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
    return_value=False,
):
    """NPU 组合标杆实现(torch 标准算子向量化, 输入/输出不跨设备)。"""
    device = query.device
    sparse = topk // pool_size
    out_len = topk + pool_size - 1

    # ---- query 归一化: (B, S1m, N1, D) + 行有效性 ----
    q, w, q_counts_t, row_valid = _normalize_query_npu(
        query, weights, actual_seq_q, layout_q, device
    )
    b, s1m, n1, d = q.shape
    if layout_q == "TND" and s1m == 0:  # 全空 query: 直接返回空输出
        indices = torch.zeros((0, out_len), dtype=torch.int32, device=device)
        if return_value:
            values = torch.full(
                (0, sparse), _NEG_INF, dtype=torch.float32, device=device
            )
        else:
            values = torch.empty(0, dtype=torch.float32, device=device)
        return indices, values

    # ---- key 归一化: (B, Pm, D) + 池有效性 ----
    k, k_counts_t, pool_valid = _normalize_key_npu(
        pool_key, actual_seq_k, block_table, layout_k, device
    )
    if k_counts_t.shape[0] != b:
        raise ValueError(f"batch mismatch: query {b} vs pool_key {k_counts_t.shape[0]}")
    pm = k.shape[1]
    tail = pool_tail_k.detach().to(device=device, dtype=torch.int64).reshape(b)
    lens = k_counts_t * pool_size + tail  # 每 batch 原始 token 总数

    # ---- 步骤 1: 池级分数(fp32 通道), 无效池列置 -inf ----
    if pm > 0:
        dot = torch.relu(
            torch.matmul(q.float(), k.transpose(-1, -2).unsqueeze(1).float())
        )  # (B, S1m, N1, Pm)
        scores = torch.matmul(w.unsqueeze(2).float(), dot).squeeze(2) * (
            d**-0.5
        )  # (B, S1m, Pm)
        scores = scores.masked_fill(~pool_valid.unsqueeze(1), _NEG_INF)

        # ---- 步骤 2: 因果可见池数(右对齐因果; 上界由池有效性掩码覆盖) ----
        if mask_mode == 3:
            row = torch.arange(s1m, device=device)
            valid = torch.div(
                lens[:, None] - q_counts_t[:, None] + row[None, :] + 1,
                pool_size,
                rounding_mode="floor",
            ).clamp(min=0)  # (B, S1m)
            col = torch.arange(pm, device=device)
            scores = scores.masked_fill(
                col[None, None, :] >= valid[:, :, None], _NEG_INF
            )

        # ---- 步骤 3: 池级 TopK(分数降序) ----
        k_sel = min(sparse, pm)
        vals, pool_idx = _topk_desc(scores, k_sel)  # (B, S1m, k_sel)
    else:  # 全部 batch 0 池: 无池可选
        k_sel = 0
        vals = torch.full((b, s1m, 0), _NEG_INF, dtype=torch.float32, device=device)
        pool_idx = torch.zeros((b, s1m, 0), dtype=torch.int64, device=device)

    # ---- 步骤 4: 选中池展开为 token 索引, 无效位置 -1 ----
    if pool_size > 1:
        off = torch.arange(pool_size, device=device)
        expanded = (pool_idx.unsqueeze(-1) * pool_size + off).reshape(
            b, s1m, k_sel * pool_size
        )
        fin = (
            torch.isfinite(vals)
            .unsqueeze(-1)
            .expand(b, s1m, k_sel, pool_size)
            .reshape(b, s1m, k_sel * pool_size)
        )
    else:
        expanded = pool_idx
        fin = torch.isfinite(vals)
    token_idx = torch.where(fin, expanded, -1)  # (B, S1m, k_sel*ps)
    if token_idx.shape[-1] < out_len:
        token_idx = torch.nn.functional.pad(
            token_idx, (0, out_len - token_idx.shape[-1]), value=-1
        )

    # ---- 步骤 5: 尾块追加(输出 topk 位置起, 容量 pool_size - 1) ----
    if pool_size > 1:
        t = torch.arange(pool_size - 1, device=device)
        if mask_mode == 3:
            row = torch.arange(s1m, device=device)
            gpos = lens[:, None] - q_counts_t[:, None] + row[None, :]
            vis = torch.clamp(torch.minimum(tail[:, None], gpos - topk + 1), min=0)
        else:
            vis = tail[:, None].expand(b, s1m)
        base = (lens - tail)[:, None, None] + t[None, None, :]
        tail_blk = torch.where(t[None, None, :] < vis[:, :, None], base, -1)
        # 0 池 batch 尾块无意义(lens = 0*ps+tail 时基址 L_orig-tail 无定义),
        # 清 -1 对齐 _reference_cpu 的 p_len==0 短路语义
        tail_blk = tail_blk.masked_fill((k_counts_t == 0)[:, None, None], -1)
        token_idx = torch.cat(
            [token_idx[..., :topk], tail_blk.to(token_idx.dtype)], dim=-1
        )

    # ---- 步骤 6: 组装输出(dtype: int32 / float32) ----
    if layout_q == "BSND" and actual_seq_q is not None:
        # BSND + actual_seq_q: 有效长度外的行填 -1 / -inf
        token_idx = token_idx.masked_fill(~row_valid.unsqueeze(-1), -1)
        if return_value:
            vals = vals.masked_fill(~row_valid.unsqueeze(-1), _NEG_INF)

    if return_value:
        values = vals.to(torch.float32)
        if k_sel < sparse:
            values = torch.nn.functional.pad(
                values, (0, sparse - k_sel), value=_NEG_INF
            )
    else:
        values = torch.empty(0, dtype=torch.float32, device=device)

    if layout_q == "TND":
        # 反填充: 按行有效性取回全局行序(批内行序 = 前缀和顺序)
        token_idx = token_idx[row_valid]
        if return_value:
            values = values[row_valid]
    return token_idx.to(torch.int32), values


def _reference_npu_blocked(
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
    return_value=False,
):
    """NPU 组合标杆的分块 streaming 实现(与 _reference_npu 数值语义一致)。

    _reference_npu 一次性物化 (B, S1m, Pm) 稠密 scores —— 极端 shape
    (ps=1 + 池数 11~17 万)下单矩阵 50~500GB, NPU HBM 放不下; 中大 shape
    虽能放下, 但 worker 长跑后 caching allocator 碎片化, 数 GB 级分配
    逐步劣化直至 aicpu 看门狗超时(进程级不可捕获, 级联饿死后续用例)。

    本实现峰值内存 = S1m*(N1 逐头中间)*blockW + S1m*k_sel*2:
      - 逐 batch 处理, pool 维按 blockW 分块;
      - N1 头维拆为逐头循环(w 可负, 不能提前乘入 q —— relu 作用在
        dot 之后), 每头仅物化 (S1m, blockW) fp32 中间;
      - 块内 topk + 与累积缓冲 cat-topk 归并(增量 streaming topk);
      - TND 反填充用显式行号 index_select(int 索引走 gather), 避免
        布尔索引触发 torch_npu AdvancedIndex/NonZero 的 AICPU 超时。

    blockW 下限 2*k_sel(保证归并信息不丢失), 上限 16384(控制 sort 规模)。
    """
    device = query.device
    sparse = topk // pool_size
    out_len = topk + pool_size - 1

    q, w, q_counts_t, row_valid = _normalize_query_npu(
        query, weights, actual_seq_q, layout_q, device
    )
    b, s1m, n1, d = q.shape
    if layout_q == "TND" and s1m == 0:  # 全空 query: 同 _reference_npu 短路
        indices = torch.zeros((0, out_len), dtype=torch.int32, device=device)
        if return_value:
            values = torch.full(
                (0, sparse), _NEG_INF, dtype=torch.float32, device=device
            )
        else:
            values = torch.empty(0, dtype=torch.float32, device=device)
        return indices, values

    k, k_counts_t, pool_valid = _normalize_key_npu(
        pool_key, actual_seq_k, block_table, layout_k, device
    )
    pm = k.shape[1]
    tail = pool_tail_k.detach().to(device=device, dtype=torch.int64).reshape(b)
    lens = k_counts_t * pool_size + tail
    scale = float(d) ** -0.5
    k_sel = min(sparse, pm)

    block_w = min(16384, max(2 * k_sel if k_sel > 0 else 4096, 4096))
    col_ar = torch.arange(pm, device=device)

    idx_list, val_list = [], []
    for bi in range(b):
        cnt = int(q_counts_t[bi])
        n_pools = int(k_counts_t[bi])
        if s1m == 0 or pm == 0 or k_sel == 0 or n_pools == 0:
            # 0 池 batch 全 -1 短路(对齐 _reference_cpu 的 p_len==0 语义):
            # 尾块追加也不发生 —— lens = 0*ps + tail 时 "L_orig-tail" 基址
            # 无意义(池不存在), 尾块 token 不可输出
            idx_list.append(
                torch.full((s1m, out_len), -1, dtype=torch.int64, device=device)
            )
            val_list.append(
                torch.full((s1m, sparse), _NEG_INF, dtype=torch.float32, device=device)
            )
            continue

        q_b = q[bi].float()  # (S1m, N1, D)
        w_b = w[bi].float()  # (S1m, N1)
        k_b = k[bi].float()  # (Pm, D)
        pv = pool_valid[bi]  # (Pm,)
        if mask_mode == 3:
            row_ar = torch.arange(s1m, device=device)
            valid_row = torch.div(
                int(lens[bi]) - cnt + row_ar + 1, pool_size, rounding_mode="floor"
            ).clamp(min=0)  # (S1m,)
        else:
            valid_row = None

        cum_v = torch.full((s1m, k_sel), _NEG_INF, dtype=torch.float32, device=device)
        cum_i = torch.full((s1m, k_sel), -1, dtype=torch.int64, device=device)
        # 单块(pm <= block_w)时 cat-argsort 归并是纯冗余: 累积缓冲全 -inf,
        # 白排一遍 2*k_sel 宽(case 633 类 8192x16384 argsort 即来自此处,
        # 属设备上最大的单 sort 调用), 直接采用本块 topk 结果
        single_block = pm <= block_w
        for lo in range(0, pm, block_w):
            hi = min(lo + block_w, pm)
            bw = hi - lo
            kb = k_b[lo:hi]  # (bw, D)
            s_blk = torch.zeros((s1m, bw), dtype=torch.float32, device=device)
            for n in range(n1):
                dot_n = torch.relu(q_b[:, n] @ kb.transpose(0, 1))  # (S1m, bw)
                s_blk += w_b[:, n : n + 1] * dot_n
            s_blk *= scale
            # 布尔 AdvancedIndex setitem 在 torch_npu 上走 NonZero/动态索引
            # 路径, 大 shape 下有 AICPU 超时/vector core 异常前科(见
            # _topk_desc 注释), 统一改 masked_fill_(与 _reference_npu 稠密
            # 路径同款安全算子)
            s_blk.masked_fill_(~pv[None, lo:hi], _NEG_INF)
            if valid_row is not None:
                s_blk.masked_fill_(
                    col_ar[None, lo:hi] >= valid_row[:, None], _NEG_INF
                )

            kk = min(k_sel, bw)
            v_blk, i_blk = _topk_desc(s_blk, kk)  # (S1m, kk)
            i_blk = i_blk + lo
            if kk < k_sel:
                pad_v = torch.full(
                    (s1m, k_sel - kk), _NEG_INF, dtype=torch.float32, device=device
                )
                pad_i = torch.full(
                    (s1m, k_sel - kk), -1, dtype=torch.int64, device=device
                )
                v_blk = torch.cat([v_blk, pad_v], dim=-1)
                i_blk = torch.cat([i_blk, pad_i], dim=-1)
            if single_block:
                cum_v, cum_i = v_blk, i_blk
                continue
            # 增量归并: cat 累积缓冲 + 本块 topk, 再取 k_sel
            cat_v = torch.cat([cum_v, v_blk], dim=-1)
            cat_i = torch.cat([cum_i, i_blk], dim=-1)
            order = torch.argsort(cat_v, dim=-1, descending=True)[..., :k_sel]
            cum_v = cat_v.gather(-1, order)
            cum_i = cat_i.gather(-1, order)

        # ---- 展开为 token 索引(同 _reference_npu 语义) ----
        if pool_size > 1:
            off = torch.arange(pool_size, device=device)
            expanded = (cum_i.unsqueeze(-1) * pool_size + off).reshape(
                s1m, k_sel * pool_size
            )
            fin = (
                torch.isfinite(cum_v)
                .unsqueeze(-1)
                .expand(s1m, k_sel, pool_size)
                .reshape(s1m, k_sel * pool_size)
            )
        else:
            expanded = cum_i
            fin = torch.isfinite(cum_v)
        token_b = torch.where(fin, expanded, -1)  # (S1m, k_sel*ps)
        if token_b.shape[-1] < out_len:
            token_b = torch.nn.functional.pad(
                token_b, (0, out_len - token_b.shape[-1]), value=-1
            )
        elif token_b.shape[-1] > out_len:
            token_b = token_b[..., :out_len]

        # ---- 尾块追加 ----
        if pool_size > 1:
            t = torch.arange(pool_size - 1, device=device)
            if mask_mode == 3:
                row_ar = torch.arange(s1m, device=device)
                gpos = int(lens[bi]) - cnt + row_ar
                vis = torch.clamp(
                    torch.minimum(
                        torch.full_like(gpos, int(tail[bi])), gpos - topk + 1
                    ),
                    min=0,
                )
            else:
                vis = torch.full(
                    (s1m,), int(tail[bi]), dtype=torch.int64, device=device
                )
            base = (int(lens[bi]) - int(tail[bi])) + t
            tail_blk = torch.where(t[None, :] < vis[:, None], base[None, :], -1)
            token_b = torch.cat(
                [token_b[..., :topk], tail_blk.to(token_b.dtype)], dim=-1
            )

        vals_b = cum_v
        if k_sel < sparse:
            vals_b = torch.nn.functional.pad(
                vals_b, (0, sparse - k_sel), value=_NEG_INF
            )
        idx_list.append(token_b)
        val_list.append(vals_b)

    token_idx = torch.stack(idx_list, dim=0)  # (B, S1m, out_len)
    values = torch.stack(val_list, dim=0)  # (B, S1m, sparse)

    if layout_q == "BSND" and actual_seq_q is not None:
        token_idx = token_idx.masked_fill(~row_valid.unsqueeze(-1), -1)
        if return_value:
            values = values.masked_fill(~row_valid.unsqueeze(-1), _NEG_INF)

    if layout_q == "TND":
        # 反填充: 显式有效行号 index_select(int 索引走 gather,
        # 避免布尔索引触发 AdvancedIndex/NonZero 的 AICPU 超时)。
        # batch b 局部行 j -> 展平索引 b*S1m + j(有效 j < counts[b])
        rows_per_batch = q_counts_t.tolist()
        sel_parts = [
            torch.arange(bi * s1m, bi * s1m + int(c), device=device)
            for bi, c in enumerate(rows_per_batch)
        ]
        sel = (
            torch.cat(sel_parts)
            if sel_parts
            else torch.zeros(0, dtype=torch.int64, device=device)
        )
        flat_idx = token_idx.reshape(b * s1m, out_len)
        token_idx = flat_idx.index_select(0, sel)
        if return_value:
            flat_val = values.reshape(b * s1m, sparse)
            values = flat_val.index_select(0, sel)

    if return_value:
        values = values.to(torch.float32)
    else:
        values = torch.empty(0, dtype=torch.float32, device=device)
    return token_idx.to(torch.int32), values


# ---------------------------------------------------------------------------
# 对外统一入口
# ---------------------------------------------------------------------------


def pool_key_indexer_reference(
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
    return_value=False,
    q_descale=None,
    k_descale=None,
    quant_mode=-1,
):
    """PoolKeyIndexer 统一标杆(按输入设备自动选择 CPU/NPU 实现)。

    入参说明:
      query: Query 张量, dtype float16/bfloat16/float32(内部统一转 float32 计算)。
        - layout_q="BSND": shape (B, S1, N1, D);
        - layout_q="TND":  shape (T1, N1, D), T1 为所有 batch 的 query token 总和。
      pool_key: 池级 Key 张量(每行一个 pool 的 key), dtype 与 query 一致:
        - layout_k="BSND":    shape (B, S2, N2, D), S2 为每 batch 池数;
        - layout_k="TND":     shape (T2, N2, D), T2 为所有 batch 池数总和;
        - layout_k="PA_BBND": shape (blockNum, blockSize, N2, D), 块内每行一个池,
          需同时传入 block_table 与 actual_seq_k; 支持 0 轴非连续(strided view)。
      weights: 多头权重, shape (B, S1, N1)(BSND) 或 (T1, N1)(TND)。
      pool_tail_k: 每 batch 尾部不完整 pool 的有效 token 数, shape (B,), int64,
        取值范围 [0, pool_size-1]。
      actual_seq_q: 可选, 每 batch query 有效 token 数:
        - layout_q="TND":  必传, 前缀和形式(长度 B, 末项等于 T1);
        - layout_q="BSND": 可传, 每 batch 有效数(非前缀和); None 表示全部有效。
      actual_seq_k: 可选, 每 batch 有效池数:
        - layout_k="TND":     必传, 前缀和形式(末项等于 T2);
        - layout_k="PA_BBND": 必传, 每 batch 池数(非前缀和);
        - layout_k="BSND":    可传, 每 batch 池数(非前缀和); None 表示全部有效。
      block_table: layout_k="PA_BBND" 时必传, shape (B, maxBlockNumPerSeq), int32,
        元素为物理块编号, 无效槽位 -1。
      layout_q: query 布局, "BSND" 或 "TND"。
      layout_k: pool_key 布局, "BSND"、"TND" 或 "PA_BBND"。
      topk: 展开后保留的 token 数, 需满足 topk % pool_size == 0。
      pool_size: 每个 pool 包含的 token 数, 取值 [1, 128]。
      mask_mode: 0 = defaultMask(无掩码); 3 = rightDownCausal(右对齐因果)。
      return_value: True 时输出 values; False 时 values 为空 tensor。

    返回值:
      (indices, values):
        indices: int32; BSND 时 (B, S1, topk+pool_size-1), TND 时
          (T1, topk+pool_size-1); 无效位置填 -1。
        values: float32; return_value=True 时 BSND 为 (B, S1, topk//pool_size),
          TND 为 (T1, topk//pool_size), 无效位置填 -inf; return_value=False 时
          为空 tensor (0,)。

    设备选择:
      - 输入 tensor 在 CPU -> CPU 金标实现(精度基准);
      - 输入 tensor 在 NPU -> NPU 组合标杆实现(性能基线);
      - 输出与输入同设备。两种实现接口能力与计算语义一致(见模块 docstring)。
    """
    _validate_args(query, pool_key, layout_q, layout_k, topk, pool_size, mask_mode)
    if query.device.type == "cpu":
        return _reference_cpu(
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
            return_value=return_value,
            q_descale=q_descale,
            k_descale=k_descale,
            quant_mode=quant_mode,
        )
    # NPU 组合标杆暂不支持量化 descale(量化精度验证走 CPU 金标)
    if q_descale is not None or k_descale is not None:
        raise ValueError(
            "NPU reference does not support quant descale yet; use CPU golden"
        )
    return _reference_npu(
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
        return_value=return_value,
    )


# ---------------------------------------------------------------------------
# 独立运行入口: 自验(CPU demo + NPU vs CPU 一致性)
# ---------------------------------------------------------------------------


def _pack_pa_demo_key(pool_key_bsnd, k_pool_lens, block_size=16):
    """将逻辑 (sum_pools, N2, D) 池数据打包为 PA 物理块布局。

    与算子 LIV2 语义一致: 块内每行 = 1 个池(key 不做 token 展开),
    每 batch 的池独立连续填块, 物理块随机排布, 尾块不足补零。
    返回 (key_pa, block_table):
      - key_pa: (blockNum, block_size, N2, D) float16
      - block_table: (B, maxBlocks) int32, 空槽 -1
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_bsnd.shape[1], pool_key_bsnd.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_bsnd.dtype)
    pool_start = blk_start = 0
    for pools, nb in zip(k_pool_lens, blocks_per_batch):
        seg = pool_key_bsnd[pool_start : pool_start + pools]
        pad = nb * block_size - pools
        if pad:  # 尾块不足补零
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start : blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)
    # 物理块重排后按新块号重建 block_table
    bt = torch.full((len(k_pool_lens), max(blocks_per_batch)), -1, dtype=torch.int32)
    cur = 0
    for b, nb in enumerate(blocks_per_batch):
        for i in range(nb):
            bt[b, i] = perm[cur]
            cur += 1
    return blocks[perm].contiguous(), bt


# ---------------------------------------------------------------------------
# demo 公共助手: 输入构造 / 期望 shape / 输出自检
# ---------------------------------------------------------------------------


def _make_demo_query_weights(layout_q, b, s1, n1, d):
    """按 query 布局构造随机 (query, weights)。"""
    if layout_q == "BSND":
        return (
            torch.randn(b, s1, n1, d, dtype=torch.float16),
            torch.randn(b, s1, n1, dtype=torch.float16),
        )
    return (
        torch.randn(b * s1, n1, d, dtype=torch.float16),
        torch.randn(b * s1, n1, dtype=torch.float16),
    )


def _make_dense_demo_inputs(layout_q, b, s1, n1, d, pools, n2=1):
    """构造 BSND/TND 密集布局随机输入。

    返回 (query, pool_key, weights, actual_seq_q, actual_seq_k);
    BSND 时两个 actual_seq 为 None, TND 时为前缀和形式。
    """
    query, weights = _make_demo_query_weights(layout_q, b, s1, n1, d)
    if layout_q == "BSND":
        pool_key = torch.randn(b, pools, n2, d, dtype=torch.float16)
        return query, pool_key, weights, None, None
    pool_key = torch.randn(b * pools, n2, d, dtype=torch.float16)
    asq = torch.tensor([s1] * b, dtype=torch.int64).cumsum(0)
    ask = torch.tensor([pools] * b, dtype=torch.int64).cumsum(0)
    return query, pool_key, weights, asq, ask


def _demo_expected_shapes(layout_q, b, s1, topk, pool_size):
    """期望输出 shape: BSND 带 (B, S1) 维, TND 为扁平 T1 行。"""
    out_len = topk + pool_size - 1
    sparse_count = topk // pool_size
    if layout_q == "BSND":
        return (b, s1, out_len), (b, s1, sparse_count)
    return (b * s1, out_len), (b * s1, sparse_count)


def _check_demo_output(indices, values, exp_idx_shape, exp_val_shape, max_token, tag):
    """公共自检: 输出 shape / 有效索引范围 / values 有限性, 并打印 OK 日志。"""
    assert indices.shape == exp_idx_shape, (indices.shape, exp_idx_shape)
    assert values.shape == exp_val_shape, (values.shape, exp_val_shape)
    valid = indices[indices >= 0]
    assert valid.numel() > 0
    assert int(valid.min()) >= 0
    assert int(valid.max()) < max_token
    assert torch.isfinite(values[torch.isfinite(values)]).all()
    print(
        f"[demo] {tag} indices{tuple(indices.shape)} valid={valid.numel()} "
        f"values{tuple(values.shape)} -> OK"
    )


# ---------------------------------------------------------------------------
# demo 场景: 密集布局 / PA 布局(CPU 实现自验)
# ---------------------------------------------------------------------------


def _demo_dense():
    """BSND/TND 密集布局自验: mask_mode 0/3, 验证输出 shape/取值范围。"""
    torch.manual_seed(0)
    b, s1, n1, d = 2, 8, 4, 128
    topk, pool_size, pools = 32, 8, 12
    tail = torch.tensor([3, 5], dtype=torch.int64)
    max_token = pools * pool_size + int(max(tail))

    for layout_q, layout_k in [("BSND", "BSND"), ("TND", "TND")]:
        query, pool_key, weights, asq, ask = _make_dense_demo_inputs(
            layout_q, b, s1, n1, d, pools
        )
        exp_idx, exp_val = _demo_expected_shapes(layout_q, b, s1, topk, pool_size)
        for mask_mode in (0, 3):
            indices, values = pool_key_indexer_reference(
                query,
                pool_key,
                weights,
                tail,
                actual_seq_q=asq,
                actual_seq_k=ask,
                layout_q=layout_q,
                layout_k=layout_k,
                topk=topk,
                pool_size=pool_size,
                mask_mode=mask_mode,
                return_value=True,
            )
            _check_demo_output(
                indices,
                values,
                exp_idx,
                exp_val,
                max_token,
                f"layout=({layout_q},{layout_k}) mask={mask_mode}",
            )


def _demo_pa():
    """PA_BBND 场景自验: 不等池数(-1 空槽+尾块 padding)、block_size 16/32、
    mask_mode 0/3、BSND/TND 两种 query 布局, 并用同一份逻辑池数据
    交叉验证 PA 输出 == BSND 输出。"""
    torch.manual_seed(7)
    b, s1, n1, d, n2 = 2, 8, 4, 128, 1
    topk, pool_size = 32, 8
    k_pool_lens = [12, 5]  # 每 batch 池数不等 -> -1 空槽与尾块 padding
    total_pools = sum(k_pool_lens)
    tail = torch.tensor([3, 5], dtype=torch.int64)
    max_token = total_pools * pool_size + int(max(tail))

    # 逻辑池数据 (与 BSND 交叉验证共用): batch0 前 12 池, batch1 后 5 池
    logical_key = torch.randn(total_pools, n2, d, dtype=torch.float16)
    ask = torch.tensor(k_pool_lens, dtype=torch.int64)
    asq_t = torch.tensor([s1] * b, dtype=torch.int64).cumsum(0)
    # 交叉验证用 BSND key (B, S2max, N2, D): 各 batch 前 pools 行为有效池
    s2max = max(k_pool_lens)
    key_bsnd = torch.zeros(b, s2max, n2, d, dtype=logical_key.dtype)
    start = 0
    for bi, pools in enumerate(k_pool_lens):
        key_bsnd[bi, :pools] = logical_key[start : start + pools]
        start += pools

    for block_size in (16, 32):
        key_pa, bt = _pack_pa_demo_key(logical_key, k_pool_lens, block_size)
        for layout_q in ("BSND", "TND"):
            query, weights = _make_demo_query_weights(layout_q, b, s1, n1, d)
            asq = asq_t if layout_q == "TND" else None
            exp_idx, exp_val = _demo_expected_shapes(layout_q, b, s1, topk, pool_size)
            for mask_mode in (0, 3):
                indices, values = pool_key_indexer_reference(
                    query,
                    key_pa,
                    weights,
                    tail,
                    actual_seq_q=asq,
                    actual_seq_k=ask,
                    block_table=bt,
                    layout_q=layout_q,
                    layout_k="PA_BBND",
                    topk=topk,
                    pool_size=pool_size,
                    mask_mode=mask_mode,
                    return_value=True,
                )
                _check_demo_output(
                    indices,
                    values,
                    exp_idx,
                    exp_val,
                    max_token,
                    f"block_size={block_size} mask={mask_mode} ({layout_q},PA_BBND)",
                )
                if layout_q == "BSND":
                    # batch1(仅5池)选中池数被截断到 5
                    per_row_cnt = (indices[1] >= 0).sum(dim=-1)
                    assert int(
                        per_row_cnt.max()
                    ) <= topk // pool_size * pool_size + min(int(tail[1]), topk), (
                        per_row_cnt.max()
                    )
                    # 交叉验证: PA 输出 == BSND 输出(同一份逻辑池数据)
                    idx_b, val_b = pool_key_indexer_reference(
                        query,
                        key_bsnd,
                        weights,
                        tail,
                        actual_seq_k=ask,
                        layout_q="BSND",
                        layout_k="BSND",
                        topk=topk,
                        pool_size=pool_size,
                        mask_mode=mask_mode,
                        return_value=True,
                    )
                    assert torch.equal(indices, idx_b), "PA indices != BSND indices"
                    assert torch.equal(values, val_b), "PA values != BSND values"


# ---------------------------------------------------------------------------
# NPU 实现 vs CPU 实现一致性自验(需 NPU)
# ---------------------------------------------------------------------------


def _selftest_build_inputs(
    lq,
    lk,
    *,
    q_lens,
    k_pools,
    ps,
    topk,
    tail,
    dtype,
    pa_block_size=16,
    seed=7,
    key_non_contig=False,
):
    """构造一致性自验输入(支持不等长/0 池/PA 非连续)。"""
    torch.manual_seed(seed)
    b = len(q_lens)
    n1, d, n2 = 8, 128, 1
    asq = ask = bt = None
    if lq == "BSND":
        query = torch.randn(b, q_lens[0], n1, d, dtype=dtype)
        weights = torch.randn(b, q_lens[0], n1, dtype=dtype)
    else:
        query = torch.randn(sum(q_lens), n1, d, dtype=dtype)
        weights = torch.randn(sum(q_lens), n1, dtype=dtype)
        asq = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)
    if lk == "BSND":
        pool_key = torch.randn(b, k_pools[0], n2, d, dtype=dtype)
        ask = torch.tensor(k_pools, dtype=torch.int64)  # 非前缀和(CPU 标杆语义)
    elif lk == "TND":
        pool_key = torch.randn(sum(k_pools), n2, d, dtype=dtype)
        ask = torch.tensor(k_pools, dtype=torch.int64).cumsum(0)
    else:  # PA_BBND
        logical = torch.randn(sum(k_pools), n2, d, dtype=dtype)
        pool_key, bt = _pack_pa_demo_key(logical, k_pools, pa_block_size)
        ask = torch.tensor(k_pools, dtype=torch.int64)  # 非前缀和
        if key_non_contig:
            block_num, block_size = pool_key.shape[0], pool_key.shape[1]
            big = torch.zeros(2 * block_num, block_size, n2, d, dtype=dtype)
            big[0::2] = pool_key
            pool_key = big[::2]  # 0 轴非连续视图
    ptk = torch.tensor(tail, dtype=torch.int64)
    return dict(
        query=query,
        pool_key=pool_key,
        weights=weights,
        pool_tail_k=ptk,
        actual_seq_q=asq,
        actual_seq_k=ask,
        block_table=bt,
    )


def _selftest_npu_vs_cpu(dev="npu:0"):
    """NPU 组合标杆 vs CPU 金标一致性自验(等长 + 不等长 + 0 池 + PA 非连续)。"""
    from result_compare import compare_indices, compare_values

    fp16, bf16 = torch.float16, torch.bfloat16
    cases = []
    # 1) 等长基础矩阵: 4 布局 x ps{1,16} x mask{0,3}
    for ps in (1, 16):
        topk = 128 if ps == 16 else 64
        tail = [ps - 1, min(ps - 1, 7)] if ps > 1 else [0, 0]
        for lq, lk in (
            ("BSND", "BSND"),
            ("TND", "TND"),
            ("BSND", "PA_BBND"),
            ("TND", "PA_BBND"),
        ):
            for mask in (0, 3):
                cases.append(
                    dict(
                        tag=f"equal_ps{ps}_{lq}_{lk}_m{mask}",
                        lq=lq,
                        lk=lk,
                        q_lens=[64, 64],
                        k_pools=[8, 8],
                        ps=ps,
                        topk=topk,
                        tail=tail,
                        dtype=fp16,
                        mask=mask,
                    )
                )
    # 2) 不等长 query / 不等长池 / 0 长度 / 0 池(TND/PA)
    cases += [
        dict(
            tag="tnd_unequal_q",
            lq="TND",
            lk="TND",
            q_lens=[32, 48, 0, 16],
            k_pools=[16, 24, 8, 4],
            ps=16,
            topk=128,
            tail=[3, 15, 0, 7],
            dtype=bf16,
            mask=3,
        ),
        dict(
            tag="tnd_unequal_pools",
            lq="TND",
            lk="TND",
            q_lens=[32, 32],
            k_pools=[40, 6],
            ps=16,
            topk=128,
            tail=[5, 15],
            dtype=fp16,
            mask=0,
        ),
        dict(
            tag="tnd_zero_pool_batch",
            lq="TND",
            lk="TND",
            q_lens=[16, 8, 8],
            k_pools=[16, 0, 33],
            ps=16,
            topk=128,
            tail=[0, 0, 12],
            dtype=fp16,
            mask=3,
        ),
        dict(
            tag="pa_unequal_pools",
            lq="BSND",
            lk="PA_BBND",
            q_lens=[32, 32, 32],
            k_pools=[16, 0, 33],
            ps=16,
            topk=128,
            tail=[2, 0, 15],
            dtype=fp16,
            mask=0,
        ),
        dict(
            tag="pa_unequal_pools_tnd",
            lq="TND",
            lk="PA_BBND",
            q_lens=[16, 0, 8],
            k_pools=[16, 8, 40],
            ps=8,
            topk=64,
            tail=[7, 3, 0],
            dtype=bf16,
            mask=3,
        ),
        dict(
            tag="pa_noncontig",
            lq="BSND",
            lk="PA_BBND",
            q_lens=[32, 32],
            k_pools=[16, 16],
            ps=16,
            topk=128,
            tail=[5, 13],
            dtype=fp16,
            mask=0,
            key_non_contig=True,
        ),
        dict(
            tag="pa_blocksize32",
            lq="TND",
            lk="PA_BBND",
            q_lens=[16, 16],
            k_pools=[20, 12],
            ps=16,
            topk=128,
            tail=[0, 15],
            dtype=fp16,
            mask=3,
            pa_block_size=32,
        ),
        dict(
            tag="pools_lt_sparse",
            lq="BSND",
            lk="BSND",
            q_lens=[16, 16],
            k_pools=[2, 2],
            ps=16,
            topk=224,
            tail=[5, 13],
            dtype=fp16,
            mask=0,
        ),
        dict(
            tag="ps1_s1_1",
            lq="BSND",
            lk="BSND",
            q_lens=[1],
            k_pools=[32],
            ps=1,
            topk=32,
            tail=[0],
            dtype=fp16,
            mask=3,
        ),
        dict(
            tag="ps127_tail126",
            lq="BSND",
            lk="BSND",
            q_lens=[8],
            k_pools=[20],
            ps=127,
            topk=254,
            tail=[126],
            dtype=fp16,
            mask=3,
        ),
        dict(
            tag="bsnd_actual_seq",
            lq="BSND",
            lk="BSND",
            q_lens=[32, 32],
            k_pools=[16, 16],
            ps=16,
            topk=128,
            tail=[3, 15],
            dtype=fp16,
            mask=3,
            bsnd_actual_seq=True,
        ),
    ]

    n_pass = 0
    for c in cases:
        inp = _selftest_build_inputs(
            c["lq"],
            c["lk"],
            q_lens=c["q_lens"],
            k_pools=c["k_pools"],
            ps=c["ps"],
            topk=c["topk"],
            tail=c["tail"],
            dtype=c["dtype"],
            pa_block_size=c.get("pa_block_size", 16),
            key_non_contig=c.get("key_non_contig", False),
        )
        # BSND + actual_seq(CPU 标杆能力, 算子本身不支持): 有效长度 < S1/S2
        if c.get("bsnd_actual_seq"):
            inp["actual_seq_q"] = torch.tensor([20, 32], dtype=torch.int64)
            inp["actual_seq_k"] = torch.tensor([16, 10], dtype=torch.int64)

        kwargs = dict(
            actual_seq_q=inp["actual_seq_q"],
            actual_seq_k=inp["actual_seq_k"],
            block_table=inp["block_table"],
            layout_q=c["lq"],
            layout_k=c["lk"],
            topk=c["topk"],
            pool_size=c["ps"],
            mask_mode=c["mask"],
            return_value=True,
        )

        g_idx, g_val = pool_key_indexer_reference(
            inp["query"], inp["pool_key"], inp["weights"], inp["pool_tail_k"], **kwargs
        )
        n_idx, n_val = pool_key_indexer_reference(
            inp["query"].to(dev),
            inp["pool_key"].to(dev),
            inp["weights"].to(dev),
            inp["pool_tail_k"].to(dev),
            actual_seq_q=(
                inp["actual_seq_q"].to(dev) if inp["actual_seq_q"] is not None else None
            ),
            actual_seq_k=(
                inp["actual_seq_k"].to(dev) if inp["actual_seq_k"] is not None else None
            ),
            block_table=(
                inp["block_table"].to(dev) if inp["block_table"] is not None else None
            ),
            layout_q=c["lq"],
            layout_k=c["lk"],
            topk=c["topk"],
            pool_size=c["ps"],
            mask_mode=c["mask"],
            return_value=True,
        )
        n_idx, n_val = n_idx.cpu(), n_val.cpu()

        ok = compare_indices(g_idx, n_idx)
        compare_values(g_val, n_val, c["dtype"], label=f"selftest-{c['tag']}")
        print(f"[selftest] {c['tag']}: {'PASS' if ok else 'FAIL'}")
        assert ok, c["tag"]
        n_pass += 1
    print(f"pool_key_indexer_reference NPU-vs-CPU selftest passed ({n_pass} cases).")


def _demo():
    """构造随机输入自验标杆, 验证输出 shape / 取值范围 / 布局一致性。"""
    _demo_dense()
    print("pool_key_indexer_reference dense demo passed.")
    _demo_pa()
    print("pool_key_indexer_reference PA demo passed.")
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            _selftest_npu_vs_cpu()
        else:
            print("NPU not available, skip NPU-vs-CPU selftest.")
    except ImportError:
        print("torch_npu not installed, skip NPU-vs-CPU selftest.")


if __name__ == "__main__":
    _demo()
