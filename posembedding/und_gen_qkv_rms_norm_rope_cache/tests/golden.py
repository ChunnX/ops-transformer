#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UndGenQkvRmsNormRopeCache 的 golden 参考实现（NPU/Ascend 950 版本）。

与竞品测试脚本 ``test_fused_qk_rmsnorm_mrope_store_kv_fp8.py`` 里的 ``_reference`` 相比：

裁掉的冗余（不在本算子范围内）
  * FP8/INT8 量化：quant_policy / k_scale / v_scale / q_scale / split_k 全部移除，
    KV Cache 固定 bf16 存储。
  * Hadamard-128 变换：2026-07-31 确认本期不做（算子未定义 use_hadamard 属性）。
  * qk_norm_policy 分支：本算子固定 policy==2（先 RMSNorm 再 MRoPE），
    policy 0/1 的路径不实现。
  * ``torch.cat([und_qkv, gen_qkv])``：按段直接 gather，避免 total*hidden 的中间拷贝。
  * num_seqlen / q_index / block_ids 三件套：已被 host 侧预计算的 slot_mapping 取代。

补齐的遗漏
  * 真正的分页 KV Cache 写入（scatter）：golden 直接产出写入后的 k_cache / v_cache，
    未被 slot_mapping 覆盖的位置保持调用方传入的原值。
  * float32 -> bf16 的显式舍入：q / k_cache / v_cache 都按 bf16 输出，
    否则与 NPU 结果比对会有系统性偏差。
  * 分页写入按 slot 拆 (block_num, row_idx)：cache 固定为连续 BBND
    [num_blocks, block_size, N, D]（2026-08-01 确认，不支持非连续 BNBD 物理布局）。
  * Optional 退化路径：gen_qkv=None / gen_len=0、cat_indices=None（恒等映射）、
    gen_weights=None；positions 允许 1D [total]（三轴广播）。
  * mrope_section 为空时退化为 [half, 0, 0]（等价标准 RoPE）。
  * 全向量化：原 golden 逐 token、逐 half 的 Python 双循环在 total=64k 时不可用。
  * 典型 case 集：headDim=128 固定，(Hq,Hk,Hv) ∈ {(8,1,1), (16,2,2)}，
    T 不设上限，case 集覆盖 1 ~ 64K；block_size 不设限，case 集覆盖 16~512（见 TYPICAL_CASES）。

约定（与《算子需求分析和方案设计说明书》一致）
  und_qkv       bf16 [und_len, N, D]，N = Hq+Hk+Hv，D = head_dim = 128
  gen_qkv       bf16 [gen_len, N, D]，可为 None
  weights       bf16 [D] x4（und_q / und_k / gen_q / gen_k）
  cos_sin_cache f32  [max_pos, D]，前半 cos 后半 sin
  positions     i64  [3, total]（或 [total]）
  cat_indices   i64  [total]，out_t -> src_t，可为 None
  slot_mapping  i64  [total]，slot = block_num * block_size + row_idx（索引类张量统一 int64）
  k_cache       bf16 [num_blocks, block_size, Hk, D]（逻辑 shape）
  v_cache       bf16 [num_blocks, block_size, Hv, D]（逻辑 shape）
  q             bf16 [total, Hq, D]

本文件是本算子的**唯一 golden 实现**，供以下用例共用：
  - tests/test_und_gen_qkv_rms_norm_rope_cache_torch.py（torch 接口 + 上板精度比对）
  - 后续接入的 ST 用例
纯 CPU torch 实现，不依赖 NPU；`python3 golden.py` 可直接自检 case 集构造是否合法。
"""

import torch

__all__ = [
    "HEAD_DIM",
    "BLOCK_SIZE",
    "HEAD_COMBOS",
    "TYPICAL_CASES",
    "make_cos_sin_cache",
    "mrope_axis_map",
    "golden_und_gen_qkv_rms_norm_rope_cache",
    "golden_dense",
    "gather_cache_rows",
    "build_case",
]

# 本期支持范围（设计说明书 §2.3 典型 case shape）
HEAD_DIM = 128
BLOCK_SIZE = 128
HEAD_COMBOS = ((8, 1, 1), (16, 2, 2))


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def make_cos_sin_cache(max_pos, head_dim, device="cpu", base=10000.0):
    """构造 cos_sin_cache [max_pos, head_dim]，前半 cos 后半 sin（与竞品一致）。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def mrope_axis_map(head_dim, mrope_section):
    """预计算 half 个索引各自取哪一轴的 cos/sin（Host Tiling 下发的 axisLut）。

    规则（与竞品 _mrope 完全一致）：
        axis = 0
        if i % 3 == 1 and i < 3 * sec[1]: axis = 1
        if i % 3 == 2 and i < 3 * sec[2]: axis = 2

    这三个数**不是**对 half 的划分：只有 sec[1]/sec[2] 被读，sec[0] 从不参与计算，
    T 是"其余全归它"的兜底轴。所以 [16,16,16] 实际得到 T/H/W = 32/16/16 而不是
    16/16/16，[64,16,16] 与 [0,16,16] 的轴映射逐位相同。
    """
    half = head_dim // 2
    if mrope_section is None or len(mrope_section) == 0:
        mrope_section = [half, 0, 0]
    assert len(mrope_section) == 3, "mrope_section 必须是长度 3 的列表"
    # 只是挡手误的粗筛，不是语义要求：参考实现没有这条约束，它自己的用例就是 sum=48。
    # 要放宽得连 op_host/..._base_tiling.cpp 的 CheckAttrsValid 一起改。
    assert sum(mrope_section) <= half, "mrope_section 三轴之和不能超过 head_dim/2"

    idx = torch.arange(half, dtype=torch.int64)
    axis = torch.zeros(half, dtype=torch.int64)
    axis[(idx % 3 == 1) & (idx < 3 * int(mrope_section[1]))] = 1
    axis[(idx % 3 == 2) & (idx < 3 * int(mrope_section[2]))] = 2
    return axis


def _normalize_qkv(x, num_heads_total, name):
    """QKV 输入约定为 3D [T, N, D]；为兼容旧脚本也接受 2D [T, N*D]。"""
    if x is None:
        return None
    if x.dim() == 2:
        assert x.shape[1] % num_heads_total == 0, f"{name} 的 hidden 无法被 N={num_heads_total} 整除"
        x = x.reshape(x.shape[0], num_heads_total, x.shape[1] // num_heads_total)
    assert x.dim() == 3, f"{name} 期望 3D [T, N, D]，实得 {tuple(x.shape)}"
    assert x.shape[1] == num_heads_total, (
        f"{name} 的 N={x.shape[1]} 与 Hq+Hk+Hv={num_heads_total} 不一致"
    )
    return x


def _normalize_positions(positions, total):
    """positions 支持 [3, total] 与 [total]（单序列，三轴广播）。"""
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    assert positions.shape == (3, total), f"positions 期望 [3, {total}]，实得 {tuple(positions.shape)}"
    return positions.to(torch.int64)


def _rmsnorm(x, weight, eps):
    """x: [T, H, D] float32；weight: [T, D] float32（每 token 已按 und/gen 选好）。"""
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv * weight.unsqueeze(1)


def _rope(x, cos, sin):
    """x: [T, H, D] float32；cos/sin: [T, half] float32（已按 mask 合并三轴）。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# --------------------------------------------------------------------------- #
# 稠密段：split + index + rmsnorm + mrope（不含 cache 写入）
# --------------------------------------------------------------------------- #
def golden_dense(
    und_qkv,
    gen_qkv,
    und_weights_q,
    und_weights_k,
    gen_weights_q,
    gen_weights_k,
    cos_sin_cache,
    positions,
    cat_indices,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    norm_eps=1e-6,
    mrope_section=None,
):
    """返回 float32 的 (q, k, v)，shape 分别 [total, Hq/Hk/Hv, D]。

    这是 KV Cache scatter 之前的中间结果，供精度定界与单元比对使用。
    """
    device = und_qkv.device
    num_heads_total = num_heads_q + num_heads_k + num_heads_v
    und_qkv = _normalize_qkv(und_qkv, num_heads_total, "und_qkv")
    gen_qkv = _normalize_qkv(gen_qkv, num_heads_total, "gen_qkv")

    und_len, _, head_dim = und_qkv.shape
    gen_len = 0 if gen_qkv is None else gen_qkv.shape[0]
    if gen_qkv is not None:
        assert gen_qkv.shape[2] == head_dim, "gen_qkv 的 head_dim 必须与 und_qkv 一致"

    total = (und_len + gen_len) if cat_indices is None else cat_indices.numel()
    if cat_indices is None:
        src = torch.arange(total, dtype=torch.int64, device=device)  # 恒等映射
    else:
        assert cat_indices.dtype in (torch.int64, torch.int32), "cat_indices 应为 int64"
        src = cat_indices.to(torch.int64)
    assert int(src.min()) >= 0 and int(src.max()) < und_len + gen_len, "cat_indices 越界"

    is_und = src < und_len

    # ---- index：按 src_t 从 und/gen 两段各自 gather，不做 concat ----
    rows = torch.empty(total, num_heads_total, head_dim, dtype=torch.float32, device=device)
    if bool(is_und.any()):
        rows[is_und] = und_qkv[src[is_und]].float()
    if bool((~is_und).any()):
        assert gen_qkv is not None, "src_t >= und_len 但 gen_qkv 为 None"
        rows[~is_und] = gen_qkv[src[~is_und] - und_len].float()

    # ---- split（N 维切 Q/K/V）----
    q = rows[:, :num_heads_q, :]
    k = rows[:, num_heads_q:num_heads_q + num_heads_k, :]
    v = rows[:, num_heads_q + num_heads_k:, :]

    # ---- rmsnorm（按 token 选 und/gen 权重）----
    und_wq = und_weights_q.float()
    und_wk = und_weights_k.float()
    gen_wq = und_wq if gen_weights_q is None else gen_weights_q.float()
    gen_wk = und_wk if gen_weights_k is None else gen_weights_k.float()
    sel = is_und.unsqueeze(-1)
    w_q = torch.where(sel, und_wq.unsqueeze(0), gen_wq.unsqueeze(0))  # [total, D]
    w_k = torch.where(sel, und_wk.unsqueeze(0), gen_wk.unsqueeze(0))
    q = _rmsnorm(q, w_q, norm_eps)
    k = _rmsnorm(k, w_k, norm_eps)

    # ---- MRoPE：三轴 cos/sin 按 axisLut 合并成一份，再做标准 RoPE ----
    half = head_dim // 2
    positions = _normalize_positions(positions, total)
    axis = mrope_axis_map(head_dim, mrope_section).to(device)          # [half]
    pos_sel = positions[axis]                                          # [half, total]
    pos_sel = pos_sel.transpose(0, 1).contiguous()                     # [total, half]
    assert int(pos_sel.min()) >= 0 and int(pos_sel.max()) < cos_sin_cache.shape[0], (
        "positions 超出 cos_sin_cache 的 max_pos 范围"
    )
    col = torch.arange(half, dtype=torch.int64, device=device).unsqueeze(0)  # [1, half]
    cos_sin_f32 = cos_sin_cache.float()
    cos = cos_sin_f32[pos_sel, col]                                    # [total, half]
    sin = cos_sin_f32[pos_sel, col + half]
    q = _rope(q, cos, sin)
    k = _rope(k, cos, sin)

    # V 分支：既不 rmsnorm 也不 rope，仅 index 后直通
    return q, k, v.contiguous()


# --------------------------------------------------------------------------- #
# 分页 KV Cache 写入 / 读回
# --------------------------------------------------------------------------- #
def _check_cache(cache, name):
    """cache 固定为连续 BBND：[num_blocks, block_size, N, D]。"""
    assert cache.dim() == 4, f"{name} 期望 4D [num_blocks, block_size, N, D]，实得 {tuple(cache.shape)}"
    assert cache.is_contiguous(), f"{name} 必须内存连续（BBND），本算子不支持非连续布局"
    return cache


def _scatter_cache(cache, slot_mapping, data):
    """按 slot_mapping 原地写入；data: [total, N, D]（已是 cache 的 dtype）。"""
    num_blocks, block_size = cache.shape[0], cache.shape[1]
    assert slot_mapping.dtype in (torch.int64, torch.int32), "slot_mapping 应为 int64（兼容 int32）"
    slot = slot_mapping.to(torch.int64)
    assert int(slot.min()) >= 0 and int(slot.max()) < num_blocks * block_size, "slot_mapping 越界"
    cache[slot // block_size, slot % block_size] = data
    return cache


def gather_cache_rows(cache, slot_mapping):
    """按 slot_mapping 从 cache 读回 [total, N, D]，用于精度比对。"""
    block_size = cache.shape[1]
    slot = slot_mapping.to(torch.int64)
    return cache[slot // block_size, slot % block_size]


# --------------------------------------------------------------------------- #
# 算子 golden 主入口
# --------------------------------------------------------------------------- #
def golden_und_gen_qkv_rms_norm_rope_cache(
    und_qkv,
    und_weights_q,
    und_weights_k,
    cos_sin_cache,
    k_cache,
    v_cache,
    slot_mapping,
    positions,
    gen_qkv=None,
    gen_weights_q=None,
    gen_weights_k=None,
    cat_indices=None,
    num_heads_q=8,
    num_heads_k=1,
    num_heads_v=1,
    norm_eps=1e-6,
    mrope_section=None,
    inplace=False,
):
    """返回 (q, k_cache, v_cache)，dtype 全为 bf16。

    k_cache / v_cache 语义与算子一致：调用方预分配，算子原地写入；
    未被 slot_mapping 命中的位置保持传入的原值。默认返回副本（inplace=False），
    传 inplace=True 则直接改写入参。
    """
    q_f32, k_f32, v_f32 = golden_dense(
        und_qkv,
        gen_qkv,
        und_weights_q,
        und_weights_k,
        gen_weights_q,
        gen_weights_k,
        cos_sin_cache,
        positions,
        cat_indices,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        norm_eps,
        mrope_section,
    )

    total = q_f32.shape[0]
    assert slot_mapping.numel() == total, (
        f"slot_mapping 长度 {slot_mapping.numel()} 与 total {total} 不一致"
    )
    assert torch.unique(slot_mapping.to(torch.int64)).numel() == total, (
        "slot_mapping 存在重复 slot，多核写入结果不确定"
    )

    _check_cache(k_cache, "k_cache")
    _check_cache(v_cache, "v_cache")
    k_out = k_cache if inplace else k_cache.clone()
    v_out = v_cache if inplace else v_cache.clone()

    # 写入前 Cast float32 -> bf16（与 Kernel 的 Cast + DataCopy 对齐）
    _scatter_cache(k_out, slot_mapping, k_f32.to(k_out.dtype))
    _scatter_cache(v_out, slot_mapping, v_f32.to(v_out.dtype))

    q_out = q_f32.to(torch.bfloat16)
    return q_out, k_out, v_out


# --------------------------------------------------------------------------- #
# 典型 case 集
# --------------------------------------------------------------------------- #
# headDim 固定 128；(Hq,Hk,Hv) 取 (8,1,1) / (16,2,2)；cache block_size=128；
# T = und_len + gen_len 不设上限（算子侧只要求为正，真实上限是 KV Cache 容量）；
# 下面的 case 集覆盖到 64K，再大只受跑用例的机器内存限制。
# slot_mode: "shuffled"（分页乱序，默认）/ "contiguous"（顺序占位）
# NOTE: 当前算子实现只支持 gen_qkv / gen_weights_q / gen_weights_k / cat_indices
#       全部提供的场景，因此所有 case 均 gen_len > 0 且 cat_indices 非 None。
#       golden 本身仍支持缺省退化路径（见 golden_dense），待算子放开后可直接加用例。
TYPICAL_CASES = (
    dict(name="t2_min",            heads=(8, 1, 1),  und_len=1,     gen_len=1,
         cat="shuffled", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t8_shuffle",        heads=(8, 1, 1),  und_len=5,     gen_len=3,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t56_cores",         heads=(16, 2, 2), und_len=40,    gen_len=16,
         cat="identity", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t96_mixed",         heads=(16, 2, 2), und_len=80,    gen_len=16,
         cat="identity", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t128_block_edge",   heads=(8, 1, 1),  und_len=127,   gen_len=1,
         cat="shuffled", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t1024_decode",      heads=(8, 1, 1),  und_len=1,     gen_len=1023,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t4096_und_heavy",   heads=(16, 2, 2), und_len=4095,  gen_len=1,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t4096_plain_rope",  heads=(8, 1, 1),  und_len=4095,  gen_len=1,
         cat="identity", slot_mode="contiguous", mrope_section=[],           big=False),
    dict(name="t10752_typical_h8", heads=(8, 1, 1),  und_len=6652,  gen_len=4100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t10752_typical_h16", heads=(16, 2, 2), und_len=6652, gen_len=4100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    # block_size 覆盖：算子对 Bs 无假设，这几条把 2 的幂与非 2 的幂、小块与大块都过一遍
    dict(name="t64_bs16",          heads=(8, 1, 1),  und_len=40,    gen_len=24,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=16),
    dict(name="t200_bs64",         heads=(16, 2, 2), und_len=150,   gen_len=50,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=64),
    dict(name="t300_bs100",        heads=(8, 1, 1),  und_len=200,   gen_len=100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=100),
    dict(name="t512_bs256",        heads=(8, 1, 1),  und_len=256,   gen_len=256,
         cat="identity", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False, block_size=256),
    dict(name="t1024_bs512",       heads=(16, 2, 2), und_len=512,   gen_len=512,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=512),
    dict(name="t65536_h8",         heads=(8, 1, 1),  und_len=32768, gen_len=32768,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=True),
    dict(name="t65536_h16",        heads=(16, 2, 2), und_len=32768, gen_len=32768,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=True),
)


def build_case(spec, device="cpu", seed=7, max_pos=4096, init_cache="randn"):
    """按 case spec 造一套完整输入，返回可直接 ``**kwargs`` 调 golden 的 dict。

    额外键（下划线开头）是 case 元信息，调用前用 ``pop`` 去掉即可，
    ``run_case`` 已代为处理。
    """
    torch.manual_seed(seed)
    hq, hk, hv = spec["heads"]
    assert (hq, hk, hv) in HEAD_COMBOS, f"{spec['name']}: 不在支持的 head 组合内"
    n = hq + hk + hv
    d = HEAD_DIM
    und_len, gen_len = spec["und_len"], spec["gen_len"]
    total = und_len + gen_len
    assert total >= 1, "T 必须为正"

    # KV Cache 预留：按 block_size 向上取整再多给 2 个 block，制造未命中位置
    # block_size 由 case 指定（默认 BLOCK_SIZE）：算子不限制 Bs，cache 展平后 slot 直接
    # 当行号用，Bs 只影响 [Bn, Bs, ...] 怎么切分同一片连续内存
    bs = int(spec.get("block_size", BLOCK_SIZE))
    assert bs >= 1, "block_size 必须为正"
    num_blocks = (total + bs - 1) // bs + 2
    k_shape, v_shape = (num_blocks, bs, hk, d), (num_blocks, bs, hv, d)  # 连续 BBND
    init = torch.randn if init_cache == "randn" else torch.zeros

    if spec["cat"] == "none":
        cat_indices = None
    elif spec["cat"] == "identity":
        cat_indices = torch.arange(total, dtype=torch.int64, device=device)
    else:  # shuffled：und/gen 交错，模拟真实 cat_indices
        cat_indices = torch.randperm(total, device=device).to(torch.int64)
    assert spec["cat"] == "none" or gen_len > 0 or und_len == total

    if spec["slot_mode"] == "contiguous":
        slot_mapping = torch.arange(total, dtype=torch.int64, device=device)
    else:
        slot_mapping = torch.randperm(num_blocks * bs, device=device)[:total].to(torch.int64)

    case = dict(
        und_qkv=torch.randn(und_len, n, d, dtype=torch.bfloat16, device=device),
        und_weights_q=torch.randn(d, dtype=torch.bfloat16, device=device),
        und_weights_k=torch.randn(d, dtype=torch.bfloat16, device=device),
        cos_sin_cache=make_cos_sin_cache(max_pos, d, device=device),
        k_cache=init(k_shape, dtype=torch.bfloat16, device=device),
        v_cache=init(v_shape, dtype=torch.bfloat16, device=device),
        slot_mapping=slot_mapping,
        positions=torch.randint(0, max_pos, (3, total), dtype=torch.int64, device=device),
        gen_qkv=(torch.randn(gen_len, n, d, dtype=torch.bfloat16, device=device)
                 if gen_len > 0 else None),
        gen_weights_q=(torch.randn(d, dtype=torch.bfloat16, device=device) if gen_len > 0 else None),
        gen_weights_k=(torch.randn(d, dtype=torch.bfloat16, device=device) if gen_len > 0 else None),
        cat_indices=cat_indices,
        num_heads_q=hq,
        num_heads_k=hk,
        num_heads_v=hv,
        norm_eps=1e-6,
        mrope_section=spec["mrope_section"],
    )
    case["_meta"] = dict(name=spec["name"], total=total, heads=(hq, hk, hv),
                         num_blocks=num_blocks, block_size=bs)
    return case


# --------------------------------------------------------------------------- #
# 自检：仅验证 case 集构造与 golden 可跑通（纯 CPU，不涉及 NPU）
# --------------------------------------------------------------------------- #
def _self_check(include_big=False):
    print(f"golden case set (headDim={HEAD_DIM}, block_size={BLOCK_SIZE}, heads in "
          f"{{{HEAD_COMBOS[0]}, {HEAD_COMBOS[1]}}})")
    for spec in TYPICAL_CASES:
        total = spec["und_len"] + spec["gen_len"]
        if spec["big"] and not include_big:
            print(f"  [SKIP] {spec['name']:<22} T={total}")
            continue
        case = build_case(spec)
        meta = case.pop("_meta")
        q, k_cache, v_cache = golden_und_gen_qkv_rms_norm_rope_cache(**case)
        assert q.shape == (total, meta["heads"][0], HEAD_DIM)
        assert q.dtype == torch.bfloat16
        print(f"  [OK]   {meta['name']:<22} T={total:<6} heads={meta['heads']} "
              f"q={tuple(q.shape)} k_cache={tuple(k_cache.shape)}")
    print("golden self-check PASS")


if __name__ == "__main__":
    import sys
    _self_check(include_big="--full" in sys.argv)
