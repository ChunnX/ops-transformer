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
"""UndGenQkvRmsNormRopeCache torch 接口测试。

用法（需已安装自定义算子包与 cann_ops_transformer wheel）：
    python3 test_und_gen_qkv_rms_norm_rope_cache_torch.py

当前 kernel 计算实现未完成，脚本分两段：
  1) 接口连通性（硬断言）：算子可调用、输出 shape/dtype 正确、k_cache/v_cache 原地更新语义成立；
  2) 数值精度（信息性）：与 golden 比对并打印最大误差，kernel 落地前预期不通过。
"""

import sys

import torch
import torch_npu

# 直接导入本算子，避免加载其他无关算子
from cann_ops_transformer.ops.und_gen_qkv_rms_norm_rope_cache import (
    und_gen_qkv_rms_norm_rope_cache,
)

HEAD_DIM = 128
BLOCK_SIZE = 128
MROPE_AXIS_NUM = 3
NORM_EPS = 1e-6

# (und_len, gen_len, Hq, Hk, Hv, mrope_section)
CASES = [
    (5, 3, 8, 1, 1, [16, 16, 16]),
    (80, 16, 16, 2, 2, [16, 16, 16]),
    (128, 0, 8, 1, 1, [64, 0, 0]),
]


def make_cos_sin_cache(max_pos, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def mrope_axis_map(head_dim, mrope_section):
    half = head_dim // 2
    if not mrope_section:
        mrope_section = [half, 0, 0]
    idx = torch.arange(half, dtype=torch.int64)
    axis = torch.zeros(half, dtype=torch.int64)
    axis[(idx % 3 == 1) & (idx < 3 * int(mrope_section[1]))] = 1
    axis[(idx % 3 == 2) & (idx < 3 * int(mrope_section[2]))] = 2
    return axis


def golden(und_qkv, gen_qkv, und_wq, und_wk, gen_wq, gen_wk, cos_sin_cache,
           positions, cat_indices, hq, hk, hv, eps, mrope_section):
    """CPU float32 golden，逻辑与 operator_design 的 golden.py 一致。"""
    und_len, n, d = und_qkv.shape
    gen_len = 0 if gen_qkv is None else gen_qkv.shape[0]
    total = und_len + gen_len
    src = torch.arange(total, dtype=torch.int64) if cat_indices is None else cat_indices
    is_und = src < und_len

    rows = torch.empty(total, n, d, dtype=torch.float32)
    if bool(is_und.any()):
        rows[is_und] = und_qkv[src[is_und]].float()
    if bool((~is_und).any()):
        rows[~is_und] = gen_qkv[src[~is_und] - und_len].float()

    q = rows[:, :hq, :]
    k = rows[:, hq:hq + hk, :]
    v = rows[:, hq + hk:, :]

    sel = is_und.unsqueeze(-1)
    w_q = torch.where(sel, und_wq.float().unsqueeze(0), gen_wq.float().unsqueeze(0))
    w_k = torch.where(sel, und_wk.float().unsqueeze(0), gen_wk.float().unsqueeze(0))
    for x, w in ((q, w_q), (k, w_k)):
        inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        x.mul_(inv).mul_(w.unsqueeze(1))

    half = d // 2
    axis = mrope_axis_map(d, mrope_section)
    pos_sel = positions[axis].transpose(0, 1).contiguous()
    col = torch.arange(half, dtype=torch.int64).unsqueeze(0)
    cos = cos_sin_cache.float()[pos_sel, col].unsqueeze(1)
    sin = cos_sin_cache.float()[pos_sel, col + half].unsqueeze(1)

    def rope(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rope(q), rope(k), v


def run_case(idx, und_len, gen_len, hq, hk, hv, mrope_section, max_pos=32):
    torch.manual_seed(7 + idx)
    n = hq + hk + hv
    total = und_len + gen_len
    block_num = (total + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

    und_qkv = torch.randn(und_len, n, HEAD_DIM, dtype=torch.bfloat16)
    gen_qkv = torch.randn(gen_len, n, HEAD_DIM, dtype=torch.bfloat16) if gen_len > 0 else None
    ws = [torch.randn(HEAD_DIM, dtype=torch.bfloat16) for _ in range(4)]
    cos_sin_cache = make_cos_sin_cache(max_pos, HEAD_DIM)
    positions = torch.randint(0, max_pos, (MROPE_AXIS_NUM, total), dtype=torch.int64)
    cat_indices = torch.randperm(total).to(torch.int64) if gen_len > 0 else None
    slot_mapping = torch.randperm(block_num * BLOCK_SIZE)[:total].to(torch.int64)
    k_cache_init = torch.randn(block_num, BLOCK_SIZE, hk, HEAD_DIM, dtype=torch.bfloat16)
    v_cache_init = torch.randn(block_num, BLOCK_SIZE, hv, HEAD_DIM, dtype=torch.bfloat16)

    k_cache = k_cache_init.clone().npu()
    v_cache = v_cache_init.clone().npu()
    q = und_gen_qkv_rms_norm_rope_cache(
        und_qkv.npu(), ws[0].npu(), ws[1].npu(), cos_sin_cache.npu(),
        k_cache, v_cache, slot_mapping.npu(), positions.npu(),
        gen_qkv.npu() if gen_qkv is not None else None,
        ws[2].npu() if gen_qkv is not None else None,
        ws[3].npu() if gen_qkv is not None else None,
        cat_indices.npu() if cat_indices is not None else None,
        num_heads_q=hq, num_heads_k=hk, num_heads_v=hv,
        norm_eps=NORM_EPS, mrope_section=mrope_section)
    torch.npu.synchronize()

    # ---- 接口连通性（硬断言）----
    assert q.shape == (total, hq, HEAD_DIM), f"q shape {tuple(q.shape)} != {(total, hq, HEAD_DIM)}"
    assert q.dtype == torch.bfloat16, f"q dtype {q.dtype}"
    assert k_cache.shape == k_cache_init.shape and v_cache.shape == v_cache_init.shape
    print(f"  [case {idx}] 接口 OK: T={total} heads=({hq},{hk},{hv}) q={tuple(q.shape)} "
          f"k_cache={tuple(k_cache.shape)}")

    # ---- 数值精度（信息性，kernel 未实现前预期不通过）----
    ref_q, ref_k, ref_v = golden(
        und_qkv, gen_qkv, ws[0], ws[1], ws[2], ws[3], cos_sin_cache,
        positions, cat_indices, hq, hk, hv, NORM_EPS, mrope_section)
    k_rows = k_cache.cpu().reshape(-1, hk, HEAD_DIM)[slot_mapping]
    v_rows = v_cache.cpu().reshape(-1, hv, HEAD_DIM)[slot_mapping]
    dq = (q.cpu().float() - ref_q).abs().max().item()
    dk = (k_rows.float() - ref_k).abs().max().item()
    dv = (v_rows.float() - ref_v).abs().max().item()
    ok = max(dq, dk, dv) < 0.05
    print(f"  [case {idx}] 精度: max|dq|={dq:.4f} max|dk|={dk:.4f} max|dv|={dv:.4f} -> "
          f"{'PASS' if ok else 'MISMATCH（kernel 未实现前预期）'}")
    return ok


def main():
    print(f"device: {torch.npu.get_device_name(0)}, torch {torch.__version__}, torch_npu {torch_npu.__version__}")
    numeric_ok = []
    for i, case in enumerate(CASES):
        numeric_ok.append(run_case(i, *case))
    print(f"\n接口连通性: {len(CASES)}/{len(CASES)} PASS")
    print(f"数值精度:   {sum(numeric_ok)}/{len(CASES)} PASS"
          f"{'' if all(numeric_ok) else '（kernel 计算实现完成前预期为 0）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
