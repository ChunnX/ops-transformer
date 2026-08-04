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
"""msa_index_score torch_extension 调用示例：非量化 + int8 前融合，对照 CPU golden。"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch_npu

import cann_ops_transformer

BLOCK_SIZE = 128
SCORE_STRIDE_ALIGN = 16
NEG_INF = -3.4028234663852886e38
ATOL = 1e-3
RTOL = 1e-3


def round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def pseudo_random(seed: int) -> float:
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    return (x / 0x7FFFFFFF) * 2.0 - 1.0


def compute_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_q: torch.Tensor,
    kv_len: torch.Tensor,
    start_loc: torch.Tensor,
    deq_scale: torch.Tensor | None,
) -> torch.Tensor:
    """CPU 参考：DOT(可选 dequant) → mask → max。"""
    total_q, hq, d = query.shape
    batch = kv_len.numel()
    max_blocks = block_table.size(1)
    score_stride = round_up(max_blocks, SCORE_STRIDE_ALIGN)
    out = torch.full((hq, total_q, score_stride), NEG_INF, dtype=torch.float32)

    q = query.float()
    k = key.float()[:, :, 0, :]  # [NP, P, D]
    for b in range(batch):
        q_begin = int(cu_seq_q[b].item())
        q_end = int(cu_seq_q[b + 1].item())
        num_blocks = (int(kv_len[b].item()) + BLOCK_SIZE - 1) // BLOCK_SIZE
        for t in range(q_begin, q_end):
            q_pos = int(start_loc[b].item()) + (t - q_begin)
            visible_end = min(q_pos + 1, int(kv_len[b].item()))
            for blk in range(num_blocks):
                key_lo = blk * BLOCK_SIZE
                valid = min(visible_end - key_lo, BLOCK_SIZE)
                if valid <= 0:
                    continue
                page = int(block_table[b, blk].item())
                k_page = k[page, :valid, :]
                if deq_scale is not None:
                    k_page = k_page * deq_scale[page, 0, :valid].unsqueeze(-1)
                s = q[t] @ k_page.T  # [Hq, valid]
                out[:, t, blk] = s.max(dim=1).values
    return out


def compare(actual: torch.Tensor, golden: torch.Tensor, name: str) -> bool:
    actual = actual.cpu().float()
    fill_a = torch.isneginf(actual)
    fill_g = torch.isneginf(golden)
    if not torch.equal(fill_a, fill_g):
        print(f"  [{name}] fill(-inf) mask mismatch -> FAIL")
        return False
    valid = ~fill_g
    if not valid.any():
        print(f"  [{name}] valid=0 -> PASS")
        return True
    diff = (actual[valid] - golden[valid]).abs()
    denom = torch.maximum(golden[valid].abs(), torch.tensor(1.0))
    bad = diff > (ATOL + RTOL * denom)
    max_diff = float(diff.max().item()) if diff.numel() else 0.0
    ok = not bool(bad.any().item())
    print(
        f"  [{name}] valid={int(valid.sum())} mismatch={int(bad.sum())} "
        f"max_abs_diff={max_diff:.6g} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_case(name: str, use_int8: bool) -> bool:
    hq, d = 2, 16
    q_len, kv_len_v, start = 2, 5, 3
    num_pages, max_blocks = 2, 1
    total_q = q_len
    score_stride = round_up(max_blocks, SCORE_STRIDE_ALIGN)

    query_f = torch.empty(total_q, hq, d, dtype=torch.float32)
    key_f = torch.empty(num_pages, BLOCK_SIZE, 1, d, dtype=torch.float32)
    for i in range(query_f.numel()):
        query_f.view(-1)[i] = pseudo_random(i + 1) * 0.5
    for i in range(key_f.numel()):
        if use_int8:
            key_f.view(-1)[i] = float(int(pseudo_random(i + 999983) * 64.0))
        else:
            key_f.view(-1)[i] = pseudo_random(i + 999983) * 0.5

    query = query_f.to(torch.float16).npu()
    if use_int8:
        key_cache = key_f.to(torch.int8).npu()
        deq = torch.empty(num_pages, 1, BLOCK_SIZE, dtype=torch.float32)
        for i in range(deq.numel()):
            deq.view(-1)[i] = 0.01 + 0.02 * (pseudo_random(i + 424242) + 1.0)
        scale = deq.npu()
    else:
        key_cache = key_f.to(torch.float16).npu()
        scale = None
        deq = None

    block_table = torch.tensor([[1 % num_pages]], dtype=torch.int32).npu()
    # 与 aclnn example 一致：打乱 page 映射
    block_table = torch.tensor([[(0 * 7 + 0 * 3 + 1) % num_pages]], dtype=torch.int32).npu()
    cu_seq_q = torch.tensor([0, q_len], dtype=torch.int32).npu()
    kv_len = torch.tensor([kv_len_v], dtype=torch.int32).npu()
    start_loc = torch.tensor([start], dtype=torch.int32).npu()

    score = cann_ops_transformer.msa_index_score(
        query,
        key_cache,
        block_table,
        cu_seq_q,
        kv_len,
        start_loc,
        scale=scale,
        block_size=BLOCK_SIZE,
        fill_invalid_tail=True,
    )
    torch.npu.synchronize()

    assert score.shape == (hq, total_q, score_stride), score.shape
    assert score.dtype == torch.float32

    golden = compute_golden(
        query.cpu().float(),
        key_f if use_int8 else key_cache.cpu().float(),
        block_table.cpu(),
        cu_seq_q.cpu(),
        kv_len.cpu(),
        start_loc.cpu(),
        deq,
    )
    return compare(score, golden, name)


def main() -> int:
    if "ASCEND_CUSTOM_OPP_PATH" not in os.environ:
        print(
            "WARN: ASCEND_CUSTOM_OPP_PATH unset; aclnnMsaIndexScore may not be found. "
            "Install custom opp and export ASCEND_CUSTOM_OPP_PATH=/path/to/vendors/custom_transformer"
        )
    print("running msa_index_score torch_extension cases")
    ok1 = run_case("L0-fp16-no-scale", use_int8=False)
    ok2 = run_case("L0-fp16-int8-dequant", use_int8=True)
    passed = int(ok1) + int(ok2)
    print(f"[{'PASS' if passed == 2 else 'FAIL'}]: {passed}/2 cases passed")
    return 0 if passed == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
