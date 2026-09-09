#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn, float8_e5m2


def broadcast_scale(scale_2d, k):
    """[rows, ceil(k/64), 2] -> [rows, k] (每个 scale 值覆盖 32 个连续 k 元素)"""
    flat = scale_2d.reshape(scale_2d.shape[0], -1)  # [rows, ceil(k/32)]
    return np.repeat(flat, 32, axis=-1)[:, :k]


def make_counts(rank_num, expert_num, m, rng):
    """生成计数矩阵 counts[src][dst*E+e]，每行和为 m（本 rank 全部 token 发出）"""
    seg = rank_num * expert_num
    counts = np.zeros((rank_num, seg), dtype=np.uint32)
    for s in range(rank_num):
        # 多项式分布分割，允许 0 段
        draws = rng.multinomial(m, np.full(seg, 1.0 / seg))
        counts[s] = draws.astype(np.uint32)
    return counts


def gen_golden(m, k, n, rank_num, expert_num, trans_b, seed, a_dtype, b_dtype):
    rng = np.random.default_rng(seed)
    counts = make_counts(rank_num, expert_num, m, rng)

    np_dt = {"e4m3": float8_e4m3fn, "e5m2": float8_e5m2}

    # 每个 rank 的 token 输入（按 (dst, expert) 有序）与 scale
    x_list = []
    x_scale_list = []
    for s in range(rank_num):
        x = rng.uniform(1, 8, (m, k)).astype(np_dt[a_dtype])
        x_scale = rng.uniform(1, 8, (m, math.ceil(k / 64), 2)).astype(float8_e8m0)
        x_list.append(x)
        x_scale_list.append(x_scale)

    # 每个 rank 的专家权重与 scale
    w_list = []
    w_scale_list = []
    for d in range(rank_num):
        if trans_b:
            w = rng.uniform(1, 8, (expert_num, n, k)).astype(np_dt[b_dtype])
            w_scale = rng.uniform(1, 8, (expert_num, n, math.ceil(k / 64), 2)).astype(float8_e8m0)
        else:
            w = rng.uniform(1, 8, (expert_num, k, n)).astype(np_dt[b_dtype])
            w_scale = rng.uniform(1, 8, (expert_num, math.ceil(k / 64), n, 2)).astype(float8_e8m0)
        w_list.append(w)
        w_scale_list.append(w_scale)

    # golden：每 rank 按 expert-major 顺序计算
    golden_list = []
    recv_rows_list = []
    for d in range(rank_num):
        # 组装 permute 后的 token（expert-major，组内按 src rank 顺序）
        rows_all = []
        for e in range(expert_num):
            for s in range(rank_num):
                cnt = int(counts[s][d * expert_num + e])
                if cnt == 0:
                    continue
                # rank s 的 gmm_x 中 (d, e) 段的起始行
                start = sum(int(v) for v in counts[s][: d * expert_num])
                start += sum(int(v) for v in counts[s][d * expert_num: d * expert_num + e])
                rows_all.append((s, start, cnt))
        total_recv = sum(cnt for _, _, cnt in rows_all)

        perm_x = np.zeros((total_recv, k), dtype=np.float32)
        perm_scale = np.zeros((total_recv, math.ceil(k / 64), 2), dtype=float8_e8m0)
        row_off = 0
        for s, start, cnt in rows_all:
            perm_x[row_off: row_off + cnt] = x_list[s][start: start + cnt].astype(np.float32)
            perm_scale[row_off: row_off + cnt] = x_scale_list[s][start: start + cnt]
            row_off += cnt

        x_deq = perm_x * broadcast_scale(perm_scale.astype(np.float32), k)

        y = np.zeros((total_recv, n), dtype=np.float32)
        group_start = 0
        for e in range(expert_num):
            group_rows = sum(int(counts[s][d * expert_num + e]) for s in range(rank_num))
            if group_rows == 0:
                continue
            if trans_b:
                w_deq = w_list[d][e].astype(np.float32) * broadcast_scale(
                    w_scale_list[d][e].astype(np.float32), k)
                y[group_start: group_start + group_rows] = (
                    x_deq[group_start: group_start + group_rows] @ w_deq.T
                )
            else:
                # 非转置 scaleB: [K/64, N, 2]，[g][n][j] 覆盖 K 区间 [64g+32j, 64g+32j+32)
                ws = w_scale_list[d][e].astype(np.float32)  # [K/64, N, 2]
                k64 = ws.shape[0]
                w_deq_scale = np.zeros((k, n), dtype=np.float32)
                for g in range(k64):
                    for j in range(2):
                        lo = g * 64 + j * 32
                        hi = min(lo + 32, k)
                        if hi > lo:
                            w_deq_scale[lo:hi, :] = ws[g, :, j]
                w_deq = w_list[d][e].astype(np.float32) * w_deq_scale
                y[group_start: group_start + group_rows] = (
                    x_deq[group_start: group_start + group_rows] @ w_deq
                )
            group_start += group_rows

        golden_list.append(torch.from_numpy(y).to(torch.bfloat16))
        recv_rows_list.append(total_recv)

    return counts, x_list, x_scale_list, w_list, w_scale_list, golden_list, recv_rows_list


def write_artifacts(base_dir, counts, x_list, x_scale_list, w_list, w_scale_list, golden_list):
    rank_num = len(x_list)
    counts.astype(np.uint32).tofile(os.path.join(base_dir, "counts.bin"))
    for r in range(rank_num):
        input_dir = os.path.join(base_dir, "input", str(r))
        output_dir = os.path.join(base_dir, "output", str(r))
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        x_list[r].view(np.uint8).tofile(os.path.join(input_dir, "input_x.bin"))
        x_scale_list[r].view(np.uint8).tofile(os.path.join(input_dir, "input_x_scale.bin"))
        w_list[r].view(np.uint8).tofile(os.path.join(input_dir, "input_weight.bin"))
        w_scale_list[r].view(np.uint8).tofile(os.path.join(input_dir, "input_weight_scale.bin"))
        # counts 每个 rank 都放一份，launcher 各自读取
        counts.astype(np.uint32).tofile(os.path.join(input_dir, "counts.bin"))
        golden_list[r].view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def main():
    if len(sys.argv) < 7:
        print("Usage: python3 gen_data.py m k n rank_num expert_num trans_b [dtype] [seed]")
        print("  m: token rows per rank")
        print("  k: hidden size (multiple of 1024)")
        print("  n: expert output dim")
        print("  trans_b: 1 weight [E,N,K] / 0 weight [E,K,N]")
        print("  dtype: e4m3_e4m3 (default) | e5m2_e5m2 | e4m3_e5m2 | e5m2_e4m3")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    expert_num = int(sys.argv[5])
    trans_b = int(sys.argv[6])
    dtype = sys.argv[7] if len(sys.argv) > 7 else "e4m3_e4m3"
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 42

    if k % 1024 != 0:
        raise ValueError(f"k={k} must be multiple of 1024")
    a_dtype, b_dtype = dtype.split("_")

    counts, x_list, x_scale_list, w_list, w_scale_list, golden_list, recv_rows = gen_golden(
        m, k, n, rank_num, expert_num, trans_b == 1, seed, a_dtype, b_dtype)
    print(f"counts (per src rank, sum={m}):")
    print(counts)
    print(f"recv rows per rank: {recv_rows}")

    write_artifacts(os.getcwd(), counts, x_list, x_scale_list, w_list, w_scale_list, golden_list)


if __name__ == "__main__":
    main()
