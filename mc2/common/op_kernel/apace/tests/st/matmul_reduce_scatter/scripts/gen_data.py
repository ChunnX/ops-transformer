#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# quantMatmulReduceScatter golden 数据生成
#
# 计算流程：
#   1. 每卡独立计算 C_i = dequant(A_i) × dequant(B)  → [m, n] bfloat16
#   2. AlltoAll 按 M 轴切分 C_i：chunk j (m/rankSize 行) 分发到 rank j
#   3. reduceSum: output_r = Σ_{j=0}^{R-1} C_j[chunk r]  → [m/rankSize, n]
#
# 与 all_to_all_quant_matmul 的区别：
#   - K 不切分（每卡完整 K），M 按 rankSize 切分
#   - 每卡 A 不同，B 相同
#   - 输出 per rank: [m/rankSize, n]
# ----------------------------------------------------------------------------------------------------------

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


def write_artifacts(base_dir, rank_id, a_fp8, b_fp8, a_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_quant_matmul_reduce_scatter(m, k, n, rank_num):
    """
    生成 quantMatmulReduceScatter 的 golden 数据。

    参数:
      m: 矩阵 A 的行数（必须被 rank_num 整除）
      k: 矩阵 A 的列数（完整 K，不切分）
      n: 矩阵 B 的列数
      rank_num: rank 数量
    """
    if m % rank_num != 0:
        raise ValueError(f"m={m} must be divisible by rank_num={rank_num}")

    chunk_m = m // rank_num  # 每个 rank 的 M 分片大小

    np.random.seed(42)

    # B 矩阵：所有 rank 共享相同的 B [n, k] (TransB=true, 列主序)
    b_ori = np.random.uniform(1, 8, (n, k)).astype(float8_e4m3fn)

    # B Scale: [n, ceil(k/64), 2] → [n, K/32] flatten
    b_scale_ori = np.random.uniform(1, 8, size=(n, math.ceil(k / 64), 2)).astype(
        float8_e8m0
    )
    b_scale_all = b_scale_ori.reshape(n, math.ceil(k / 64) * 2).flatten()

    # 反量化 B: [n, k] → transpose → [k, n]
    # MXFP8 scale: [n, ceil(K/64), 2] → 每组 2 个 scale 各覆盖 32 个元素
    # reshape 到 [n, ceil(K/64), 2, 1] → repeat 32 → [n, ceil(K/64), 2, 32] → reshape [n, ceil(K/64)*64] → 截断到 K
    k_padded = math.ceil(k / 64) * 64
    b_scale_broadcast = b_scale_ori.reshape(n, math.ceil(k / 64), 2, 1)
    b_scale_broadcast = np.repeat(
        b_scale_broadcast, 32, axis=-1
    )  # [n, ceil(K/64), 2, 32]
    b_scale_broadcast = b_scale_broadcast.reshape(n, k_padded)[..., :k]  # [n, k]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)  # [k, n]
    b_dequant = b_ori.astype(np.float32).T * b_scale_broadcast_transpose  # [k, n]

    # 每卡独立的 A 矩阵
    a_list = []
    a_scale_list = []
    c_list = []

    for rank_id in range(rank_num):
        # A [m, k] fp8_e4m3fn
        a_local = np.random.uniform(1, 8, (m, k)).astype(float8_e4m3fn)
        # ScaleA [m, ceil(k/64), 2] fp8_e8m0
        a_scale = np.random.uniform(1, 8, size=(m, math.ceil(k / 64), 2)).astype(
            float8_e8m0
        )

        a_list.append(a_local)
        a_scale_list.append(a_scale)

        # 反量化 A
        # MXFP8 scale: [m, ceil(K/64), 2] → 每组 2 个 scale 各覆盖 32 个元素
        a_scale_broadcast = a_scale.reshape(m, math.ceil(k / 64), 2, 1)
        a_scale_broadcast = np.repeat(
            a_scale_broadcast, 32, axis=-1
        )  # [m, ceil(K/64), 2, 32]
        a_scale_broadcast = a_scale_broadcast.reshape(m, k_padded)[..., :k]  # [m, k]
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast  # [m, k]

        # C_i = A_i × B  → [m, n]
        a_cpu = torch.from_numpy(a_dequant)
        b_cpu = torch.from_numpy(b_dequant)
        c_i = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)  # [m, n]
        c_list.append(c_i)

    # AlltoAll + reduceSum
    # AlltoAll: 每卡将 C_i 按 M 轴切分，chunk j 发送到 rank j
    # reduceSum: rank r 收到所有卡的 chunk r，逐元素求和
    # output_r = Σ_{j=0}^{R-1} C_j[rows r*chunk_m : (r+1)*chunk_m, :]
    # 注意: 累加在 float32 中进行, 最后再转 bfloat16, 减少 bf16 累加误差
    for dst_rank in range(rank_num):
        output = torch.zeros((chunk_m, n), dtype=torch.float32)
        for src_rank in range(rank_num):
            row_start = dst_rank * chunk_m
            row_end = row_start + chunk_m
            chunk = c_list[src_rank][row_start:row_end, :].float()
            output += chunk

        output = output.to(torch.bfloat16)

        # 写入文件
        current_dir = os.getcwd()
        write_artifacts(
            current_dir,
            dst_rank,
            a_list[dst_rank],
            b_ori,
            a_scale_list[dst_rank],
            b_scale_all,
            output,
        )

    print(f"Generated golden data for m={m}, k={k}, n={n}, rank_num={rank_num}")
    print(f"  chunk_m={chunk_m}, output per rank: [{chunk_m}, {n}]")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py m k n rank_num")
        print("  m: matrix M dimension (must be divisible by rank_num)")
        print("  k: matrix K dimension (full K, NOT split across ranks)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])

    gen_golden_data_quant_matmul_reduce_scatter(m, k, n, rank_num)
