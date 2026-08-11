#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
ReduceScatter via AllToAll PUT + AtomicAdd 的 golden 数据生成。

ReduceScatter 语义（DESIGN.md §1.1）：
  输入:
    rank i 持有:
      A_i  - shape [M, K_local],  dtype = fp8_e4m3fn_t   (K_local = K / rankSize)
      ScaleA_i - shape [M, CeilDiv(K_local,64)*2], dtype = fp8_e8m0_t  (MXFP8 scale)
      B_full - shape [K, N],      dtype = fp8_e4m3fn_t   (全量复制, K = rankSize × K_local)
      ScaleB_full - shape [CeilDiv(K,64)*2, N], dtype = fp8_e8m0_t    (全量复制)
  输出:
    rank r 获得:
      C_r - shape [M/rankSize, N], dtype = bfloat16_t     (ReduceScatter 输出)

  C_r = Σ_{i=0}^{rankSize-1} dequant(A_i[r*M/R : (r+1)*M/R, :]) × dequant(B_i[:, :])
"""

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn, float8_e5m2


# dtype 字符串 -> numpy dtype 映射
_DTYPE_MAP = {
    "e4m3": float8_e4m3fn,
    "e5m2": float8_e5m2,
}


def parse_dtype(dtype_str):
    """解析 dtype 字符串 'e4m3e4m3' -> ('e4m3', 'e4m3')"""
    if len(dtype_str) != 8:
        raise ValueError(f"Invalid dtype string: {dtype_str}, expected format like 'e4m3e4m3'")
    a_dtype_str = dtype_str[:4]
    b_dtype_str = dtype_str[4:]
    if a_dtype_str not in _DTYPE_MAP or b_dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Invalid dtype string: {dtype_str}, supported: e4m3, e5m2")
    return a_dtype_str, b_dtype_str


def write_artifacts(base_dir, rank_id, a_segment_fp8, b_fp8, a_segment_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_segment_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_segment_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_reduce_scatter_matmul(m, k, n, rank_num, dtype_str="e4m3e4m3"):
    """
    ReduceScatter golden 计算:
      C_r[M/R × N] = Σ_{i=0}^{R-1} dequant(A_i[r*M/R:(r+1)*M/R, :]) × dequant(B_i[:, :])

    数据布局:
      - A_i: [M × Ka] per rank（K-split, Ka = K/rank_num）
      - B_full: [K × N] 全量复制（K = rank_num × Ka）
      - C_r: [M/rank_num × N] per rank（M-split 输出）

    dtype_str: 'e4m3e4m3' | 'e5m2e5m2' | 'e4m3e5m2' | 'e5m2e4m3'
    """
    M = m
    K = k
    N = n
    Ka = K // rank_num
    m_local = M // rank_num  # ReduceScatter per-rank 输出行数

    a_dtype_str, b_dtype_str = parse_dtype(dtype_str)
    a_np_dtype = _DTYPE_MAP[a_dtype_str]
    b_np_dtype = _DTYPE_MAP[b_dtype_str]

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")
    if M % rank_num != 0:
        raise ValueError(f"M={M} must be divisible by rank_num={rank_num}")
    if Ka % 64 != 0:
        raise ValueError(f"Ka={Ka} must be divisible by 64 (MXFP8 scale alignment)")

    np.random.seed(42)

    # B: [N, K] C-order (= [K, N] 列主序，TransB=true 时 kernel 用 DNExtLayoutPtn 按列主序读)
    b_ori = np.random.uniform(1, 8, (N, K)).astype(b_np_dtype)
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)  # 逻辑 B = [K, N]

    # ScaleB: [N, CeilDiv(K,64), 2] -> [N, K/32] C-order (= [K/32, N] 列主序)
    b_scale_ori = np.random.uniform(1, 8, size=(N, math.ceil(K / 64), 2)).astype(float8_e8m0)
    b_scale_reshape = b_scale_ori.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :K]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)
    # b_dequant shape: [K, N]

    # TransB=true 时 kernel 用 ScaleBDNLayoutPtn (column-major) 访问 scaleB
    b_scale_all = b_scale_ori.reshape(N, K // 32).flatten()

    # 每 rank 的 A_i 和 ScaleA_i
    a_local_list = []
    a_scale_list = []
    a_dequant_list = []

    for rank_id in range(rank_num):
        # A_i: [M × Ka]
        a_local = np.random.uniform(1, 8, (M, Ka)).astype(a_np_dtype)
        # ScaleA_i: [M, CeilDiv(Ka,64), 2]
        a_scale = np.random.uniform(1, 8, size=(M, math.ceil(Ka / 64), 2)).astype(float8_e8m0)

        a_local_list.append(a_local)
        a_scale_list.append(a_scale)

        # 反量化 A_i
        a_scale_reshape = a_scale.reshape(M, -1)
        a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :Ka]
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast.astype(np.float32)
        a_dequant_list.append(a_dequant)
        # a_dequant shape: [M, Ka]

    # ReduceScatter golden: C_r[M/R × N] = Σ_i A_i[r*M/R:(r+1)*M/R, :] × B_i[Ka × N]
    # B_i 是 B 的第 i 个 Ka 段: B_i = b_dequant[i*Ka:(i+1)*Ka, :]
    out_all_ranks = []
    for dst_rank in range(rank_num):
        # dst_rank 获得的 M 段: [r*M/R : (r+1)*M/R]
        m_start = dst_rank * m_local
        m_end = m_start + m_local

        # C_r = Σ_i dequant(A_i[m_start:m_end, :]) × dequant(B_i[:, :])
        c_sum = np.zeros((m_local, N), dtype=np.float32)
        for src_rank in range(rank_num):
            # A_i 的 M 段
            a_seg = a_dequant_list[src_rank][m_start:m_end, :]  # [m_local, Ka]
            # B_i = b_dequant 的第 i 个 Ka 段
            b_seg = b_dequant[src_rank * Ka:(src_rank + 1) * Ka, :]  # [Ka, N]
            c_sum += np.matmul(a_seg, b_seg)

        out = torch.from_numpy(c_sum).to(torch.bfloat16)
        out_all_ranks.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        write_artifacts(current_dir, rank_id, a_local_list[rank_id], b_ori,
                       a_scale_list[rank_id], b_scale_all, out_all_ranks[rank_id])


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage: python3 gen_data.py m k n rank_num [dtype]")
        print("  m: matrix M dimension (total, must be divisible by rank_num)")
        print("  k: matrix K dimension (total, must be divisible by rank_num)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        print("  dtype: optional, FP8 dtype combination (default: e4m3e4m3)")
        print("    supported: e4m3e4m3, e5m2e5m2, e4m3e5m2, e5m2e4m3")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    dtype_str = sys.argv[5] if len(sys.argv) > 5 else "e4m3e4m3"

    gen_golden_data_reduce_scatter_matmul(m, k, n, rank_num, dtype_str)
