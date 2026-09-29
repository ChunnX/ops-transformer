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

import os
import sys

import numpy as np
import torch


def write_artifacts(base_dir, rank_id, a_segment, b_ori, bias, out, a_after):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # a_local: (M_total, Ka) fp16，ND 行主序
    a_segment.tofile(os.path.join(input_dir, "input_a.bin"))
    # b_ori: (N, K) fp16。kernel TransB=true 用 DN 列主序读 [K,N]，
    # host 写 [N,K] C-order 即等价于 [K,N] 列主序，故直接写 b_ori 不转置。
    b_ori.tofile(os.path.join(input_dir, "input_b.bin"))
    # bias: (N,) float32
    bias.tofile(os.path.join(input_dir, "input_bias.bin"))
    out.numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))
    # 通信后 rank 窗口的期望内容（fp16）：
    #   窗口[r] = concat_s a_local[s][r*m:(r+1)*m, :]，即 a_after_alltoall_list[r]。
    #   注意：AllToAll PUT 跳过 self（self 贡献由 kernel 直接读 localAGmAddr），
    #   故窗口内 self 槽位保持 memset 的 0，期望文件需将 self 槽位置 0。
    #   用于单独验证通信：对比 main.cpp dump 的 comm_buffer.bin。
    a_after.tofile(os.path.join(output_dir, "expected_comm.bin"))


def gen_golden_data_all_to_all_matmul(m, k, n, rank_num):
    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")

    np.random.seed(42)

    # fp16 输入取 [0.5, 1)，避免 K 较大时 fp16 中间累加溢出
    a_local_list = [
        np.random.uniform(0.5, 1.0, (M_total, Ka)).astype(np.float16)
        for _ in range(rank_num)
    ]
    b_ori = np.random.uniform(0.5, 1.0, (N, K)).astype(np.float16)
    bias = np.random.uniform(-0.5, 0.5, (N,)).astype(np.float32)
    #bias = np.random.uniform(0.0, 0.0, (N,)).astype(np.float32)

    # all-to-all：dst 收集各 src 的 chunk
    #   a_after_alltoall[dst][src*m:(src+1)*m, :] = a_local[src][dst*m:(dst+1)*m, :]
    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after = np.zeros((M_total, Ka), dtype=np.float32)
        for src_rank in range(rank_num):
            chunk = a_local_list[src_rank][dst_rank * m:(dst_rank + 1) * m, :].astype(np.float32)
            a_after[src_rank * m:(src_rank + 1) * m, :] = chunk
        a_after_alltoall_list.append(a_after)

    b_dequant = b_ori.astype(np.float32)  # (N, K)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_after = a_after_alltoall_list[dst_rank]  # (M_total, Ka)
        a_permuted = np.zeros((m, K), dtype=np.float32)
        for src_rank in range(rank_num):
            a_permuted[:, src_rank * Ka:(src_rank + 1) * Ka] = a_after[src_rank * m:(src_rank + 1) * m, :]

        a_cpu = torch.from_numpy(a_permuted)   # (m, K)
        b_cpu = torch.from_numpy(b_dequant)    # (N, K)，逻辑 B = [K,N] 取转置
        out = torch.matmul(a_cpu, b_cpu.t())   # (m, N)
        out = out + torch.from_numpy(bias).float()
        out = out.to(torch.float16)
        out_all_ranks.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        # 期望窗口：self 槽位（rows [rank_id*m, (rank_id+1)*m)）置 0，与 PUT 跳过 self 的语义一致
        a_after_expected = a_after_alltoall_list[rank_id].astype(np.float16).copy()
        a_after_expected[rank_id * m:(rank_id + 1) * m, :] = 0
        write_artifacts(
            current_dir,
            rank_id,
            a_local_list[rank_id],
            b_ori,
            bias,
            out_all_ranks[rank_id],
            a_after_expected,
        )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py m k n rank_num")
        print("  m: matrix M dimension (per-rank output rows)")
        print("  k: matrix K dimension (total, must be divisible by rank_num)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])

    gen_golden_data_all_to_all_matmul(m, k, n, rank_num)
