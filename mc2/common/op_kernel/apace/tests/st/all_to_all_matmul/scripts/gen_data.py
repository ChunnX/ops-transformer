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


def _target_dtype(dtype):
    return torch.bfloat16 if dtype == "bf16" else torch.float16


def write_tensor(t, path):
    """按 2 字节原始位写入 fp16/bf16 张量（bf16 无 numpy 类型，经 uint16 视图写）。"""
    if t.dtype == torch.bfloat16:
        t.contiguous().view(torch.uint16).numpy().tofile(path)
    else:
        t.numpy().tofile(path)


def write_artifacts(base_dir, rank_id, a_segment, b_ori, bias, out, a_after):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # a_local: (M_total, Ka) fp16/bf16，ND 行主序
    write_tensor(a_segment, os.path.join(input_dir, "input_a.bin"))
    # b_ori: (N, K) fp16/bf16。kernel TransB=true 用 DN 列主序读 [K,N]，
    # host 写 [N,K] C-order 即等价于 [K,N] 列主序，故直接写 b_ori 不转置。
    write_tensor(b_ori, os.path.join(input_dir, "input_b.bin"))
    # bias: (N,) float32
    bias.numpy().tofile(os.path.join(input_dir, "input_bias.bin"))
    write_tensor(out, os.path.join(output_dir, "cpu_output.bin"))
    # 通信后 rank 窗口的期望内容（fp16/bf16）：
    #   窗口[r] = concat_s a_local[s][r*m:(r+1)*m, :]，即 a_after_alltoall_list[r]。
    #   注意：AllToAll PUT 跳过 self（self 贡献由 kernel 直接读 localAGmAddr），
    #   故窗口内 self 槽位保持 memset 的 0，期望文件需将 self 槽位置 0。
    #   用于单独验证通信：对比 main.cpp dump 的 comm_buffer.bin。
    write_tensor(a_after, os.path.join(output_dir, "expected_comm.bin"))


def gen_golden_data_all_to_all_matmul(m, k, n, rank_num, dtype="fp16", lm=0):
    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num
    target = _target_dtype(dtype)

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")

    #np.random.seed(42)
    np.random.seed(274)

    # fp16/bf16 输入取 [0.5, 1)，避免 K 较大时中间累加溢出
    a_local_f32 = [
        #np.random.uniform(0.5, 1.0, (M_total, Ka)).astype(np.float32)
        #np.random.uniform(-5.0, 5.0, (M_total, Ka)).astype(np.float32)
        np.random.uniform(0, 10.0, (M_total, Ka)).astype(np.float32)
        for _ in range(rank_num)
    ]
    #b_ori_f32 = np.random.uniform(0.5, 1.0, (N, K)).astype(np.float32)
    b_ori_f32 = np.random.uniform(-3.0, 3.0, (N, K)).astype(np.float32)
    #b_ori_f32 = np.random.uniform(0, 10.0, (N, K)).astype(np.float32)
    #bias_f32 = np.random.uniform(-0.5, 0.5, (N,)).astype(np.float32)
    bias_f32 = np.random.uniform(-10.0, 0.0, (N,)).astype(np.float32)
    #bias_f32 = np.random.uniform(0.0, 10.0, (N,)).astype(np.float32)

    # 量化到目标 dtype（kernel 输入与 golden 必须用同一份量化值）
    a_local_list = [torch.from_numpy(x).to(target) for x in a_local_f32]
    b_ori = torch.from_numpy(b_ori_f32).to(target)  # (N, K)
    bias = torch.from_numpy(bias_f32)  # (N,) float32

    # all-to-all：dst 收集各 src 的 chunk
    #   a_after_alltoall[dst][src*m:(src+1)*m, :] = a_local[src][dst*m:(dst+1)*m, :]
    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after = torch.zeros(M_total, Ka, dtype=torch.float32)
        for src_rank in range(rank_num):
            chunk = a_local_list[src_rank][dst_rank * m:(dst_rank + 1) * m, :].float()
            a_after[src_rank * m:(src_rank + 1) * m, :] = chunk
        a_after_alltoall_list.append(a_after)

    b_dequant = b_ori.float()  # (N, K)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_after = a_after_alltoall_list[dst_rank]  # (M_total, Ka) fp32
        a_permuted = torch.zeros(m, K, dtype=torch.float32)
        for src_rank in range(rank_num):
            a_permuted[:, src_rank * Ka:(src_rank + 1) * Ka] = a_after[
                src_rank * m:(src_rank + 1) * m, :
            ]

        a_cpu = a_permuted  # (m, K)
        b_cpu = b_dequant  # (N, K)，逻辑 B = [K,N] 取转置
        out = torch.matmul(a_cpu, b_cpu.t())  # (m, N)
        out = out + bias

        if lm == 2:
            # lm=2（GM累加）：golden 按 kernel 实际算法模拟，用于验证 kernel 机制正确性。
            # kernel：LOCAL 在 L0C fp32 算 self部分和+bias → 取整成输出 dtype 写 GM；
            # REMOTE 把其余 rank 部分和在 fp32 累加 → 原子加（取整）到 GM。
            # 故：self_q=round(self+bias); remote_q=round(Σ_remote); out=round(self_q+remote_q)。
            self_partial = torch.zeros(m, N, dtype=torch.float32)
            for src_rank in range(rank_num):
                if src_rank != dst_rank:
                    continue
                seg_a = a_after[src_rank * m:(src_rank + 1) * m, :]
                seg_b = b_dequant[:, src_rank * Ka:(src_rank + 1) * Ka]
                self_partial = self_partial + torch.matmul(seg_a, seg_b.t())

            remote_partial = torch.zeros(m, N, dtype=torch.float32)
            for src_rank in range(rank_num):
                if src_rank == dst_rank:
                    continue
                seg_a = a_after[src_rank * m:(src_rank + 1) * m, :]
                seg_b = b_dequant[:, src_rank * Ka:(src_rank + 1) * Ka]
                remote_partial = remote_partial + torch.matmul(seg_a, seg_b.t())

            self_q = (self_partial + bias).to(target)
            remote_q = remote_partial.to(target)
            # 显式 .to(target)：保证输出为 fp16/bf16 写盘（避免部分 torch 版本 half+half 返回 fp32）
            out = (self_q + remote_q).to(target)
        else:
            out = out.to(target)

        out_all_ranks.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        # 期望窗口：self 槽位（rows [rank_id*m, (rank_id+1)*m)）置 0，与 PUT 跳过 self 的语义一致
        a_after_expected = a_after_alltoall_list[rank_id].to(target).clone()
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
    if len(sys.argv) not in (5, 6, 7):
        print("Usage: python3 gen_data.py m k n rank_num [dtype] [lm]")
        print("  m: matrix M dimension (per-rank output rows)")
        print("  k: matrix K dimension (total, must be divisible by rank_num)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        print("  dtype: optional, fp16 (default) | bf16")
        print("  lm: optional, localMatmul mode (0 default | 2: golden 模拟 GM 累加取整)")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    dtype = sys.argv[5] if len(sys.argv) >= 6 else "fp16"
    if dtype not in ("fp16", "bf16"):
        print(f"ERROR: dtype must be 'fp16' or 'bf16', got {dtype}")
        sys.exit(1)
    lm = int(sys.argv[6]) if len(sys.argv) == 7 else 0
    if lm not in (0, 1, 2):
        print(f"ERROR: lm must be 0/1/2, got {lm}")
        sys.exit(1)

    gen_golden_data_all_to_all_matmul(m, k, n, rank_num, dtype, lm)
