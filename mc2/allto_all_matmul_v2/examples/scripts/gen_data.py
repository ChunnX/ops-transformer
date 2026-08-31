#!/usr/bin/python3
# coding=utf-8
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# ------------------------------------------------------------------------------
"""Generate MX FP8 input data + CPU golden for AlltoAllMatmulV2 aclnn test.

Adapted from mc2/common/op_kernel/apace/tests/st/all_to_all_quant_matmul/scripts/gen_data.py
for the aclnn direct invoke test.

Per-rank data layout (matching the kernel ST test):
  x1:        [M_total, Ka]    FP8_E4M3FN  — per-rank A where Ka = K_total / rank_num
  x2:        [N, K_total]     FP8_E4M3FN  — B matrix (shared across ranks)
  x1_scale:  [M_total, Ceil(Ka/64), 2]   FP8_E8M0   — MX scale for x1
  x2_scale:  [N * Ceil(K_total/64), 2] → flattened to [N * Ceil(K_total/64) * 2]

Golden computation: simulate All-to-All shuffle + dequant + matmul → BF16 output.
"""

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


def write_artifacts(
    base_dir, rank_id, a_local_fp8, b_fp8, a_scale, b_scale_flat, golden_out
):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_local_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale_flat.astype(np.uint8).tofile(os.path.join(input_dir, "input_scaleB.bin"))
    golden_out.view(torch.uint16).numpy().tofile(
        os.path.join(output_dir, "cpu_output.bin")
    )


def gen_golden_data(m, k, n, rank_num):
    """Generate per-rank input data and CPU golden output.

    Args:
        m: per-rank M dimension (total M = m * rank_num)
        k: total K dimension (per-rank Ka = k / rank_num)
        n: N dimension
        rank_num: number of ranks
    """
    m_total = m * rank_num
    k_total = k
    ka = k_total // rank_num

    if k_total % rank_num != 0:
        raise ValueError(f"K={k_total} must be divisible by rank_num={rank_num}")

    np.random.seed(42)

    # B matrix: [N, K_total] FP8
    b_ori = np.random.uniform(1, 8, (n, k_total)).astype(float8_e4m3fn)
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)

    # B scale: [N, Ceil(K/64), 2] → flatten as [N, Ceil(K/64)*2]
    b_scale_ceil_k = math.ceil(k_total / 64)
    b_scale_ori = np.random.uniform(1, 8, size=(n, b_scale_ceil_k, 2)).astype(
        float8_e8m0
    )

    # Dequant B: [K_total, N] float32 (TransB=true → K×N)
    b_scale_reshape = b_scale_ori.reshape(n, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :k_total]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(
        np.float32
    )

    # B scale flattened for binary write:
    # Kernel uses ScaleBDNLayoutPtn (column-major) → write as [N, K/32] C-order.
    # Use .view(np.uint8) to preserve raw E8M0 bytes (NOT .astype which does value conversion!)
    b_scale_flat = b_scale_ori.reshape(n, k_total // 32).flatten().view(np.uint8)

    # Per-rank A_local: [M_total, Ka] FP8 + scale
    a_local_list = []
    a_scale_list = []
    for _ in range(rank_num):
        a_local = np.random.uniform(1, 8, (m_total, ka)).astype(float8_e4m3fn)
        a_scale = np.random.uniform(1, 8, size=(m_total, math.ceil(ka / 64), 2)).astype(
            float8_e8m0
        )
        a_local_list.append(a_local)
        a_scale_list.append(a_scale)

    # Dequant per-rank A_local
    a_dequant_list = []
    for rank_id in range(rank_num):
        a_local = a_local_list[rank_id]
        a_scale = a_scale_list[rank_id]
        a_scale_reshape = a_scale.reshape(m_total, -1)
        a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :ka]
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast.astype(np.float32)
        a_dequant_list.append(a_dequant)

    # Simulate All-to-All: each rank collects the right chunks from all ranks
    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after = np.zeros((m_total, ka), dtype=np.float32)
        for src_rank in range(rank_num):
            chunk_row_start = dst_rank * m
            chunk_row_end = chunk_row_start + m
            chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]
            dst_row_start = src_rank * m
            dst_row_end = dst_row_start + m
            a_after[dst_row_start:dst_row_end, :] = chunk
        a_after_alltoall_list.append(a_after)

    # Matmul: after alltoall, each rank has full [m_total, k_total] → [m, k_total]
    golden_list = []
    for dst_rank in range(rank_num):
        a_permuted = np.zeros((m, k_total), dtype=np.float32)
        for src_rank in range(rank_num):
            src_row_start = src_rank * m
            src_row_end = src_row_start + m
            src_chunk = a_after_alltoall_list[dst_rank][src_row_start:src_row_end, :]
            dst_col_start = src_rank * ka
            dst_col_end = dst_col_start + ka
            a_permuted[:, dst_col_start:dst_col_end] = src_chunk

        a_cpu = torch.from_numpy(a_permuted)
        b_cpu = torch.from_numpy(b_dequant)
        out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)
        golden_list.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        write_artifacts(
            current_dir,
            rank_id,
            a_local_list[rank_id],
            b_ori,
            a_scale_list[rank_id],
            b_scale_flat,
            golden_list[rank_id],
        )

    print(
        f"[GEN] Generated data: M={m} K={k} N={n} rank={rank_num} (M_total={m_total}, Ka={ka})"
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py m k n rank_num")
        print("  m: per-rank M (total M = m * rank_num)")
        print("  k: total K (per-rank Ka = k / rank_num)")
        print("  n: N dimension")
        print("  rank_num: number of ranks")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    gen_golden_data(m, k, n, rank_num)
