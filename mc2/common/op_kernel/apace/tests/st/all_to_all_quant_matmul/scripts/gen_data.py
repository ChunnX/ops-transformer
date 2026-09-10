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

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn


def write_artifacts(
    base_dir, rank_id, a_segment_fp8, b_fp8, a_segment_scale, b_scale, out
):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_segment_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_segment_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


# ---------------------- HiF4 (hifloat4 + hif4_scale) ----------------------
# S1P2(E2M1) 4-bit 元素码值: 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit3 为符号
HIF4_S1P2_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
HIF4_GROUP_SIZE = 64


def gen_hif4_packed(rows, cols, rng):
    """生成 S1P2 2:1 打包数据。

    打包规则(与 fp4x2_e2m1_t 一致): 偶数下标元素占低 4 bit, 奇数下标占高 4 bit。
    返回 (packed uint8[rows, cols//2], 数值 float32[rows, cols])。
    """
    idx = rng.integers(0, 8, size=(rows, cols))
    sign = rng.integers(0, 2, size=(rows, cols))
    vals = np.where(sign == 0, 1.0, -1.0) * np.take(HIF4_S1P2_VALUES, idx)
    codes = (idx.astype(np.uint8) | (sign.astype(np.uint8) << 3))
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    packed = (even | (odd << 4)).astype(np.uint8)
    return packed, vals.astype(np.float32)


def gen_hif4_meta(rows, groups, rng):
    """生成 hif4_scale 元数据: 每 64 元素组 1 个 uint32。

    字节布局(little-endian):
      byte0 = E6M2 全局 scale (fp8 e6m2: sign1 + exp6 + man2)
      byte1 = E1_8  (8 bit, bit[i/8] 对应组内第 8i..8i+7 个元素)
      byte2-3 = E1_16 (16 bit, bit[i/4] 对应组内第 4i..4i+3 个元素)
    E6M2 指数取 [30, 32](数值 0.5~8), 保证 golden 数值范围适中。
    返回 (meta uint32[rows, groups], e6m2/e1_8/e1_16 各分量)。
    """
    e6m2_exp = rng.integers(30, 33, size=(rows, groups)).astype(np.uint8)
    e6m2_man = rng.integers(0, 4, size=(rows, groups)).astype(np.uint8)
    e6m2 = ((e6m2_exp << 2) | e6m2_man).astype(np.uint8)  # 符号位恒 0
    e1_8 = rng.integers(0, 2, size=(rows, groups)).astype(np.uint8)
    # E1_16 每 bit 以概率 1/4 置位, 控制动态范围
    e1_16_bits = rng.random((rows, groups, 16)) < 0.25
    e1_16 = (e1_16_bits * (1 << np.arange(16))).sum(axis=2).astype(np.uint16)
    meta = (
        e6m2.astype(np.uint32)
        | (e1_8.astype(np.uint32) << 8)
        | (e1_16.astype(np.uint32) << 16)
    )
    return meta, e6m2, e1_8.astype(np.uint8), e1_16


def hif4_dequant(vals, meta, rows, cols):
    """HiF4 反量化: V = E6M2 * 2^(E1_8[i/8] + E1_16[i/4]) * S1P2[i]。

    vals: float32[rows, cols]; meta: uint32[rows, cols//64]。
    """
    groups = cols // HIF4_GROUP_SIZE
    meta_bytes = meta.view(np.uint8).reshape(rows, groups, 4)
    e6m2 = meta_bytes[:, :, 0].astype(np.uint8)
    e1_8 = meta_bytes[:, :, 1].astype(np.uint8)
    e1_16_lo = meta_bytes[:, :, 2].astype(np.uint8)
    e1_16_hi = meta_bytes[:, :, 3].astype(np.uint8)
    e1_16 = (e1_16_hi.astype(np.uint16) << 8) | e1_16_lo.astype(np.uint16)

    # E6M2 解码: 2^((E-31)) * (1 + M/4), 符号位恒 0
    exp_man = (e6m2 & 0x7F).astype(np.float32)
    e6m2_val = np.power(2.0, (np.floor(exp_man / 4.0) - 31.0)) * (1.0 + np.mod(exp_man, 4.0) / 4.0)

    # 组内第 i 个元素: E1_8 bit (i//8), E1_16 bit (i//4)
    g = np.arange(HIF4_GROUP_SIZE)
    e18_bit = ((e1_8[:, :, None] >> g[None, None, :] // 8) & 1).astype(np.float32)
    e116_bit = ((e1_16[:, :, None] >> g[None, None, :] // 4) & 1).astype(np.float32)
    expo = e18_bit + e116_bit  # [rows, groups, 64]
    scale = (e6m2_val[:, :, None] * np.power(2.0, expo)).astype(np.float32)
    scale_elem = scale.reshape(rows, cols)
    return vals * scale_elem


def write_artifacts_hif4(base_dir, rank_id, a_packed, b_packed, a_meta, b_meta, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_packed.tofile(os.path.join(input_dir, "input_a.bin"))
    b_packed.tofile(os.path.join(input_dir, "input_b.bin"))
    a_meta.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_meta.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_all_to_all_matmul_hif4(m, k, n, rank_num, trans_b=True):
    """HiF4 全量化 AllToAllMatmul golden 生成。

    布局约定(与 main_hif4.cpp 读取侧一致):
      A: [M_total, Ka] S1P2 打包行主序, scaleA [M_total, Ka/64] uint32 行主序
      B(trans_b=True):  [N, K] S1P2 打包行主序(= DN 列主序 [K, N]), 与 MX 一致;
                        scaleB [N, K/64] uint32 行主序(= [K/64, N] 列主序)
      B(trans_b=False): [K, N] S1P2 打包行主序(ND), scaleB [K/64, N] uint32 行主序
    """
    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")
    # Ka 按 128 对齐统一校验(同时保证 Ka%64==0 且 scale 组数为偶数, 与 MX 约束统一)
    if Ka % 128 != 0:
        raise ValueError(f"Ka={Ka} must be 128-aligned (Ka % 128 == 0)")

    np.random.seed(42)
    rng = np.random.default_rng(42)

    # B: 逻辑 [K, N], 生成 [N, K] 再由 golden 转置
    b_packed_nk, b_vals = gen_hif4_packed(N, K, rng)  # [N, K/2] bytes
    b_meta_nk, _, _, _ = gen_hif4_meta(N, K // HIF4_GROUP_SIZE, rng)  # [N, K/64]
    b_dequant = hif4_dequant(b_vals, b_meta_nk, N, K)  # [N, K]
    b_dequant = np.swapaxes(b_dequant, -1, -2)  # 逻辑 [K, N]

    if trans_b:
        # DN 布局: 直接写 [N, K] 行主序(= [K, N] 列主序), 与 MX 修正一致
        b_packed_write = b_packed_nk
        b_meta_write = b_meta_nk.reshape(N, -1)
    else:
        # ND 布局: [K, N] 行主序, 由逻辑数值重新打包
        b_vals_kn = np.swapaxes(b_vals, -1, -2)
        idx = np.searchsorted(HIF4_S1P2_VALUES, np.abs(b_vals_kn))
        sign = (b_vals_kn < 0).astype(np.uint8)
        codes = (idx.astype(np.uint8) | (sign << 3))
        b_packed_write = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        # scaleB ND: [K/64, N] 行主序 = b_meta_nk 转置展平
        b_meta_write = b_meta_nk.T.copy().reshape(-1)

    a_packed_list = []
    a_meta_list = []
    a_dequant_list = []
    for _ in range(rank_num):
        a_packed, a_vals = gen_hif4_packed(M_total, Ka, rng)  # [M_total, Ka/2]
        a_meta, _, _, _ = gen_hif4_meta(M_total, Ka // HIF4_GROUP_SIZE, rng)
        a_packed_list.append(a_packed)
        a_meta_list.append(a_meta)
        a_dequant_list.append(hif4_dequant(a_vals, a_meta, M_total, Ka))

    # AllToAll 模拟: M 轴 scatter, K 轴 gather(与 MX 版本一致)
    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after_alltoall = np.zeros((M_total, Ka), dtype=np.float32)
        for src_rank in range(rank_num):
            chunk_row_start = dst_rank * m
            chunk_row_end = chunk_row_start + m
            chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]
            dst_row_start = src_rank * m
            dst_row_end = dst_row_start + m
            a_after_alltoall[dst_row_start:dst_row_end, :] = chunk
        a_after_alltoall_list.append(a_after_alltoall)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_permuted = np.zeros((m, K), dtype=np.float32)
        for src_rank in range(rank_num):
            src_row_start = src_rank * m
            src_row_end = src_row_start + m
            src_chunk = a_after_alltoall_list[dst_rank][src_row_start:src_row_end, :]
            dst_col_start = src_rank * Ka
            dst_col_end = dst_col_start + Ka
            a_permuted[:, dst_col_start:dst_col_end] = src_chunk

        a_cpu = torch.from_numpy(a_permuted)
        b_cpu = torch.from_numpy(b_dequant)
        out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)
        out_all_ranks.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        write_artifacts_hif4(
            current_dir,
            rank_id,
            a_packed_list[rank_id],
            b_packed_write,
            a_meta_list[rank_id].reshape(M_total, -1),
            b_meta_write.reshape(-1),
            out_all_ranks[rank_id],
        )


def gen_golden_data_all_to_all_matmul(m, k, n, rank_num):
    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")

    # [NOTE] 2026-06-22: fixed seed 方便与 sanity_check.py 对比 shmem_A.bin。
    np.random.seed(42)

    b_ori = np.random.uniform(1, 8, (N, K)).astype(float8_e4m3fn)
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)

    b_scale_ori = np.random.uniform(1, 8, size=(N, math.ceil(K / 64), 2)).astype(
        float8_e8m0
    )

    b_scale_reshape = b_scale_ori.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :K]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(
        np.float32
    )

    # [FIX] 2026-06-22: TransB=true 时 kernel 用 ScaleBDNLayoutPtn (column-major)
    # 访问 scaleB，layoutScaleB = (rankDim*kScaleSize, N) = (K/32, N) 按列主序读。
    # host 必须写成 [N, K/32] C-order (= [K/32, N] 列主序) 才能被正确解读。
    # 之前 .transpose(1, 0, 2).reshape(K//32, N) 写的是行主序，被 kernel 按列主序
    # 读成乱序字节，导致最终输出与 golden rel_err 288% 不相关。
    b_scale_all = b_scale_ori.reshape(N, K // 32).flatten()

    a_local_list = []
    a_scale_list = []

    for rank_id in range(rank_num):
        a_local = np.random.uniform(1, 8, (M_total, Ka)).astype(float8_e4m3fn)
        a_scale = np.random.uniform(1, 8, size=(M_total, math.ceil(Ka / 64), 2)).astype(
            float8_e8m0
        )

        a_local_list.append(a_local)
        a_scale_list.append(a_scale)

    a_dequant_list = []
    for rank_id in range(rank_num):
        a_local = a_local_list[rank_id]
        a_scale = a_scale_list[rank_id]
        a_scale_reshape = a_scale.reshape(M_total, -1)
        a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :Ka]
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast.astype(np.float32)
        a_dequant_list.append(a_dequant)

    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after_alltoall = np.zeros((M_total, Ka), dtype=np.float32)
        for src_rank in range(rank_num):
            chunk_row_start = dst_rank * m
            chunk_row_end = chunk_row_start + m
            chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]

            dst_row_start = src_rank * m
            dst_row_end = dst_row_start + m
            a_after_alltoall[dst_row_start:dst_row_end, :] = chunk
        a_after_alltoall_list.append(a_after_alltoall)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_permuted = np.zeros((m, K), dtype=np.float32)
        for src_rank in range(rank_num):
            src_row_start = src_rank * m
            src_row_end = src_row_start + m
            src_chunk = a_after_alltoall_list[dst_rank][src_row_start:src_row_end, :]

            dst_col_start = src_rank * Ka
            dst_col_end = dst_col_start + Ka
            a_permuted[:, dst_col_start:dst_col_end] = src_chunk

        a_cpu = torch.from_numpy(a_permuted)
        b_cpu = torch.from_numpy(b_dequant)
        out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)
        out_all_ranks.append(out)

    # [FIX] 2026-06-22: TransB=true 时 kernel 用 DNExtLayoutPtn (column-major)
    # 访问 B，layoutB = (rankDim*Ka, N) = (K, N) 按列主序读。
    # host 必须写成 [N, K] C-order (= [K, N] 列主序)，即直接写 b_ori 不转置。
    # 之前传 b_ori_transpose 写的是 [K, N] 行主序，被 kernel 按列主序读成乱序字节。
    # golden 计算里仍用 b_ori_transpose (逻辑 B = [K, N])，不动。
    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        write_artifacts(
            current_dir,
            rank_id,
            a_local_list[rank_id],
            b_ori,
            a_scale_list[rank_id],
            b_scale_all,
            out_all_ranks[rank_id],
        )


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print("Usage: python3 gen_data.py m k n rank_num [quant_type] [trans_b]")
        print("  m: matrix M dimension (per-rank output M)")
        print("  k: matrix K dimension (total, must be divisible by rank_num)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        print("  quant_type: 'mxfp8' (default) | 'hif4'")
        print("  trans_b: hif4 only, 'true' (default) | 'false'")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    quant_type = sys.argv[5] if len(sys.argv) >= 6 else "mxfp8"
    trans_b = True
    if len(sys.argv) >= 7:
        if sys.argv[6] not in ("true", "false", "1", "0"):
            print("ERROR: trans_b must be 'true' or 'false'")
            sys.exit(1)
        trans_b = sys.argv[6] in ("true", "1")

    if quant_type == "hif4":
        gen_golden_data_all_to_all_matmul_hif4(m, k, n, rank_num, trans_b)
    elif quant_type == "mxfp8":
        gen_golden_data_all_to_all_matmul(m, k, n, rank_num)
    else:
        print(f"ERROR: unknown quant_type '{quant_type}' (mxfp8 | hif4)")
        sys.exit(1)
