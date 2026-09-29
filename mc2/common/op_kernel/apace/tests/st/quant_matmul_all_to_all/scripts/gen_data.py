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

"""
Generate MXFP test data for QuantMatmulAllToAll operator.

Inputs (per rank):
  - x1 (A): quantized, ND layout, shape [M, K]
      quant_type: 0=FP4 E2M1 (packed 2/byte), 1=FP8 E4M3, 2=FP8 E5M2
  - x2 (B): quantized, DN layout (col-major ND), shape [K, N] (full B per rank)
  - x1Scale: fp8e8m0_t scale for A, shape [M, ceil(K/64), 2]
  - x2Scale: fp8e8m0_t scale for B, shape [ceil(K/64), 2, N]
  - bias: float32 [N] (only when is_bias=1), amplitude auto-scaled to ~1/8 of the
    typical matmul output magnitude so it stays visible in BF16/FP16 outputs

Output (per rank):
  - output: bfloat16 or float16 (is_fp16), ND layout, shape [rankNum*M, N/rankNum]

Data flow:
  1. Matmul:   C_i = dequant(A_i) x dequant(B_i) (+ bias) -> [M, N]
  2. Permute:  C_i.view(M, rankNum, Np).permute(1, 0, 2) -> [rankNum, M, Np]
  3. AllToAll: rank j receives C_i[:, j*Np:(j+1)*Np] from every rank i,
     concatenated along M -> [rankNum*M, Np]

Usage:
  python3 gen_data.py m k n rank_num [quant_type] [is_fp16] [is_bias]
    quant_type: 0=E2M1E2M1(FP4), 1=E4M3E4M3, 2=E5M2E5M2, 3=E4M3E5M2, 4=E5M2E4M3 (default: 1)
    is_fp16:    output float16 instead of bfloat16 (default: 0)
    is_bias:    add bias input [N] (default: 0)
"""

import math
import os
import sys

import numpy as np
import torch

BASE_SEED = 42

QUANT_TYPES = {
    0: ("e2m1", "e2m1"),
    1: ("e4m3", "e4m3"),
    2: ("e5m2", "e5m2"),
    3: ("e4m3", "e5m2"),
    4: ("e5m2", "e4m3"),
}

DATA_RANGE = {"e2m1": (1.0, 6.0), "e4m3": (1.0, 8.0), "e5m2": (1.0, 8.0)}

FP4_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def write_artifacts(base_dir, rank_id, a_file, b_file, a_scale, b_scale, bias, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # A: ND layout (row-major), store as-is (FP4 already packed along K)
    a_file.tofile(os.path.join(input_dir, "input_a.bin"))
    # B: DN layout (col-major), already transposed to [N, K] (FP4 packed along K)
    b_file.tofile(os.path.join(input_dir, "input_b.bin"))
    # ScaleA: ND layout
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    # ScaleB: [ceil(K/64), 2, N] -> store as [N, ceil(K/64), 2] for DN layout
    np.ascontiguousarray(b_scale.transpose(2, 0, 1)).tofile(
        os.path.join(input_dir, "input_scaleB.bin")
    )
    if bias is not None:
        bias.tofile(os.path.join(input_dir, "input_bias.bin"))
    # Output: bf16/fp16 as uint16
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


# ---------------- FP8 E4M3FN (NPU bias=7) ----------------


def float8_e4m3fn_to_float(data_uint8):
    """Convert FP8E4M3FN (uint8) to FP32 with NPU bias=7."""
    data = data_uint8.astype(np.uint8)
    sign = (data >> 7) & 1
    exp = (data >> 3) & 15
    mant = data & 7

    fp32 = np.zeros_like(data, dtype=np.float32)
    zero_mask = (exp == 0) & (mant == 0)
    fp32[zero_mask] = 0.0
    normal_mask = ~zero_mask
    fp32[normal_mask] = (
        (1.0 - 2.0 * sign[normal_mask])
        * np.power(2.0, exp[normal_mask] - 7.0)
        * (1.0 + mant[normal_mask] / 8.0)
    )
    return fp32


def float_to_fp8_e4m3fn_vec(fp32_arr):
    """Vectorized FP32 -> FP8E4M3FN with NPU bias=7."""
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)

    MAX_NORMAL = 1.875 * 128.0  # 240.0
    MIN_NORMAL = 2.0 ** (-6)  # 0.015625

    log2_val = np.log2(np.clip(abs_val, 1e-30, None))
    e = np.floor(log2_val).astype(np.int32) + 7  # bias = 7
    e = np.clip(e, 1, 14)

    normalized = abs_val / np.power(2.0, (e - 7).astype(np.float32))
    mant = np.clip(np.round((normalized - 1.0) * 8.0).astype(np.int32), 0, 7)

    overflow = abs_val >= MAX_NORMAL
    underflow = abs_val < MIN_NORMAL
    e[overflow] = 14
    mant[overflow] = 7
    e[underflow] = 0
    mant[underflow] = 0
    sign[underflow] = 0

    result = (sign << 7) | (e.astype(np.uint8) << 3) | mant.astype(np.uint8)
    result[fp32 == 0.0] = 0
    return result


# ---------------- FP8 E5M2 (standard bias=15) ----------------


def float8_e5m2_to_float(data_uint8):
    """Convert FP8E5M2 (uint8) to FP32 with standard bias=15."""
    data = data_uint8.astype(np.uint8)
    sign = (data >> 7) & 1
    exp = (data >> 2) & 31
    mant = data & 3

    fp32 = np.zeros_like(data, dtype=np.float32)
    zero_mask = (exp == 0) & (mant == 0)
    sub_mask = (exp == 0) & (mant != 0)
    normal_mask = exp > 0
    fp32[normal_mask] = (
        (1.0 - 2.0 * sign[normal_mask])
        * np.power(2.0, exp[normal_mask] - 15.0)
        * (1.0 + mant[normal_mask] / 4.0)
    )
    fp32[sub_mask] = (
        (1.0 - 2.0 * sign[sub_mask]) * (mant[sub_mask] / 4.0) * np.power(2.0, -14.0)
    )
    return fp32


def float_to_fp8_e5m2_vec(fp32_arr):
    """Vectorized FP32 -> FP8E5M2 with standard bias=15."""
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)

    log2_val = np.floor(np.log2(np.clip(abs_val, 1e-30, None)))
    e = (log2_val + 15.0).astype(np.int32)
    e = np.clip(e, 1, 30)

    normalized = abs_val / np.power(2.0, (e - 15).astype(np.float32))
    mant = np.rint((normalized - 1.0) * 4.0).astype(np.int32)
    carry = mant >= 4
    e[carry] += 1
    mant[carry] = 0

    result = (sign << 7) | (e.astype(np.uint8) << 2) | mant.astype(np.uint8)
    result[fp32 == 0.0] = 0
    return result


# ---------------- FP4 E2M1 (packed fp4x2) ----------------


def float_to_fp4_e2m1_vec(fp32_arr):
    """FP32 -> FP4 E2M1 element codes (uint8 nibble, bit3=sign, bits2-0=magnitude code)."""
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)

    flat = abs_val.reshape(-1)
    codes = np.empty(flat.shape, dtype=np.uint8)
    chunk = 1 << 20
    for s in range(0, flat.size, chunk):
        blk = flat[s : s + chunk]
        idx = np.argmin(np.abs(blk[:, None] - FP4_MAGS), axis=-1)
        codes[s : s + chunk] = idx
    codes = codes.reshape(abs_val.shape)
    return (sign << 3) | codes


def fp4_e2m1_to_float(codes):
    """FP4 E2M1 element codes (nibble) -> FP32."""
    sign = (codes >> 3) & 1
    mag = FP4_MAGS[(codes & 7).astype(np.int64)]
    return (1.0 - 2.0 * sign) * mag


def pack_fp4x2(codes):
    """Pack FP4 codes along the last axis, 2 elements per byte (low nibble first)."""
    even = codes[..., 0::2] & 0xF
    odd = codes[..., 1::2] & 0xF
    return (even | (odd << 4)).astype(np.uint8)


# ---------------- Scale (FP8 E8M0) ----------------


def generate_fp8_e8m0_scale(shape):
    """Generate FP8E8M0 scale values (exp ~127-129 -> scale ~1-4)."""
    exp_values = np.random.randint(127, 129, shape).astype(np.uint8)
    return exp_values


def float8_e8m0_to_float(scale_uint8):
    """Convert FP8E8M0 (uint8) to FP32 as power of 2."""
    exp = scale_uint8.astype(np.float32)
    return np.power(2.0, exp - 127.0)


# ---------------- Quantize / dequantize helpers ----------------


def quantize(kind, fp32_arr):
    """FP32 -> element codes (uint8 per element; nibble code for FP4)."""
    if kind == "e4m3":
        return float_to_fp8_e4m3fn_vec(fp32_arr)
    if kind == "e5m2":
        return float_to_fp8_e5m2_vec(fp32_arr)
    if kind == "e2m1":
        return float_to_fp4_e2m1_vec(fp32_arr)
    raise ValueError(f"Unknown quant kind: {kind}")


def dequantize(kind, codes):
    """Element codes -> FP32 (scale not applied)."""
    if kind == "e4m3":
        return float8_e4m3fn_to_float(codes)
    if kind == "e5m2":
        return float8_e5m2_to_float(codes)
    if kind == "e2m1":
        return fp4_e2m1_to_float(codes)
    raise ValueError(f"Unknown quant kind: {kind}")


def to_file_layout(codes, kind, transpose):
    """Element codes [rows, cols] -> file bytes (optionally col-major, packed for FP4)."""
    arr = np.ascontiguousarray(codes.T) if transpose else codes
    if kind == "e2m1":
        return pack_fp4x2(arr)
    return arr


def apply_mxfp_scale(data_fp32, scale_fp8, divisor=64, c0=2):
    """
    Apply MXFP scale to already-decoded FP32 data.
      A: data [M, K], scale [M, ceil(K/64), 2]
      B: data [K, N], scale [ceil(K/64), 2, N]
    Each group of `divisor` elements shares `c0` scale values (sub-group = divisor/c0).
    """
    sub_group = divisor // c0
    fp32_scale = float8_e8m0_to_float(scale_fp8.astype(np.uint8))

    is_a_scale = (
        scale_fp8.ndim == 3 and scale_fp8.shape[-1] == c0 and scale_fp8.shape[-2] > c0
    )
    is_b_scale = scale_fp8.ndim == 3 and scale_fp8.shape[1] == c0

    if is_a_scale:
        k_shape = data_fp32.shape[1]
        bcast = np.repeat(
            fp32_scale.reshape(fp32_scale.shape[0], -1), sub_group, axis=-1
        )[..., :k_shape]
        return data_fp32 * bcast
    if is_b_scale:
        k_shape = data_fp32.shape[0]
        # [CK, 2, N] -> [N, CK*2], expand 32x along K, back to [K, N]
        bcast_t = np.repeat(
            np.ascontiguousarray(fp32_scale.transpose(2, 0, 1)).reshape(
                fp32_scale.shape[2], -1
            ),
            sub_group,
            axis=-1,
        )[..., :k_shape]
        return data_fp32 * bcast_t.T
    raise ValueError(f"Unexpected scale shape: {scale_fp8.shape}")


def gen_golden_data_matmul_all_to_all(m, k, n, rank_num, quant_type, is_fp16, is_bias):
    """
    Generate golden data for QuantMatmulAllToAll fusion operator.

    Each rank holds A_i [M, K] and the full B_i [K, N].

    CPU golden:
      1. C_i = dequant(A_i) x dequant(B_i) (+ bias) -> [M, N]
      2. permute: C.view(M, rankSize, Np).permute(1, 0, 2) -> [rankSize, M, Np]
      3. AllToAll exchange: rank j receives C_i[:, j*Np:(j+1)*Np] from every
         rank i, concatenated along M -> [rankSize*M, Np]
    """
    a_kind, b_kind = QUANT_TYPES[quant_type]
    M = m
    K = k
    N = n
    np_ = N // rank_num
    ck = math.ceil(K / 64)

    print(
        f"  M={M}, K={K}, N={N}, Np={np_} (per rank), quantType={quant_type} (A={a_kind}, B={b_kind}), "
        f"isFp16={is_fp16}, isBias={is_bias}"
    )

    # 每个 rank 的 A / scaleA / B / scaleB / bias 都用独立 seed 生成
    a_codes_list = []
    a_scale_list = []
    b_codes_list = []
    b_scale_list = []
    bias_list = []
    # fp16 动态范围 ±65504，全正数据 C≈K·E[a]·E[b] 随 K 增长会溢出 inf 使比对失效，
    # 预估 C 幅度并对 A/B 幅度等比压缩（bias_scale 由实际 a/b mean 计算，自动跟随）
    amp = 1.0
    if is_fp16:
        a_mid = (DATA_RANGE[a_kind][0] + DATA_RANGE[a_kind][1]) / 2.0 * 1.5
        b_mid = (DATA_RANGE[b_kind][0] + DATA_RANGE[b_kind][1]) / 2.0 * 1.5
        amp = min(1.0, 5.0e4 / (a_mid * b_mid * K))
    for rank_id in range(rank_num):
        np.random.seed(BASE_SEED + rank_id)
        a_lo, a_hi = DATA_RANGE[a_kind]
        b_lo, b_hi = DATA_RANGE[b_kind]
        a_codes = quantize(a_kind, np.random.uniform(a_lo, a_hi, (M, K)) * amp)
        a_scale = generate_fp8_e8m0_scale((M, ck, 2))
        b_codes = quantize(b_kind, np.random.uniform(b_lo, b_hi, (K, N)) * amp)
        b_scale = generate_fp8_e8m0_scale((ck, 2, N))
        a_codes_list.append(a_codes)
        a_scale_list.append(a_scale)
        b_codes_list.append(b_codes)
        b_scale_list.append(b_scale)
        bias_list.append(None)

    print(
        f"  [DIAG] A[0]: {a_kind} shape={a_codes_list[0].shape}, scale.shape={a_scale_list[0].shape}"
    )
    print(
        f"  [DIAG] B[0]: {b_kind} shape={b_codes_list[0].shape}, scale.shape={b_scale_list[0].shape}"
    )

    # bias 幅度按典型 C 量级自适应（~1/8），避免大数值 matmul 输出把 bias 淹没
    bias_scale = 0.0
    if is_bias:
        a_mean = float(
            np.abs(
                apply_mxfp_scale(dequantize(a_kind, a_codes_list[0]), a_scale_list[0])
            ).mean()
        )
        b_mean = float(
            np.abs(
                apply_mxfp_scale(dequantize(b_kind, b_codes_list[0]), b_scale_list[0])
            ).mean()
        )
        bias_scale = 0.125 * a_mean * b_mean * K
    for rank_id in range(rank_num):
        if is_bias:
            np.random.seed(BASE_SEED + 100 + rank_id)
            bias_list[rank_id] = np.random.uniform(
                -bias_scale, bias_scale, (N,)
            ).astype(np.float32)
    if is_bias:
        print(
            f"  [DIAG] bias range [-{bias_scale:.1f}, {bias_scale:.1f}] (auto-scaled to typical C magnitude)"
        )

    # Step 1: per-rank MXFP dequantize -> FP32 matmul (+ bias) -> C_i [M, N]
    cpu_outputs_full = []
    for rank_id in range(rank_num):
        a_deq = apply_mxfp_scale(
            dequantize(a_kind, a_codes_list[rank_id]), a_scale_list[rank_id]
        )
        b_deq = apply_mxfp_scale(
            dequantize(b_kind, b_codes_list[rank_id]), b_scale_list[rank_id]
        )
        a_cpu = torch.from_numpy(a_deq)
        b_cpu = torch.from_numpy(b_deq)
        c_full = torch.matmul(a_cpu, b_cpu)  # [M, N]
        if is_bias:
            c_full = c_full + torch.from_numpy(bias_list[rank_id])
        cpu_outputs_full.append(c_full)
        print(
            f"  Rank {rank_id}: C shape {c_full.shape}, range [{c_full.min():.2f}, {c_full.max():.2f}]"
        )

    # Step 2: permute — C.view(M, rankSize, Np).permute(1, 0, 2) -> [rankSize, M, Np]
    allto_all_inputs = [
        c.view(M, rank_num, np_).permute(1, 0, 2) for c in cpu_outputs_full
    ]

    # Step 3: AllToAll exchange — rank j gets C_i[:, j*Np:(j+1)*Np] from every rank i
    allto_all_outputs = []
    for dst_rank in range(rank_num):
        out_blocks = []
        for src_rank in range(rank_num):
            out_blocks.append(allto_all_inputs[src_rank][dst_rank])
        allto_all_out = torch.cat(out_blocks, dim=0)  # [rankSize*M, Np]
        allto_all_outputs.append(allto_all_out)
        print(f"  Rank {dst_rank}: AllToAll output shape {allto_all_out.shape}")

    # Convert to BF16 / FP16
    out_dtype = torch.float16 if is_fp16 else torch.bfloat16
    allto_all_outputs_out = [out.to(out_dtype) for out in allto_all_outputs]
    for out in allto_all_outputs_out:
        if not torch.isfinite(out).all():
            print(
                f"Error: golden output contains non-finite values (inf/nan) after cast to {out_dtype}"
            )
            sys.exit(1)

    base_dir = os.getcwd()
    for rank_id in range(rank_num):
        write_artifacts(
            base_dir,
            rank_id,
            to_file_layout(a_codes_list[rank_id], a_kind, transpose=False),
            to_file_layout(b_codes_list[rank_id], b_kind, transpose=True),
            a_scale_list[rank_id],
            b_scale_list[rank_id],
            bias_list[rank_id],
            allto_all_outputs_out[rank_id],
        )



# ============================ HiF4 全量化(A6/ascend960dt) ============================
# HiF4: 64 个 S1P2(4-bit) 元素 + 32bits 三级共享元数据(1 个 hif4_scale, 4 字节)。
#   Level-1 E6M2 (Byte0): exponent(6bit, bias=48) + mantissa(2bit)
#   Level-2 E1_8 (Byte1): 8 个 1-bit 微指数, 每 bit 覆盖 8 个元素
#   Level-3 E1_16 (Byte2~3): 16 个 1-bit 微指数, 每 bit 覆盖 4 个元素(Byte2 低 8 位在前)
#   V[i] = E6M2 * 2^(E1_8[i//8] + E1_16[i//4]) * S1P2[i]
# 约束: K 为偶数(无 32/64/128 对齐约束); 不支持 bias; y 仅 FP16/BF16。


def gen_hif4_meta(m, k_groups, rng=None):
    """生成 hif4_scale 三级元数据: Byte0=E6M2, Byte1=E1_8, Byte2=E1_16_lo, Byte3=E1_16_hi"""
    rng = rng if rng is not None else np.random
    meta = np.zeros((m, k_groups, 4), dtype=np.uint8)
    meta[:, :, 0] = rng.integers(0, 0x80, (m, k_groups))   # E6M2
    meta[:, :, 1] = rng.integers(0, 0x100, (m, k_groups))  # E1_8
    meta[:, :, 2] = rng.integers(0, 0x100, (m, k_groups))  # E1_16 低 8 位
    meta[:, :, 3] = rng.integers(0, 0x100, (m, k_groups))  # E1_16 高 8 位
    return meta


def hif4_dequant(data_packed, meta):
    """HiF4 反量化: V = E6M2 * 2^(E1_8[i//8] + E1_16[i//4]) * S1P2[i], 输出 FP32 [rows, K]"""
    m, k_half = data_packed.shape
    k = k_half * 2
    data = np.asarray(data_packed, dtype=np.uint8)
    lo = (data & 0x0F).astype(np.int32)
    hi = ((data >> 4) & 0x0F).astype(np.int32)
    s1p2_code = np.empty((m, k), dtype=np.int32)
    s1p2_code[:, 0::2] = lo
    s1p2_code[:, 1::2] = hi
    # S1P2 解码: sign(1) + int(1) + frac(2)
    sign = np.where((s1p2_code >> 3) & 0x1 == 1, -1.0, 1.0)
    int_part = (s1p2_code >> 2) & 0x1
    frac_part = s1p2_code & 0x3
    s1p2_val = sign * (int_part.astype(np.float32) + frac_part.astype(np.float32) * 0.25)
    # E6M2 解码: exponent(6bit, bias=48) + mantissa(2bit)
    e6m2_code = meta[:, :, 0].astype(np.int32)
    e6m2_exp = ((e6m2_code >> 2) & 0x3F).astype(np.int32) - 48
    e6m2_val = np.power(2.0, e6m2_exp.astype(np.float32)) * (
        1.0 + (e6m2_code & 0x3).astype(np.float32) * 0.25
    )
    # E1_8 / E1_16 微指数(每 8/4 元素共享 1 bit)
    e1_8_bits = np.unpackbits(meta[:, :, 1].astype(np.uint8), axis=-1)
    e1_16 = (meta[:, :, 3].astype(np.uint16) << 8) | meta[:, :, 2].astype(np.uint16)
    e1_16_bits = np.zeros((m, k_groups_of(meta), 16), dtype=np.float32)
    for bit in range(16):
        e1_16_bits[:, :, bit] = ((e1_16 >> (15 - bit)) & 0x1).astype(np.float32)
    idx = np.arange(k, dtype=np.int64) % 64
    exp_scale = e1_8_bits[:, :, idx // 8].astype(np.float32) * 1.0 + e1_16_bits[:, :, idx // 4]
    group_idx = np.arange(k, dtype=np.int64) // 64
    scale = e6m2_val[:, group_idx] * np.power(2.0, exp_scale[:, group_idx, np.arange(k)])
    return (scale * s1p2_val).astype(np.float32)


def k_groups_of(meta):
    """meta 形状 [rows, k_groups, 4] -> k_groups"""
    return meta.shape[1]


def gen_hif4_data(m, k, n, rank_num, is_fp16):
    """生成 HiF4 全量化 ST 数据与 golden(与 gen_golden_data_matmul_all_to_all 同源的 alltoall 流程)。"""
    assert k % 2 == 0, "K must be even (4-bit 2:1 packing)"
    assert n % rank_num == 0, "n must be divisible by rank_num"
    np_ = n // rank_num
    ck = math.ceil(k / 64)
    print(f"  M={m}, K={k}, N={n}, Np={np_} (per rank), quantType=hif4(hifloat4 x hifloat4), isFp16={is_fp16}")

    a_packed_list = []
    a_meta_list = []
    b_packed_list = []
    b_meta_list = []
    for rank_id in range(rank_num):
        rng = np.random.default_rng(BASE_SEED + rank_id)
        # A: (m, k) 沿 K 打包 -> (m, k//2); B: 转置 (n, k) 沿 K 打包 -> (n, k//2)
        a_packed_list.append(rng.integers(0, 0x100, (m, k // 2), dtype=np.uint8))
        b_packed_list.append(rng.integers(0, 0x100, (n, k // 2), dtype=np.uint8))
        a_meta_list.append(gen_hif4_meta(m, ck, rng))
        b_meta_list.append(gen_hif4_meta(n, ck, rng))

    # Step 1: 反量化 -> FP32 matmul(HiF4 不支持 bias) -> C_i [M, N]
    cpu_outputs_full = []
    for rank_id in range(rank_num):
        a_deq = hif4_dequant(a_packed_list[rank_id], a_meta_list[rank_id])
        b_deq = hif4_dequant(b_packed_list[rank_id], b_meta_list[rank_id])
        c_full = torch.matmul(torch.from_numpy(a_deq), torch.from_numpy(b_deq).T)
        cpu_outputs_full.append(c_full)
        print(f"  Rank {rank_id}: C shape {c_full.shape}, range [{c_full.min():.2f}, {c_full.max():.2f}]")

    # Step 2/3: alltoall 模拟(与 MX 流程一致)
    allto_all_inputs = [c.view(m, rank_num, np_).permute(1, 0, 2) for c in cpu_outputs_full]
    out_dtype = torch.float16 if is_fp16 else torch.bfloat16
    allto_all_outputs_out = []
    for dst_rank in range(rank_num):
        out = torch.cat([allto_all_inputs[src][dst_rank] for src in range(rank_num)], dim=0).to(out_dtype)
        if not torch.isfinite(out).all():
            print(f"Error: golden output contains non-finite values (inf/nan) after cast to {out_dtype}")
            sys.exit(1)
        allto_all_outputs_out.append(out)
        print(f"  Rank {dst_rank}: AllToAll output shape {out.shape}")

    # HiF4 产物布局(不复用 write_artifacts 的 MX scaleB 转置逻辑):
    #   input_a.bin:      (m, k//2) uint8, hifloat4 沿 K 2:1 打包(ND)
    #   input_b.bin:      (n, k//2) uint8, 转置 B 沿 K 2:1 打包(DN)
    #   input_scaleA.bin: (m, ceil(k/64), 4) uint8, hif4_scale 三级元数据 [M, K/64, 1]
    #   input_scaleB.bin: (n, ceil(k/64), 4) uint8, [N, K/64, 1]
    #   cpu_output.bin:   golden (rank_num*m, n/rank_num) fp16/bf16
    base_dir = os.getcwd()
    for rank_id in range(rank_num):
        input_dir = os.path.join(base_dir, "input", str(rank_id))
        output_dir = os.path.join(base_dir, "output", str(rank_id))
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        a_packed_list[rank_id].tofile(os.path.join(input_dir, "input_a.bin"))
        b_packed_list[rank_id].tofile(os.path.join(input_dir, "input_b.bin"))
        a_meta_list[rank_id].tofile(os.path.join(input_dir, "input_scaleA.bin"))
        b_meta_list[rank_id].tofile(os.path.join(input_dir, "input_scaleB.bin"))
        allto_all_outputs_out[rank_id].view(torch.uint16).numpy().tofile(
            os.path.join(output_dir, "cpu_output.bin")
        )


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 8:
        print(
            "Usage: python3 gen_data.py m k n rank_num [quant_type] [is_fp16] [is_bias]"
        )
        print("  m: matrix A row dimension M")
        print("  k: matrix A column dimension K (full K, not split)")
        print("  n: matrix B column dimension N (split to N/rank_num per rank)")
        print("  rank_num: number of ranks")
        print(
            "  quant_type: 0=E2M1E2M1(FP4), 1=E4M3E4M3, 2=E5M2E5M2, 3=E4M3E5M2, 4=E5M2E4M3, 5=HIF4HIF4 (default: 1)"
        )
        print("  is_fp16: output float16 instead of bfloat16 (default: 0)")
        print("  is_bias: add bias input [N] (default: 0)")
        print("\nExample: python3 gen_data.py 2048 3584 4096 4 1 0 0")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    quant_type = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    is_fp16 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    is_bias = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    if rank_num <= 0:
        print(f"Error: rank_num={rank_num} must be a positive integer")
        sys.exit(1)

    if k % 32 != 0:
        print(f"Error: K={k} not divisible by 32")
        sys.exit(1)

    if math.ceil(k / 64) % 2 != 0:
        print(f"Error: ceil(K/64)={math.ceil(k / 64)} not even")
        sys.exit(1)

    if n % rank_num != 0:
        print(f"Error: n={n} not divisible by rank_num={rank_num}")
        sys.exit(1)

    if quant_type != 5 and quant_type not in QUANT_TYPES:
        print(f"Error: quant_type={quant_type} must be 0-5")
        sys.exit(1)

    if quant_type == 5:
        # HiF4: K 仅需偶数(无 32/64/128 对齐约束), 不支持 bias
        if k % 2 != 0:
            print(f"Error: K={k} must be even in hif4 quant mode (4-bit 2:1 packing)")
            sys.exit(1)
        if is_bias:
            print("Error: bias is not supported in hif4 quant mode")
            sys.exit(1)
        gen_hif4_data(m, k, n, rank_num, is_fp16)
        print("Test data generation completed!")
        sys.exit(0)

    print("Generating QuantMatmulAllToAll test data:")
    print(f"  M={m}, K_full={k}, N={n}, rankSize={rank_num}")
    print(f"  Np={n // rank_num} (per rank, K not split)")

    gen_golden_data_matmul_all_to_all(m, k, n, rank_num, quant_type, is_fp16, is_bias)

    print("Test data generation completed!")
