#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
from typing import Tuple

# ----------------------------------------------------------------------
# mxfp4e2m1 编解码（4位）
# 位布局: [s][e1][e0][m]
#   s: 符号位 (0正,1负)
#   e: 2位指数 (0~3)
#   m: 1位尾数 (0或1)
# 解码值:
#   e==0: (-1)^s * 2^(-1) * (m/2) = (-1)^s * m/4
#   e>0:  (-1)^s * 2^(e-1) * (1 + m/2)
# ----------------------------------------------------------------------
def _mxfp4_decode(bits: int) -> float:
    """将4位整数(0~15)解码为浮点数值"""
    s = (bits >> 3) & 0x1
    e = (bits >> 1) & 0x3
    m = bits & 0x1
    if e == 0:
        val = (m / 4.0)   # 非规格化
    else:
        val = (1 + m / 2.0) * (1 << (e - 1))  # 规格化
    return -val if s else val

def _mxfp4_encode(x: float) -> int:
    """将浮点数舍入到最近的mxfp4表示，返回4位编码(0~15)"""
    # 所有可能的编码及其值
    candidates = [(i, _mxfp4_decode(i)) for i in range(16)]
    # 找到最接近的值
    best_idx = min(candidates, key=lambda t: abs(t[1] - x))[0]
    return best_idx

# 预计算所有编码的值，用于向量化量化（可选）
_MXFP4_VALUES = np.array([_mxfp4_decode(i) for i in range(16)], dtype=np.float32)

def quantize_to_mxfp4(x: np.ndarray) -> np.ndarray:
    """将浮点数数组量化为mxfp4，返回uint8数组，每个元素存储低4位编码"""
    x_flat = x.ravel().astype(np.float32)
    # 对每个元素找到最近编码的索引
    # 利用广播计算所有差值，argmin（注意内存，若数组很大可循环）
    # 这里使用简单循环，适合中等规模；大数据可优化为向量化最近邻搜索
    codes = np.empty(len(x_flat), dtype=np.uint8)
    for i, val in enumerate(x_flat):
        # 边界裁剪，避免搜索时溢出
        clipped = np.clip(val, -6.0, 6.0)
        codes[i] = _mxfp4_encode(clipped)
    return codes.reshape(x.shape)

# ----------------------------------------------------------------------
# fp8e8m0 编解码（8位）
# 位布局: [s][e6][e5][e4][e3][e2][e1][e0]
#   s: 符号位 (0正,1负) – 此处缩放因子为正，固定为0
#   e: 7位指数 (0~127)，偏置63，表示 2^(e-63)
# ----------------------------------------------------------------------
def float_to_fp8e8m0(x: float) -> np.uint8:
    """将正浮点数转换为fp8e8m0位表示（假设x > 0）"""
    if x <= 0:
        return np.uint8(0)
    # 计算指数: 2^(e-63) ≈ x  => e = round(log2(x)) + 63
    log2x = np.log2(x)
    e = int(np.round(log2x)) + 63
    e = np.clip(e, 0, 127)   # 7位指数范围
    # 符号位为0，组合成8位
    bits = (0 << 7) | e
    return np.uint8(bits)

def fp8e8m0_to_float(bits: np.uint8) -> float:
    """将fp8e8m0位表示转换回浮点数"""
    s = (bits >> 7) & 0x1
    e = bits & 0x7F
    val = 2.0 ** (e - 63)
    return -val if s else val

# ----------------------------------------------------------------------
# 主量化函数
# ----------------------------------------------------------------------
def quantize_bf16_to_mxfp4(
    array: np.ndarray,
    dtype_in: np.dtype = np.float32   # 实际应为bfloat16，但numpy无原生支持，用float32代替
) -> Tuple[np.ndarray, np.uint8]:
    """
    将bfloat16数组量化为mxfp4，并返回量化系数(fp8e8m0格式)。

    参数:
        array: 输入数组，元素类型应为bfloat16（此处用float32模拟）
        dtype_in: 输入数据类型（仅用于类型检查）

    返回:
        (quantized_codes, scale_fp8)
            quantized_codes: uint8数组，形状与输入相同，每个元素存储4位mxfp4编码
            scale_fp8: uint8标量，fp8e8m0格式的量化系数
    """
    # 转换为float64进行精确计算
    x = array.astype(np.float64)
    max_abs = np.max(np.abs(x))

    if max_abs == 0:
        # 全零数组，缩放因子任意（取1），量化结果全0
        scale = 1.0
        scale_fp8 = float_to_fp8e8m0(scale)
        quantized = np.zeros(x.shape, dtype=np.uint8)
        return quantized, scale_fp8

    # 确定缩放因子 scale = 2^exp，使得 max_abs/scale ≤ 6
    # 即 scale = 2^ceil(log2(max_abs/6))
    exp = int(np.ceil(np.log2(max_abs / 6.0)))
    scale = 2.0 ** exp
    scale_fp8 = float_to_fp8e8m0(scale)

    # 缩放并量化
    x_scaled = x / scale
    quantized = quantize_to_mxfp4(x_scaled)

    return quantized, scale_fp8

# ----------------------------------------------------------------------
# 反量化（验证用）
# ----------------------------------------------------------------------
def dequantize_mxfp4(codes: np.ndarray, scale_fp8: np.uint8) -> np.ndarray:
    """
    将mxfp4编码和fp8e8m0缩放因子恢复为浮点数数组。
    """
    scale = fp8e8m0_to_float(scale_fp8)
    # 将每个编码解码为mxfp4数值
    values = np.empty(codes.shape, dtype=np.float64)
    for i, code in enumerate(codes.ravel()):
        values.ravel()[i] = _mxfp4_decode(int(code))
    return values * scale


# ======================================================================
# 示例用法
if __name__ == "__main__":
    # 生成随机bf16数据（模拟）
    np.random.seed(42)
    data = np.random.uniform(-100, 100, (4, 4)).astype(np.float32)  # 范围约[-30,30]

    # 量化
    codes, scale_fp8 = quantize_bf16_to_mxfp4(data)
    print("原始数据 (部分):\n", data[:2, :2])
    print("\n量化编码 (uint8, 低4位有效):\n", codes[:2, :2])
    print(f"缩放因子 (fp8e8m0): 0x{scale_fp8:02X}, 值 = {fp8e8m0_to_float(scale_fp8):.6g}")

    # 反量化对比
    reconstructed = dequantize_mxfp4(codes, scale_fp8)
    print("\n反量化结果 (部分):\n", reconstructed[:2, :2])
    error = np.abs(data - reconstructed)
    print(f"最大绝对误差: {np.max(error):.6f}")


