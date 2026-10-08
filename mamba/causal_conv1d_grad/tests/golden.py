#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal_conv1d_grad CPU Golden 参考实现

实现反向数学公式（重算 z → silu' → grad_z → dx/dweight/dbias）
全程 fp32 累加后 cast 回输入 dtype
"""

import numpy as np


def causal_conv1d_grad_golden(grad_y, x, weight, bias, activation="silu"):
    """
    causal_conv1d_grad CPU Golden 参考实现

    Args:
        grad_y: 上游梯度，shape [B, T, D]，dtype float16/bfloat16
        x: 前向输入序列，shape [B, T, D]，dtype 同 grad_y
        weight: 卷积权重，shape [kW, D]，kW=4，dtype 同 grad_y
        bias: 偏置，shape [D]，dtype 同 grad_y
        activation: 激活函数类型，固定 "silu"

    Returns:
        dx: 对输入 x 的梯度，shape [B, T, D]，dtype 同 grad_y
        dweight: 对权重的梯度，shape [kW, D]，dtype 同 grad_y
        dbias: 对偏置的梯度，shape [D]，dtype 同 grad_y
    """
    # 保存原始 dtype
    original_dtype = grad_y.dtype

    # 全部转换为 fp32 进行计算
    grad_y_fp32 = grad_y.astype(np.float32)
    x_fp32 = x.astype(np.float32)
    weight_fp32 = weight.astype(np.float32)
    bias_fp32 = bias.astype(np.float32)

    B, T, D = x_fp32.shape
    kW = weight_fp32.shape[0]  # 固定 4

    # 验证 activation 类型
    assert activation == "silu", f"仅支持 activation='silu'，当前为 '{activation}'"

    # 验证 shape 约束
    assert grad_y_fp32.shape == x_fp32.shape, f"grad_y.shape {grad_y_fp32.shape} != x.shape {x_fp32.shape}"
    assert weight_fp32.shape == (kW, D), f"weight.shape {weight_fp32.shape} != ({kW}, {D})"
    assert bias_fp32.shape == (D,), f"bias.shape {bias_fp32.shape} != ({D},)"
    assert kW == 4, f"kW 必须为 4，当前为 {kW}"
    # 移除 D=12288 的硬编码限制，允许测试不同 dim

    # ========== 第 1 步：重算 pre-activation z ==========
    # 左侧 padding kW-1=3 个零（冷启动）
    zeros_pad = np.zeros((B, kW - 1, D), dtype=np.float32)
    x_pad = np.concatenate([zeros_pad, x_fp32], axis=1)  # [B, T+3, D]

    # z_t = Σ_{j=0..3} w_j · x_{t-j} + b
    # kW=4 展开（无循环）
    z = (weight_fp32[0] * x_pad[:, 3:T + 3, :]
         + weight_fp32[1] * x_pad[:, 2:T + 2, :]
         + weight_fp32[2] * x_pad[:, 1:T + 1, :]
         + weight_fp32[3] * x_pad[:, 0:T + 0, :]
         + bias_fp32)  # [B, T, D]

    # ========== 第 2 步：计算 SiLU 导数 ==========
    # sigmoid(z) = 1 / (1 + exp(-z))
    sigmoid_z = 1.0 / (1.0 + np.exp(-z))

    # silu'(z) = σ(z) * (1 + z * (1 - σ(z)))
    silu_deriv = sigmoid_z * (1.0 + z * (1.0 - sigmoid_z))

    # ========== 第 3 步：计算 grad_z ==========
    grad_z = grad_y_fp32 * silu_deriv  # [B, T, D]

    # ========== 第 4 步：计算 dbias ==========
    # dbias = Σ_{B,T} grad_z
    dbias = np.sum(grad_z, axis=(0, 1))  # [D]

    # ========== 第 5 步：计算 dweight ==========
    # dweight[j,d] = Σ_{B,T} grad_z_t · x_{t-j}
    dw0 = np.sum(grad_z * x_pad[:, 3:T + 3, :], axis=(0, 1))
    dw1 = np.sum(grad_z * x_pad[:, 2:T + 2, :], axis=(0, 1))
    dw2 = np.sum(grad_z * x_pad[:, 1:T + 1, :], axis=(0, 1))
    dw3 = np.sum(grad_z * x_pad[:, 0:T + 0, :], axis=(0, 1))
    dweight = np.stack([dw0, dw1, dw2, dw3], axis=0)  # [4, D]

    # ========== 第 6 步：计算 dx ==========
    # dx[t,d] = Σ_{j=0..3} grad_z_{t+j} · w_j
    # 右侧 padding grad_z（t+j > T-1 时取 0）
    gz_rpad = np.concatenate([grad_z, np.zeros((B, kW - 1, D), dtype=np.float32)], axis=1)  # [B, T+3, D]

    dx = (weight_fp32[0] * gz_rpad[:, 0:T + 0, :]
          + weight_fp32[1] * gz_rpad[:, 1:T + 1, :]
          + weight_fp32[2] * gz_rpad[:, 2:T + 2, :]
          + weight_fp32[3] * gz_rpad[:, 3:T + 3, :])  # [B, T, D]

    # ========== 第 7 步：Cast 回原始 dtype ==========
    dx = dx.astype(original_dtype)
    dweight = dweight.astype(original_dtype)
    dbias = dbias.astype(original_dtype)

    return dx, dweight, dbias


def test_golden():
    """简单自测：验证 golden 实现的基本正确性"""
    print("========== Golden 自测开始 ==========")

    # 测试参数
    B, T, D = 2, 8, 12288
    kW = 4
    dtype = np.float16

    # 生成随机输入
    np.random.seed(42)
    grad_y = np.random.randn(B, T, D).astype(dtype) * 0.1
    x = np.random.randn(B, T, D).astype(dtype) * 0.5
    weight = np.random.randn(kW, D).astype(dtype) * 0.1
    bias = np.random.randn(D).astype(dtype) * 0.01

    # 调用 golden
    dx, dweight, dbias = causal_conv1d_grad_golden(grad_y, x, weight, bias, activation="silu")

    # 验证输出 shape
    assert dx.shape == (B, T, D), f"dx.shape {dx.shape} != {(B, T, D)}"
    assert dweight.shape == (kW, D), f"dweight.shape {dweight.shape} != {(kW, D)}"
    assert dbias.shape == (D,), f"dbias.shape {dbias.shape} != {(D,)}"

    # 验证输出 dtype
    assert dx.dtype == dtype, f"dx.dtype {dx.dtype} != {dtype}"
    assert dweight.dtype == dtype, f"dweight.dtype {dweight.dtype} != {dtype}"
    assert dbias.dtype == dtype, f"dbias.dtype {dbias.dtype} != {dtype}"

    # 验证非零输出（检查是否有梯度回传）
    assert np.any(dx != 0), "dx 全为零，可能计算错误"
    assert np.any(dweight != 0), "dweight 全为零，可能计算错误"
    assert np.any(dbias != 0), "dbias 全为零，可能计算错误"

    # 打印统计信息
    print(f"dx    范围: [{dx.min():.6f}, {dx.max():.6f}], 均值: {dx.mean():.6f}, 标准差: {dx.std():.6f}")
    print(f"dweight 范围: [{dweight.min():.6f}, {dweight.max():.6f}], 均值: {dweight.mean():.6f}, 标准差: {dweight.std():.6f}")
    print(f"dbias   范围: [{dbias.min():.6f}, {dbias.max():.6f}], 均值: {dbias.mean():.6f}, 标准差: {dbias.std():.6f}")

    print("========== Golden 自测通过 ==========\n")


if __name__ == "__main__":
    test_golden()
