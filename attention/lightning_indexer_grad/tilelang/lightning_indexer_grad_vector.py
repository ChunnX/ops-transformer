#!/usr/bin/env python3
"""
TileLang-Ascend Vector 内核 - 用于 lightning_indexer_grad

本模块提供 Vector 单元内核:
- ReluGrad: 应用 ReLU 反向掩码
- GatherTopk: 基于 TopK 索引收集 Key 数据 (CPU + NPU 混合)
- ScatterAdd: 使用原子加法将梯度散射到 Key 位置

由于 TileLang 主要为计算密集型操作设计，
涉及不规则内存访问的 Gather 和 Scatter 操作采用混合方式处理。
"""

import tilelang
import tilelang.language as T
from tilelang import jit
import torch
import numpy as np

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@jit(out_idx=[-1], pass_configs=pass_configs)
def relu_grad_kernel(
    M: int, K: int,
    block_m: int, block_k: int,
    dtype: str = 'float16'
):
    """
    ReluGrad 内核: 应用 ReLU 反向掩码
    output = input * (relu_in > 0)
    
    用于 Vec2 步骤。
    
    输入1: (M, K) - relu_grad_in (待应用的梯度)
    输入2: (M, K) - relu_in (用于掩码生成的 ReLU 输入)
    输出: (M, K) - relu_grad_out
    """
    
    m_num = M // block_m
    k_num = K // block_k
    
    @T.prim_func
    def main(
        relu_grad_in: T.Tensor((M, K), dtype),
        relu_in: T.Tensor((M, K), dtype),
        relu_grad_out: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(m_num * k_num, threads=2, is_npu=True) as (cid):
            m_idx = cid // k_num
            k_idx = cid % k_num
            
            # 分配 UB 缓冲区
            grad_ub = T.alloc_ub((block_m, block_k), dtype)
            in_ub = T.alloc_ub((block_m, block_k), dtype)
            out_ub = T.alloc_ub((block_m, block_k), dtype)
            # Ascend AI Core 不支持 half 精度比较运算，需要 float32 中间缓冲区
            in_fp32_ub = T.alloc_ub((block_m, block_k), 'float32')
            out_fp32_ub = T.alloc_ub((block_m, block_k), 'float32')

            # 加载输入分块
            T.copy(relu_grad_in[m_idx * block_m, k_idx * block_k], grad_ub)
            T.copy(relu_in[m_idx * block_m, k_idx * block_k], in_ub)

            # Cast to float32 for comparison
            for i, j in T.Parallel(block_m, block_k):
                in_fp32_ub[i, j] = T.cast(in_ub[i, j], 'float32')

            # 应用 ReLU 反向: output = input * (relu_in > 0)，在 float32 下比较
            for i, j in T.Parallel(block_m, block_k):
                if in_fp32_ub[i, j] > T.float32(0):
                    out_fp32_ub[i, j] = T.cast(grad_ub[i, j], 'float32')
                else:
                    out_fp32_ub[i, j] = T.float32(0)

            # Cast back to original dtype
            for i, j in T.Parallel(block_m, block_k):
                out_ub[i, j] = T.cast(out_fp32_ub[i, j], dtype)

            # 写入输出
            T.copy(out_ub, relu_grad_out[m_idx * block_m, k_idx * block_k])
    
    return main


@jit(out_idx=[-1], pass_configs=pass_configs)
def scatter_add_kernel(
    K: int, D: int,
    block_k: int, block_d: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    ScatterAdd 内核，使用原子加法
    将梯度从 (K, D) 散射到索引位置的 dKey
    
    注意: 这是简化版本。完整散射需要索引处理。
    实际实现使用 T.tile.atomic_add 配合适当的偏移量。
    
    输入: (K, D) - scatter_add 值
    索引: (K,) - 散射位置的索引 (单独传递)
    输出: dKey 在索引位置累加
    """
    
    k_num = K // block_k
    d_num = D // block_d
    
    @T.prim_func
    def main(
        scatter_add: T.Tensor((K, D), dtype),
        indices: T.Tensor((K,), 'int32'),
        d_key: T.Tensor((S2, D), dtype),  # S2 = seqlen_k
    ):
        with T.Kernel(k_num * d_num, threads=2, is_npu=True) as (cid):
            k_idx = cid // d_num
            d_idx = cid % d_num
            
            # 分配 UB 缓冲区
            scatter_ub = T.alloc_ub((block_k, block_d), dtype)
            scatter_acc_ub = T.alloc_ub((block_k, block_d), accum_dtype)
            
            # 加载 scatter_add 分块
            T.copy(scatter_add[k_idx * block_k, d_idx * block_d], scatter_ub)
            
            # 转换为累加数据类型用于原子加
            for i, j in T.Parallel(block_k, block_d):
                scatter_acc_ub[i, j] = T.cast(scatter_ub[i, j], accum_dtype)
            
            # 对分块中的每个索引，原子加到 dKey
            # 这是简化版 - 实际实现需要索引迭代
            for i in T.serial(block_k):
                # 读取索引值 (简化版)
                # idx = indices[k_idx * block_k + i]
                
                # 原子加到 dKey[idx, d_idx * block_d + j]
                # 这需要动态偏移计算
                # T.tile.atomic_add(d_key[idx * D + d_idx * block_d], scatter_acc_ub[i, :])
                pass
            
            # 简化版: 直接复制用于演示
            # 实际散射需要正确的基于索引的原子加
            T.copy(scatter_ub, d_key[k_idx * block_k, d_idx * block_d])
    
    return main


# ============================================================
# Python 辅助函数用于 Gather 操作
# ============================================================

def gather_topk_python(key_cpu, indices_cpu, n2_idx, head_dim):
    """
    使用 Python/NumPy 基于 TopK 索引收集 Key 数据。
    
    在 CPU 上运行，生成可传输到 NPU 的张量，
    用于后续 MatMul 操作。
    
    参数:
        key_cpu: CPU 上的 Key 张量, 形状 (batch, seqlen_k, head_num_k, head_dim)
        indices_cpu: CPU 上的索引张量, 形状 (batch, seqlen_q, topk)
        n2_idx: Key 头索引
        head_dim: 头维度
    
    返回:
        key_gather: 收集的 Key 数据, 形状 (batch, seqlen_q, topk, head_dim)
    """
    batch = key_cpu.shape[0]
    seqlen_q = indices_cpu.shape[1]
    topk = indices_cpu.shape[2]
    
    # 初始化输出
    key_gather = np.zeros((batch, seqlen_q, topk, head_dim), dtype=key_cpu.dtype)
    
    # Gather 操作
    for b in range(batch):
        for s in range(seqlen_q):
            for k in range(topk):
                idx = indices_cpu[b, s, k]
                key_gather[b, s, k, :] = key_cpu[b, idx, n2_idx, :]
    
    return key_gather


def scatter_add_python(d_key_cpu, scatter_add_cpu, indices_cpu, n2_idx, head_dim):
    """
    使用 Python/NumPy 将梯度散射到 Key 位置。
    
    在 CPU 上运行以获得确定性结果，或可在 NPU 上使用原子加。
    
    参数:
        d_key_cpu: CPU 上的 dKey 梯度张量, 形状 (batch, seqlen_k, head_num_k, head_dim)
        scatter_add_cpu: CPU 上的散射值, 形状 (batch, seqlen_q, topk, head_dim)
        indices_cpu: CPU 上的索引张量, 形状 (batch, seqlen_q, topk)
        n2_idx: Key 头索引
        head_dim: 头维度
    
    返回:
        d_key_cpu: 更新后的 dKey，包含散射的梯度
    """
    batch = scatter_add_cpu.shape[0]
    seqlen_q = scatter_add_cpu.shape[1]
    topk = scatter_add_cpu.shape[2]
    
    # Scatter add 操作
    for b in range(batch):
        for s in range(seqlen_q):
            for k in range(topk):
                idx = indices_cpu[b, s, k]
                d_key_cpu[b, idx, n2_idx, :] += scatter_add_cpu[b, s, k, :]
    
    return d_key_cpu


# ============================================================
# TileLang 内核用于简单复制操作
# ============================================================

@jit(out_idx=[-1], pass_configs=pass_configs)
def copy_kernel(M: int, N: int, block_m: int, block_n: int, dtype: str = 'float16'):
    """
    简单复制内核，用于数据移动。
    用于加载/存储数据分块。
    """
    
    m_num = M // block_m
    n_num = N // block_n
    
    @T.prim_func
    def main(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            m_idx = cid // n_num
            n_idx = cid % n_num
            
            ub = T.alloc_ub((block_m, block_n), dtype)
            
            T.copy(src[m_idx * block_m, n_idx * block_n], ub)
            T.copy(ub, dst[m_idx * block_m, n_idx * block_n])
    
    return main


# 测试函数

def test_relu_grad():
    M, K = 64, 128
    kernel = relu_grad_kernel(M, K, 64, 128)
    
    relu_grad_in = torch.randn(M, K, dtype=torch.float16).npu()
    relu_in = torch.randn(M, K, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    
    relu_grad_out = kernel(relu_grad_in, relu_in)
    
    torch.npu.synchronize()
    
    # 参考结果
    mask = (relu_in > 0).float()
    ref_out = relu_grad_in * mask.half()
    
    torch.testing.assert_close(relu_grad_out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    print("ReluGrad 内核测试通过!")


if __name__ == "__main__":
    test_relu_grad()