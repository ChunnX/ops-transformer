#!/usr/bin/env python3
"""
TileLang-Ascend MatMul 内核 - 用于 lightning_indexer_grad

本模块为 lightning_indexer_grad 的核心计算提供优化的 MatMul 内核:
- Cube1: Query @ KeyGather^T
- Cube2: Weights @ Dy
- Cube3: ReluGrad @ KeyGather
- Cube4: ReluGrad^T @ Query

这些内核可作为完整算子的构建模块。
"""

import tilelang
import tilelang.language as T
from tilelang import jit
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

BLOCK_M = 64
BLOCK_K = 128
BLOCK_N = 128
VEC_NUM = 2


@jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_kernel(
    M: int, K: int, N: int,
    block_m: int, block_k: int, block_n: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    MatMul 内核: A @ B^T (不带激活函数)
    用于 Cube1: Query @ KeyGather^T
    
    A: (M, K) - Query 分块 (FP16)
    B: (N, K) - KeyGather (将被转置) (FP16)
    输出: (M, N) - FP16
    
    Expert 模式内存分配:
        - gemm_v0 输入必须是 L1 (shared)
        - gemm_v0 输出必须是 L0C (fragment)
        - 编译器自动处理 L1→L0A/L0B 搬运
    
    内存对齐要求:
        - Ascend 要求内存地址对齐
        - block 大小建议为 16 或 32 的倍数
    """
    
    m_num = M // block_m
    k_num = K // block_k
    n_num = N // block_n
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=VEC_NUM, is_npu=True) as (cid):
            m_idx = cid // n_num
            n_idx = cid % n_num
            
            # Expert 模式: 使用 alloc_L1 和 alloc_L0C
            A_L1 = T.alloc_L1((block_m, block_k), dtype)
            B_L1 = T.alloc_L1((block_n, block_k), dtype)
            C_L0C = T.alloc_L0C((block_m, block_n), accum_dtype)
            
            for i, j in T.Parallel(block_m, block_n):
                C_L0C[i, j] = T.cast(0, accum_dtype)
            
            for k_idx in T.Pipelined(k_num, num_stages=2):
                T.copy(A[m_idx * block_m, k_idx * block_k], A_L1)
                T.copy(B[n_idx * block_n, k_idx * block_k], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0C, transpose_B=True, init=(k_idx == 0))
            
            T.copy(C_L0C, C[m_idx * block_m, n_idx * block_n])
    
    return main


@jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_relu_kernel(
    M: int, K: int, N: int,
    block_m: int, block_k: int, block_n: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    MatMul 内核: A @ B^T 并带 ReLU 激活
    
    A: (M, K) - Query 分块 (FP16)
    B: (N, K) - KeyGather (将被转置) (FP16)
    输出: (M, N) 并应用 ReLU (FP16)
    
    Expert 模式内存分配:
        - gemm_v0 输入必须是 L1 (shared)
        - gemm_v0 输出必须是 L0C (fragment)
    """
    
    m_num = M // block_m
    k_num = K // block_k
    n_num = N // block_n
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=VEC_NUM, is_npu=True) as (cid):
            m_idx = cid // n_num
            n_idx = cid % n_num
            
            A_L1 = T.alloc_L1((block_m, block_k), dtype)
            B_L1 = T.alloc_L1((block_n, block_k), dtype)
            C_L0C = T.alloc_L0C((block_m, block_n), accum_dtype)
            
            for i, j in T.Parallel(block_m, block_n):
                C_L0C[i, j] = T.cast(0, accum_dtype)
            
            for k_idx in T.Pipelined(k_num, num_stages=2):
                T.copy(A[m_idx * block_m, k_idx * block_k], A_L1)
                T.copy(B[n_idx * block_n, k_idx * block_k], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0C, transpose_B=True, init=(k_idx == 0))
            
            for i, j in T.Parallel(block_m, block_n):
                C_L0C[i, j] = T.max(C_L0C[i, j], T.cast(0, accum_dtype))
            
            T.copy(C_L0C, C[m_idx * block_m, n_idx * block_n])
    
    return main


@jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_broadcast_kernel(
    M: int, N: int,
    block_m: int, block_n: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    外积内核 (使用 Cube 单元): A(M,1) @ B(1,N) -> C(M,N)
    用于 Cube2: Weights ⊗ Dy
    
    输入:
        A: (M, 1) - Weights reshape
        B: (1, N) - Dy reshape
    输出:
        C: (M, N) - 外积结果
    
    Expert 模式内存分配:
        - gemm_v0 输入必须是 L1 (shared)
        - gemm_v0 输出必须是 L0C (fragment)
    """
    
    m_num = M // block_m
    n_num = N // block_n
    
    @T.prim_func
    def main(
        A: T.Tensor((M, 1), dtype),
        B: T.Tensor((1, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=VEC_NUM, is_npu=True) as (cid):
            m_idx = cid // n_num
            n_idx = cid % n_num
            
            A_L1 = T.alloc_L1((block_m, 1), dtype)
            B_L1 = T.alloc_L1((1, block_n), dtype)
            C_L0C = T.alloc_L0C((block_m, block_n), accum_dtype)
            
            T.copy(A[m_idx * block_m, 0], A_L1)
            T.copy(B[0, n_idx * block_n], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0C, init=True)
            T.copy(C_L0C, C[m_idx * block_m, n_idx * block_n])
    
    return main


@jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_dq_kernel(
    M: int, K: int, D: int,
    block_m: int, block_k: int, block_d: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    MatMul 内核，用于计算 dQuery
    用于 Cube3: ReluGrad @ KeyGather
    
    A: (M, K) - ReluGrad (FP16)
    B: (K, D) - KeyGather (FP16)
    输出: (M, D) - dQuery (FP16)
    
    Expert 模式内存分配:
        - gemm_v0 输入必须是 L1 (shared)
        - gemm_v0 输出必须是 L0C (fragment)
    """
    
    m_num = M // block_m
    k_num = K // block_k
    d_num = D // block_d
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, D), dtype),
        C: T.Tensor((M, D), dtype),
    ):
        with T.Kernel(m_num * d_num, threads=VEC_NUM, is_npu=True) as (cid):
            m_idx = cid // d_num
            d_idx = cid % d_num
            
            A_L1 = T.alloc_L1((block_m, block_k), dtype)
            B_L1 = T.alloc_L1((block_k, block_d), dtype)
            C_L0C = T.alloc_L0C((block_m, block_d), accum_dtype)
            
            for i, j in T.Parallel(block_m, block_d):
                C_L0C[i, j] = T.cast(0, accum_dtype)
            
            for k_idx in T.Pipelined(k_num, num_stages=2):
                T.copy(A[m_idx * block_m, k_idx * block_k], A_L1)
                T.copy(B[k_idx * block_k, d_idx * block_d], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0C, init=(k_idx == 0))
            
            T.copy(C_L0C, C[m_idx * block_m, d_idx * block_d])
    
    return main


@jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_scatter_kernel(
    K: int, M: int, D: int,
    block_k: int, block_m: int, block_d: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    MatMul 内核，用于计算 scatterAdd
    用于 Cube4: ReluGrad^T @ Query
    
    A: (K, M) - ReluGrad 转置后的 (FP16)
    B: (M, D) - Query (FP16)
    输出: (K, D) - scatterAdd (FP16)
    
    Expert 模式内存分配:
        - gemm_v0 输入必须是 L1 (shared)
        - gemm_v0 输出必须是 L0C (fragment)
    """
    
    k_num = K // block_k
    m_num = M // block_m
    d_num = D // block_d
    
    @T.prim_func
    def main(
        A: T.Tensor((K, M), dtype),
        B: T.Tensor((M, D), dtype),
        C: T.Tensor((K, D), dtype),
    ):
        with T.Kernel(k_num * d_num, threads=VEC_NUM, is_npu=True) as (cid):
            k_idx = cid // d_num
            d_idx = cid % d_num
            
            A_L1 = T.alloc_L1((block_k, block_m), dtype)
            B_L1 = T.alloc_L1((block_m, block_d), dtype)
            C_L0C = T.alloc_L0C((block_k, block_d), accum_dtype)
            
            for i, j in T.Parallel(block_k, block_d):
                C_L0C[i, j] = T.cast(0, accum_dtype)
            
            for m_idx in T.Pipelined(m_num, num_stages=2):
                T.copy(A[k_idx * block_k, m_idx * block_m], A_L1)
                T.copy(B[m_idx * block_m, d_idx * block_d], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0C, init=(m_idx == 0))
            
            T.copy(C_L0C, C[k_idx * block_k, d_idx * block_d])
    
    return main


@jit(out_idx=[-1], pass_configs=pass_configs)
def reduce_sum_kernel(
    M: int, K: int,
    block_m: int, block_k: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32'
):
    """
    ReduceSum 内核，用于计算 dWeights
    沿 K 维度归约 (M, K) -> (M,)
    
    Expert 模式内存分配:
        - Vector 操作使用 alloc_ub
        - Ascend Vector Reduce 需要 FP32 精度
    """
    
    m_num = M // block_m
    k_num = K // block_k
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(m_num, threads=VEC_NUM, is_npu=True) as (cid):
            m_idx = cid
            
            A_UB = T.alloc_ub((block_m, block_k), dtype)
            A_fp32_UB = T.alloc_ub((block_m, block_k), accum_dtype)
            C_fp32_UB = T.alloc_ub((block_m,), accum_dtype)
            C_UB = T.alloc_ub((block_m,), dtype)
            
            for i in T.Parallel(block_m):
                C_fp32_UB[i] = T.cast(0, accum_dtype)
            
            for k_idx in T.serial(k_num):
                T.copy(A[m_idx * block_m, k_idx * block_k], A_UB)
                
                for i, j in T.Parallel(block_m, block_k):
                    A_fp32_UB[i, j] = T.cast(A_UB[i, j], accum_dtype)
                
                T.reduce_sum(A_fp32_UB, C_fp32_UB, dim=-1, clear=False)
            
            for i in T.Parallel(block_m):
                C_UB[i] = T.cast(C_fp32_UB[i], dtype)
            
            T.copy(C_UB, C[m_idx * block_m])
    
    return main


# 测试函数

def test_matmul_relu():
    M, K, N = 64, 128, 128
    kernel = matmul_relu_kernel(M, K, N, BLOCK_M, BLOCK_K, BLOCK_N)
    
    A = torch.randn(M, K, dtype=torch.float16).npu()
    B = torch.randn(N, K, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    
    C = kernel(A, B)
    
    torch.npu.synchronize()
    
    # 参考结果
    ref_C = torch.relu(torch.matmul(A, B.transpose(0, 1)))
    
    torch.testing.assert_close(C.float(), ref_C.float(), rtol=1e-2, atol=1e-2)
    print("MatMul ReLU 内核测试通过!")


if __name__ == "__main__":
    test_matmul_relu()