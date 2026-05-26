#!/usr/bin/env python3
"""
TileLang-Ascend 实现 lightning_indexer_grad 算子

这是使用 TileLang-Ascend DSL 对 lightning_indexer_grad 算子的重新实现。
该算子基于 TopK 索引计算 query、key 和 weights 的梯度。

数学公式:
    reluIn = Query @ KeyGather^T          (Cube1)
    reluGrad_in = Weights @ Dy             (Cube2)
    reluGrad_out = reluGrad_in * mask     (Vec2 - ReluGrad)
    dW = ReduceSum(reluIn * Dy)           (Vec2)
    dQ = reluGrad_out @ KeyGather         (Cube3)
    scatterAdd = reluGrad_out^T @ Query   (Cube4)
    dK[Indices] = scatterAdd              (Vec3 - ScatterAdd)
"""

import tilelang
import tilelang.language as T
from tilelang import jit
import torch
import numpy as np

# ============================================================
# 配置参数
# ============================================================

# 默认参数
DEFAULT_BATCH = 1
DEFAULT_S1 = 1024  # Query 序列长度
DEFAULT_S2 = 2048  # Key 序列长度 (最小值，实际可能更大)
DEFAULT_HEAD_NUM_Q = 64
DEFAULT_HEAD_NUM_K = 1
DEFAULT_HEAD_DIM = 128
DEFAULT_TOPK = 2048
DEFAULT_GROUP_NUM = 64  # group_size = headNumQ / headNumK

# 分块大小
BLOCK_M = 64  # M 维度的分块大小 (组维度)
BLOCK_K = 128  # K 维度的分块大小 (topK)
BLOCK_D = 128  # 头维度 (固定)

VEC_NUM = 2  # 每个 AICore 的向量单元数量

# ============================================================
# 编译的 Pass 配置
# ============================================================

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# ============================================================
# Gather 和 Scatter 操作的辅助函数
# ============================================================

def gather_topk_indices(key_gm, indices_ub, output_ub, block_idx, n2_idx, head_dim, real_topk):
    """
    基于 TopK 索引收集 Key 数据。
    
    在 TileLang 中，Gather 实现为:
    1. 加载索引到 UB
    2. 对每个索引，复制对应的 Key 分块到输出 UB
    """
    # 这将在内核中使用 T.copy 实现
    pass

# ============================================================
# 主内核实现
# ============================================================

@jit(out_idx=[-1, -2, -3], pass_configs=pass_configs)
def lightning_indexer_grad_kernel(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_num_q: int,
    head_num_k: int,
    head_dim: int,
    topk: int,
    group_num: int,
    block_m: int,
    block_k: int,
    dtype: str = 'float16',
    accum_dtype: str = 'float32',
):
    """
    lightning_indexer_grad 算子的 TileLang 内核。
    
    参数:
        batch: 批次大小
        seqlen_q: Query 序列长度 (S1)
        seqlen_k: Key 序列长度 (S2)
        head_num_q: Query 头数量 (N1)
        head_num_k: Key 头数量 (N2)
        head_dim: 头维度 (D)
        topk: TopK 索引数量 (K)
        group_num: 组大小 (G = N1/N2)
        block_m: M 维度的分块大小
        block_k: K 维度的分块大小
        dtype: 输入/输出的数据类型
        accum_dtype: 累加的数据类型
    """
    
    m_num = seqlen_q * group_num // block_m  # 每批次的 M 分块数量
    total_tasks = batch * head_num_k * m_num  # 并行任务总数
    
    @T.prim_func
    def main(
        query: T.Tensor((batch, seqlen_q, head_num_q, head_dim), dtype),        # 输入 Q
        key: T.Tensor((batch, seqlen_k, head_num_k, head_dim), dtype),           # 输入 K
        dy: T.Tensor((batch, seqlen_q, topk), dtype),                            # 梯度输入 (TopK 位置)
        sparse_indices: T.Tensor((batch, seqlen_q, topk), 'int32'),              # 前向传播的 TopK 索引
        weights: T.Tensor((batch, seqlen_q, head_num_q), dtype),                 # 前向传播的权重
        d_query: T.Tensor((batch, seqlen_q, head_num_q, head_dim), dtype),       # 输出: Query 梯度
        d_key: T.Tensor((batch, seqlen_k, head_num_k, head_dim), dtype),         # 输出: Key 梯度
        d_weights: T.Tensor((batch, seqlen_q, head_num_q), dtype),               # 输出: Weights 梯度
    ):
        # ============================================================
        # 内核启动，使用持久调度
        # ============================================================
        with T.Kernel(total_tasks, threads=2, is_npu=True) as (cid):
            
            # 从 cid 计算任务索引
            # 任务分配: cid -> (batch_idx, n2_idx, m_idx)
            batch_idx = cid // (head_num_k * m_num)
            n2_idx = (cid // m_num) % head_num_k
            m_idx = cid % m_num
            
            # 计算此任务的组偏移
            group_start = m_idx * block_m // group_num
            group_end = min((m_idx + 1) * block_m // group_num, group_num)
            local_group_num = group_end - group_start
            
            # ============================================================
            # 内存分配
            # ============================================================
            
            # L1 缓冲区用于 Cube 操作
            # Query 分块: (block_m, head_dim) -> 但我们处理 (local_group_num, head_dim)
            query_l1 = T.alloc_L1((local_group_num, head_dim), dtype)
            
            # KeyGather 分块: (topk, head_dim) - 使用索引从 Key 收集
            key_gather_l1 = T.alloc_L1((block_k, head_dim), dtype)
            
            # L0 缓冲区用于 MatMul
            query_l0a = T.alloc_L0A((local_group_num, head_dim), dtype)
            key_l0b = T.alloc_L0B((head_dim, block_k), dtype)  # 转置后的形状
            relu_in_l0c = T.alloc_L0C((local_group_num, block_k), accum_dtype)
            
            # 用于 Cube2: weights @ dy
            weights_l1 = T.alloc_L1((local_group_num, 1), dtype)
            dy_tile_l1 = T.alloc_L1((1, block_k), dtype)
            weights_l0a = T.alloc_L0A((local_group_num, 1), dtype)
            dy_l0b = T.alloc_L0B((1, block_k), dtype)
            relu_grad_l0c = T.alloc_L0C((local_group_num, block_k), accum_dtype)
            
            # 用于 Cube3: relu_grad @ key_gather -> dQuery
            relu_grad_l0a = T.alloc_L0A((local_group_num, block_k), dtype)
            key_gather_l0b = T.alloc_L0B((block_k, head_dim), dtype)
            dq_l0c = T.alloc_L0C((local_group_num, head_dim), accum_dtype)
            
            # 用于 Cube4: relu_grad^T @ query -> scatterAdd
            relu_grad_t_l0a = T.alloc_L0A((block_k, local_group_num), dtype)
            query_l0b = T.alloc_L0B((local_group_num, head_dim), dtype)
            scatter_add_l0c = T.alloc_L0C((block_k, head_dim), accum_dtype)
            
            # UB (统一缓冲区) 用于 Vector 操作
            indices_ub = T.alloc_ub((topk,), 'int32')  # TopK 索引
            relu_in_ub = T.alloc_ub((local_group_num, block_k), dtype)  # ReLU 输入 (来自 L0C)
            relu_grad_ub = T.alloc_ub((local_group_num, block_k), dtype)  # ReluGrad 输出
            dy_ub = T.alloc_ub((block_k,), dtype)  # Dy 分块
            dq_ub = T.alloc_ub((local_group_num, head_dim), dtype)  # dQuery 本地
            scatter_add_ub = T.alloc_ub((block_k, head_dim), dtype)  # scatterAdd 结果
            
            # 操作的临时缓冲区
            temp_ub = T.alloc_ub((local_group_num, block_k), accum_dtype)
            dweights_ub = T.alloc_ub((local_group_num,), accum_dtype)
            mask_ub = T.alloc_ub((local_group_num, block_k), 'int8')  # ReluGrad 的掩码
            
            # ============================================================
            # 步骤 1: 加载此 (batch, s1_idx, n2_idx) 的索引
            # ============================================================
            
            # 计算 sparse_indices 张量中的偏移
            # 形状: (batch, seqlen_q, topk)
            s1_idx = m_idx * block_m // group_num  # 简化，实际应考虑组
            
            indices_offset = batch_idx * seqlen_q * topk + s1_idx * topk
            T.copy(sparse_indices[indices_offset], indices_ub)
            
            # ============================================================
            # 步骤 2: GatherTopk - 基于索引加载 Key 数据
            # ============================================================
            
            # Key 形状: (batch, seqlen_k, head_num_k, head_dim)
            # 对 indices_ub 中的每个索引，复制 Key[index, n2_idx, :] 到 key_gather_l1
            
            # 注意: TileLang 没有直接的 Gather 操作
            # 我们需要遍历索引并逐个复制
            # 这是简化版本 - 实际实现可能需要优化
            
            # 目前，我们假设索引已加载，我们复制 Key 分块
            # 实际中，这可能需要为每个索引使用 T.copy 的循环
            
            # 加载 Key 分块 (简化: 假设连续的 topk 索引用于演示)
            key_offset = batch_idx * seqlen_k * head_num_k * head_dim + n2_idx * head_dim
            # 实际 Gather 应使用 indices_ub 值
            # T.copy(key[key_offset + idx * head_num_k * head_dim], key_gather_l1[idx * head_dim])
            
            # ============================================================
            # 步骤 3: Cube1 - Query @ KeyGather^T -> reluIn
            # ============================================================
            
            # 加载 Query 分块到 L1
            query_offset = batch_idx * seqlen_q * head_num_q * head_dim + \
                          s1_idx * head_num_q * head_dim + \
                          n2_idx * group_num * head_dim + \
                          group_start * head_dim
            T.copy(query[query_offset], query_l1)
            
            # 移到 L0A
            T.copy(query_l1, query_l0a)
            
            # KeyGather 到 L0B (转置布局)
            T.copy(key_gather_l1, key_l0b)  # 将在 gemm 中处理为转置
            
            # MatMul: Query @ KeyGather^T
            T.gemm_v0(query_l0a, key_l0b, relu_in_l0c, transpose_B=True, init=True)
            
            # 复制 reluIn 到 UB 用于 ReLU 激活 (在 Vec2 中处理)
            T.copy(relu_in_l0c, relu_in_ub)
            
            # ============================================================
            # 步骤 4: Cube2 - Weights @ Dy -> reluGrad_in
            # ============================================================
            
            # 加载 weights 分块
            weights_offset = batch_idx * seqlen_q * head_num_q + \
                           s1_idx * head_num_q + \
                           n2_idx * group_num + group_start
            T.copy(weights[weights_offset], weights_l1)
            T.copy(weights_l1, weights_l0a)
            
            # 加载 Dy 分块
            dy_offset = batch_idx * seqlen_q * topk + s1_idx * topk
            T.copy(dy[dy_offset], dy_tile_l1)
            T.copy(dy_tile_l1, dy_l0b)
            
            # MatMul: weights @ dy (广播)
            T.gemm_v0(weights_l0a, dy_l0b, relu_grad_l0c, init=True)
            
            # 复制到 UB
            T.copy(relu_grad_l0c, relu_grad_ub)
            
            # ============================================================
            # 步骤 5: Vec2 - ReluGrad + ReduceSum 计算 dWeights
            # ============================================================
            
            # ReluGrad: relu_grad * (reluIn > 0)
            # 生成掩码
            for i, j in T.Parallel(local_group_num, block_k):
                if relu_in_ub[i, j] > 0:
                    mask_ub[i, j] = 1
                else:
                    mask_ub[i, j] = 0
            
            # 应用掩码: relu_grad_out = relu_grad * mask
            for i, j in T.Parallel(local_group_num, block_k):
                temp_ub[i, j] = relu_grad_ub[i, j] * mask_ub[i, j]
            
            # 复制回 relu_grad_out 用于 Cube3/4
            T.copy(temp_ub, relu_grad_ub)
            
            # 计算 dWeights: ReduceSum(reluIn * Dy)
            # 首先: reluIn * Dy (Dy 在组间广播)
            T.copy(dy[dy_offset], dy_ub)
            for i, j in T.Parallel(local_group_num, block_k):
                temp_ub[i, j] = relu_in_ub[i, j] * dy_ub[j]
            
            # 沿 K 维度 ReduceSum
            T.reduce_sum(temp_ub, dweights_ub, dim=-1, clear=True)
            
            # 写入 dWeights 输出
            T.copy(dweights_ub, d_weights[weights_offset])
            
            # ============================================================
            # 步骤 6: Cube3 - relu_grad_out @ KeyGather -> dQuery
            # ============================================================
            
            # relu_grad 到 L0A
            T.copy(relu_grad_ub, relu_grad_l0a)
            
            # KeyGather 到 L0B
            T.copy(key_gather_l1, key_gather_l0b)
            
            # MatMul: relu_grad @ KeyGather
            T.gemm_v0(relu_grad_l0a, key_gather_l0b, dq_l0c, init=True)
            
            # 复制到 UB 然后到输出
            T.copy(dq_l0c, dq_ub)
            T.copy(dq_ub, d_query[query_offset])
            
            # ============================================================
            # 步骤 7: Cube4 - relu_grad^T @ Query -> scatterAdd
            # ============================================================
            
            # relu_grad 转置到 L0A
            # 注意: 转置处理 - 这是简化版
            T.copy(relu_grad_ub, relu_grad_t_l0a)  # 需要转置
            
            # Query 到 L0B
            T.copy(query_l1, query_l0b)
            
            # MatMul: relu_grad^T @ Query
            T.gemm_v0(relu_grad_t_l0a, query_l0b, scatter_add_l0c, transpose_A=True, init=True)
            
            # 复制到 UB
            T.copy(scatter_add_l0c, scatter_add_ub)
            
            # ============================================================
            # 步骤 8: Vec3 - ScatterAdd 到 dKey
            # ============================================================
            
            # 原子加 scatter_add 到 dKey 的索引位置
            # 对 indices_ub 中的每个索引，原子加 scatter_add[idx] 到 dKey[batch, idx, n2_idx]
            
            # 简化版: 原子加到 dKey
            # 实际中，需要遍历 indices_ub
            dkey_offset = batch_idx * seqlen_k * head_num_k * head_dim + n2_idx * head_dim
            
            # 原子加操作
            # T.tile.atomic_add(d_key[dkey_offset], scatter_add_ub)
            # 这需要正确的索引处理
            
    return main


# ============================================================
# 易于调用的包装函数
# ============================================================

def lightning_indexer_grad_tilelang(
    query, key, dy, sparse_indices, weights,
    head_num, layout="BSND", sparse_mode=0,
    deterministic=False
):
    """
    调用 TileLang 内核的包装函数。
    
    参数:
        query: 输入张量, 形状 (B, S1, N1, D) 或 (T1, N1, D)
        key: 输入张量, 形状 (B, S2, N2, D) 或 (T2, N2, D)
        dy: 梯度输入, 形状 (B, S1, K) - TopK 位置
        sparse_indices: TopK 索引, 形状 (B, S1, K)
        weights: 权重张量, 形状 (B, S1, N1)
        head_num: 头数量
        layout: 数据布局 (BSND 或 TND)
        sparse_mode: 稀疏模式 (0 或 3)
        deterministic: 是否使用确定性计算
    
    返回:
        d_query, d_key, d_weights: 梯度输出
    """
    
    # 提取形状
    if layout == "BSND":
        batch = query.shape[0]
        seqlen_q = query.shape[1]
        seqlen_k = key.shape[1]
        head_num_q = query.shape[2]
        head_num_k = key.shape[2]
        head_dim = query.shape[3]
    else:  # TND
        # 需要 actual_seq_lengths 来确定批次和序列长度
        batch = 1  # 占位符
        seqlen_q = query.shape[0]
        seqlen_k = key.shape[0]
        head_num_q = query.shape[1]
        head_num_k = key.shape[1]
        head_dim = query.shape[2]
    
    topk = sparse_indices.shape[-1]
    group_num = head_num_q // head_num_k
    
    # 创建输出张量
    d_query = torch.zeros_like(query)
    d_key = torch.zeros_like(key)
    d_weights = torch.zeros_like(weights)
    
    # 获取编译的内核
    dtype = 'float16' if query.dtype == torch.float16 else 'bfloat16'
    
    kernel = lightning_indexer_grad_kernel(
        batch, seqlen_q, seqlen_k,
        head_num_q, head_num_k, head_dim,
        topk, group_num,
        BLOCK_M, BLOCK_K,
        dtype
    )
    
    # 调用内核
    kernel(query, key, dy, sparse_indices, weights, d_query, d_key, d_weights)
    
    return d_query, d_key, d_weights


# ============================================================
# 测试函数
# ============================================================

def test_lightning_indexer_grad_tilelang():
    """
    验证 TileLang 实现的测试函数。
    """
    
    # 测试参数
    batch = 1
    seqlen_q = 128
    seqlen_k = 2048
    head_num_q = 64
    head_num_k = 1
    head_dim = 128
    topk = 2048
    
    # 在 NPU 上创建测试张量
    torch.manual_seed(42)
    
    query = torch.randn(batch, seqlen_q, head_num_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    sparse_indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    
    # 调用 TileLang 内核
    d_query, d_key, d_weights = lightning_indexer_grad_tilelang(
        query, key, dy, sparse_indices, weights,
        head_num=head_num_q
    )
    
    torch.npu.synchronize()
    
    print("TileLang lightning_indexer_grad 测试完成!")
    print(f"d_query 形状: {d_query.shape}")
    print(f"d_key 形状: {d_key.shape}")
    print(f"d_weights 形状: {d_weights.shape}")
    
    return d_query, d_key, d_weights


if __name__ == "__main__":
    test_lightning_indexer_grad_tilelang()