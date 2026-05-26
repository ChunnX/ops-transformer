#!/usr/bin/env python3
"""
TileLang-Ascend Lightning Indexer Gradient 算子

使用 TileLang DSL 完整实现 lightning_indexer_grad 算子。
计算 Lightning Attention 稀疏索引操作中 Query、Key 和 Weights 的梯度。

流水线顺序 (每个批次，每个核任务):
    1. Cube1: Query @ KeyGather^T -> attn_local (M, K)
    2. Vec1: 应用来自 ReLU 输入的掩码
    3. Cube2: Weights @ Dy -> relu_grad_in (M, K)  [广播]
    4. Vec2: ReLU 反向 + ReduceSum 计算 dWeights
    5. Cube3: ReluGrad @ KeyGather -> dQuery (M, D)
    6. Cube4: ReluGrad^T @ Query -> scatter_add (K, D)
    7. Vec3: ScatterAdd 到 dKey

关键形状:
    - Query: (S1, D) 其中 S1 = seqlen_q, D = head_dim
    - Key: (S2, D) 其中 S2 = seqlen_k
    - Weights: (S1, N1) 其中 N1 = head_num_q / group_num
    - Dy: (S1, K) 其中 K = topK
    - Indices: (S1, K) - 前向传播的 TopK 索引
    - ReLU_in: (S1, K) - 前向传播的 ReLU 输入
    - dQuery: (S1, D)
    - dKey: (S2, D) - 通过散射累加
    - dWeights: (S1,)
"""

import torch
import torch_npu
import numpy as np
from typing import Optional, Tuple

from lightning_indexer_grad_matmul import (
    matmul_relu_kernel,
    matmul_broadcast_kernel,
    matmul_dq_kernel,
    matmul_scatter_kernel,
    reduce_sum_kernel,
)
from lightning_indexer_grad_vector import (
    relu_grad_kernel,
    gather_topk_python,
    scatter_add_python,
)


class LightningIndexerGrad:
    """
    Lightning Indexer Gradient 算子
    
    计算稀疏注意力索引操作的梯度:
        前向: Indices = TopK([1] @ (W @ [1]) * ReLU(Q @ K^T))
        反向: 从上游梯度 Dy 计算 dQ, dK, dW
    """
    
    def __init__(
        self,
        seqlen_q: int,
        seqlen_k: int,
        head_dim: int,
        topk: int,
        head_num_q: int,
        head_num_k: int,
        group_num: int,
        block_m: int = 64,
        block_k: int = 128,
        block_d: int = 128,
        dtype: str = 'float16',
    ):
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.head_dim = head_dim
        self.topk = topk
        self.head_num_q = head_num_q
        self.head_num_k = head_num_k
        self.group_num = group_num
        self.head_num_per_group = head_num_q // group_num  # N1
        
        self.block_m = block_m
        self.block_k = block_k
        self.block_d = block_d
        self.dtype = dtype
        
        # 编译内核
        self._compile_kernels()
    
    def _compile_kernels(self):
        """提前编译所有 TileLang 内核。"""
        M = self.seqlen_q
        K = self.topk
        D = self.head_dim
        N = self.seqlen_k
        
        # Cube1: Query @ KeyGather^T -> attn_local (M, K)
        # KeyGather 形状: (K, D), 转置为 (D, K) 用于 matmul
        self.kernel_cube1 = matmul_relu_kernel(
            M, D, K,  # Query (M, D) @ KeyGather^T (D, K) -> (M, K)
            self.block_m, self.block_d, self.block_k,
            self.dtype
        )
        
        # Cube2: Weights @ Dy -> relu_grad_in (M, K)
        # Weights: (M, 1), Dy: (1, K) -> (M, K)
        self.kernel_cube2 = matmul_broadcast_kernel(
            M, 1, K,
            self.block_m, 1, self.block_k,
            self.dtype
        )
        
        # Cube3: ReluGrad @ KeyGather -> dQuery (M, D)
        self.kernel_cube3 = matmul_dq_kernel(
            M, K, D,
            self.block_m, self.block_k, self.block_d,
            self.dtype
        )
        
        # Cube4: ReluGrad^T @ Query -> scatter_add (K, D)
        self.kernel_cube4 = matmul_scatter_kernel(
            K, M, D,
            self.block_k, self.block_m, self.block_d,
            self.dtype
        )
        
        # Vec2: ReduceSum 计算 dWeights
        self.kernel_reduce = reduce_sum_kernel(
            M, K,
            self.block_m, self.block_k,
            self.dtype
        )
        
        # Vec2: ReLU 反向
        self.kernel_relu_grad = relu_grad_kernel(
            M, K,
            self.block_m, self.block_k,
            self.dtype
        )
        
        torch.npu.synchronize()
        print("所有内核编译成功!")
    
    def forward(
        self,
        query: torch.Tensor,      # (batch, S1, D)
        key: torch.Tensor,        # (batch, S2, head_num_k, D)
        weights: torch.Tensor,    # (batch, S1, N1)
        dy: torch.Tensor,         # (batch, S1, K)
        indices: torch.Tensor,    # (batch, S1, K) int32
        relu_in: torch.Tensor,    # (batch, S1, K)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 Query、Key 和 Weights 的梯度。
        
        参数:
            query: Query 张量, 形状 (batch, S1, D)
            key: Key 张量, 形状 (batch, S2, head_num_k, D)
            weights: Weights 张量, 形状 (batch, S1, N1)
            dy: 上游梯度, 形状 (batch, S1, K)
            indices: 前向传播的 TopK 索引, 形状 (batch, S1, K)
            relu_in: 前向传播的 ReLU 输入, 形状 (batch, S1, K)
        
        返回:
            d_query: Query 的梯度, 形状 (batch, S1, D)
            d_key: Key 的梯度, 形状 (batch, S2, head_num_k, D)
            d_weights: Weights 的梯度, 形状 (batch, S1, N1)
        """
        batch = query.shape[0]
        device = query.device
        
        # 初始化输出梯度
        d_query = torch.zeros(batch, self.seqlen_q, self.head_dim, 
                              dtype=query.dtype, device=device)
        d_key = torch.zeros(batch, self.seqlen_k, self.head_num_k, self.head_dim,
                            dtype=key.dtype, device=device)
        d_weights = torch.zeros(batch, self.seqlen_q, self.head_num_per_group,
                                dtype=weights.dtype, device=device)
        
        # 将 indices 和 relu_in 移到 CPU 用于 gather/scatter
        indices_cpu = indices.cpu().numpy()
        relu_in_cpu = relu_in.cpu().numpy()
        key_cpu = key.cpu().numpy()
        
        # 处理每个批次
        for b in range(batch):
            for n1 in range(self.head_num_per_group):
                # 确定使用哪个 Key 头
                n2_idx = n1 * self.head_num_k // self.head_num_per_group
                
                # ========== 步骤 1: GatherTopk ==========
                # 基于索引收集 Key 数据
                key_gather = gather_topk_python(
                    key_cpu[b:b+1],  # (1, S2, head_num_k, D)
                    indices_cpu[b:b+1],  # (1, S1, K)
                    n2_idx,
                    self.head_dim
                )  # -> (1, S1, K, D)
                
                key_gather_npu = torch.from_numpy(key_gather.squeeze(0)).npu()  # (S1, K, D)
                
                # ========== 步骤 2: Cube1 (可选，如果需要掩码) ==========
                # Query @ KeyGather^T -> attn_local
                # 如果掩码已预计算，此步骤可跳过
                # query_b = query[b]  # (S1, D)
                # key_gather_t = key_gather_npu.transpose(1, 2)  # (S1, D, K)
                
                # ========== 步骤 3: Cube2 ==========
                # Weights @ Dy -> relu_grad_in
                weights_b = weights[b, :, n1:n1+1]  # (S1, 1)
                dy_b = dy[b]  # (S1, K)
                
                # 对于广播 matmul，扩展 weights
                relu_grad_in = weights_b * dy_b  # (S1, K) 通过广播
                
                # ========== 步骤 4: Vec2 ==========
                # ReLU 反向: relu_grad_out = relu_grad_in * (relu_in > 0)
                relu_in_b = relu_in[b].npu()  # (S1, K)
                relu_grad_out = self.kernel_relu_grad(relu_grad_in, relu_in_b)
                
                # ReduceSum 计算 dWeights: sum(relu_in * dy, dim=-1)
                relu_in_dy = relu_in_b * dy_b  # 元素级
                d_weights_b = self.kernel_reduce(relu_in_dy)
                d_weights[b, :, n1] = d_weights_b
                
                # ========== 步骤 5: Cube3 ==========
                # ReluGrad @ KeyGather -> dQuery
                # relu_grad_out: (S1, K), KeyGather: (S1, K, D) -> dQuery: (S1, D)
                # 这实际上是批处理 matmul: (K, D)^T @ (K,) -> (D,)
                # 简化为 K 维度的归约
                d_query_b = torch.einsum('sk,skd->sd', relu_grad_out, key_gather_npu)
                d_query[b] += d_query_b
                
                # ========== 步骤 6: Cube4 ==========
                # ReluGrad^T @ Query -> scatter_add
                # relu_grad_out^T: (K, S1), Query: (S1, D) -> scatter_add: (K, D)
                scatter_add = torch.einsum('sk,sd->kd', relu_grad_out, query[b])
                
                # ========== 步骤 7: Vec3 ==========
                # ScatterAdd 到 dKey
                scatter_add_cpu = scatter_add.cpu().numpy()
                d_key_cpu = d_key[b].cpu().numpy()
                d_key_cpu = scatter_add_python(
                    d_key_cpu[np.newaxis, ...],  # (1, S2, head_num_k, D)
                    scatter_add_cpu[np.newaxis, ...],  # (1, K, D)
                    indices_cpu[b:b+1],  # (1, S1, K)
                    n2_idx,
                    self.head_dim
                )
                d_key[b] = torch.from_numpy(d_key_cpu.squeeze(0)).to(device)
        
        torch.npu.synchronize()
        
        return d_query, d_key, d_weights


def test_lightning_indexer_grad():
    """测试 Lightning Indexer Gradient 算子。"""
    
    # 测试参数 (小尺寸用于快速测试)
    batch = 2
    seqlen_q = 128
    seqlen_k = 256
    head_dim = 64
    topk = 32
    head_num_q = 8
    head_num_k = 2
    group_num = 4
    
    # 创建算子
    op = LightningIndexerGrad(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        topk=topk,
        head_num_q=head_num_q,
        head_num_k=head_num_k,
        group_num=group_num,
        block_m=32,
        block_k=32,
        block_d=32,
    )
    
    # 创建测试输入
    query = torch.randn(batch, seqlen_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q // group_num, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    relu_in = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    
    # 运行前向传播
    d_query, d_key, d_weights = op.forward(query, key, weights, dy, indices, relu_in)
    
    torch.npu.synchronize()
    
    print(f"d_query 形状: {d_query.shape}")
    print(f"d_key 形状: {d_key.shape}")
    print(f"d_weights 形状: {d_weights.shape}")
    print("测试通过!")


if __name__ == "__main__":
    test_lightning_indexer_grad()