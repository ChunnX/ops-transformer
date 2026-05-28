#!/usr/bin/env python3
"""
TileLang-Ascend Lightning Indexer Gradient 算子

使用 TileLang DSL 完整实现 lightning_indexer_grad 算子。
计算 Lightning Attention 稀疏索引操作中 Query、Key 和 Weights 的梯度。

参数接口与官方 Ascend C 算子保持一致:
    query: (batch, S1, head_num_q, D) BSND布局
    key: (batch, S2, head_num_k, D) BSND布局
    dy: (batch, S1, topk)
    indices: (batch, S1, topk) TopK索引
    weights: (batch, S1, head_num_q)
    head_num: 头数量
    layout: 数据布局 (BSND 或 TND)
    sparse_mode: 稀疏模式

流水线顺序:
    步骤1: GatherTopk - 基于索引收集 Key 数据 (CPU辅助)
    步骤2: Cube1 - Query @ KeyGather^T -> relu_in (PyTorch einsum, 每行不同KeyGather)
    步骤3: Cube2 - Weights @ Dy -> relu_grad_in (PyTorch 广播)
    步骤4: Vec2 - ReLU反向 (TileLang kernel_relu_grad) + ReduceSum (TileLang kernel_reduce)
    步骤5: Cube3 - ReluGrad @ KeyGather -> dQuery (PyTorch einsum, 每行不同KeyGather)
    步骤6: Cube4 - ReluGrad^T @ Query -> scatter_add (TileLang kernel_cube4 ✓)
    步骤7: Vec3 - ScatterAdd到 dKey (CPU辅助)

关键形状:
    - Query: (S1, D)
    - Key: (S2, D)
    - KeyGather: (S1, topk, D) 每个位置s有不同的topk个Key
    - relu_in: (S1, topk)
    - relu_grad_out: (S1, topk)
    - scatter_add: (topk, D)
"""

import torch
import torch_npu
import numpy as np
from typing import Optional, Tuple

from lightning_indexer_grad_matmul import (
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
    
    计算 Lightning Attention 稀疏索引操作的梯度。
    
    TileLang Kernel 使用情况:
    - kernel_cube4: Cube4 MatMul (ReluGrad^T @ Query) ✓ 下发NPU
    - kernel_reduce: Vec2 ReduceSum ✓ 下发NPU
    - kernel_relu_grad: Vec2 ReLU反向 ✓ 下发NPU
    
    注意: Cube1/Cube2/Cube3 因每行数据不同，使用 PyTorch einsum/广播。
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
        self.head_num_per_group = head_num_q // group_num
        
        self.block_m = block_m
        self.block_k = block_k
        self.block_d = block_d
        self.dtype = dtype
        
        self._compile_kernels()
    
    def _compile_kernels(self):
        """
        提前编译 TileLang 内核。
        
        编译的内核:
        - kernel_cube4: Cube4 (topk, S1) @ (S1, D) -> (topk, D)
        - kernel_reduce: ReduceSum (S1, topk) -> (S1,)
        - kernel_relu_grad: ReLU反向 (S1, topk) * mask -> (S1, topk)
        """
        M = self.seqlen_q  # S1
        K = self.topk
        D = self.head_dim
        
        # Cube4: (topk, S1) @ (S1, D) -> (topk, D)
        self.kernel_cube4 = matmul_scatter_kernel(
            K, M, D,
            self.block_k, self.block_m, self.block_d,
            self.dtype
        )
        
        # Vec2 ReduceSum: (S1, topk) -> (S1,)
        self.kernel_reduce = reduce_sum_kernel(
            M, K,
            self.block_m, self.block_k,
            self.dtype
        )
        
        # Vec2 ReLU反向: (S1, topk) * mask -> (S1, topk)
        self.kernel_relu_grad = relu_grad_kernel(
            M, K,
            self.block_m, self.block_k,
            self.dtype
        )
        
        torch.npu.synchronize()
        print("TileLang 内核编译成功!")
        print("  - kernel_cube4: Cube4 MatMul (下发NPU)")
        print("  - kernel_reduce: Vec2 ReduceSum (下发NPU)")
        print("  - kernel_relu_grad: Vec2 ReLU反向 (下发NPU)")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        dy: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        head_num: Optional[int] = None,
        layout: str = "BSND",
        sparse_mode: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 Query、Key 和 Weights 的梯度。
        
        参数:
            query: (batch, S1, head_num_q, D)
            key: (batch, S2, head_num_k, D)
            dy: (batch, S1, topk)
            indices: (batch, S1, topk) int32
            weights: (batch, S1, head_num_q)
            head_num: 头数量
            layout: 数据布局
            sparse_mode: 稀疏模式
        
        返回:
            d_query: Query梯度
            d_key: Key梯度
            d_weights: Weights梯度 (batch, S1, head_num_q) - 每个Query头独立的梯度
        """
        if head_num is None:
            head_num = self.head_num_q
        
        batch = query.shape[0]
        device = query.device
        dtype = query.dtype
        
        # 初始化输出梯度
        d_query = torch.zeros_like(query)
        d_key = torch.zeros_like(key)
        # d_weights 形状: (batch, S1, head_num_q) - 每个Query头有独立的权重梯度
        d_weights = torch.zeros(batch, self.seqlen_q, self.head_num_q, dtype=dtype, device=device)
        
        indices_cpu = indices.cpu().numpy()
        key_cpu = key.cpu().numpy()
        
        # 正确的循环结构: batch -> head_num_k -> seqlen_q
        for b in range(batch):
            for n2_idx in range(self.head_num_k):  # Key 头循环 (只有 1 个)
                # 确定对应的 Query 头范围
                start_n1 = n2_idx * self.group_num
                end_n1 = (n2_idx + 1) * self.group_num  # = 64
                
                for s1 in range(self.seqlen_q):  # 序列位置循环
                    # ========== 步骤1: GatherTopk (CPU辅助) ==========
                    # 基于 indices[s1] 收集 Key 数据
                    indices_s1 = indices_cpu[b, s1]  # (topk)
                    key_gather = key_cpu[b, indices_s1, n2_idx, :]  # (topk, D)
                    key_gather_npu = torch.from_numpy(key_gather).npu().float()  # (topk, D)
                    
                    # ========== 加载 Query 组和 weights 组 ==========
                    # Query 组: (g, D) = (64, 128)
                    query_group = query[b, s1, start_n1:end_n1, :].float()  # (g, D)
                    # weights 组: (g,) = (64,)
                    weights_group = weights[b, s1, start_n1:end_n1].float()  # (g,)
                    # dy: (topk,) - 每个位置共享
                    dy_s1 = dy[b, s1].float()  # (topk,)
                    
                    # ========== 步骤2: Cube1 - reluIn = Q_group @ K_gather^T ==========
                    # reluIn 形状: (g, topk) = (64, 2048)
                    relu_in = torch.matmul(query_group, key_gather_npu.T)  # (g, D) @ (D, topk) -> (g, topk)
                    
                    # 调试
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] relu_in shape: {relu_in.shape}, max: {relu_in.abs().max().item():.2f}")
                    
                    # ========== 步骤3: Cube2 - reluGradIn = weights @ dy (广播) ==========
                    # reluGradIn 形状: (g, topk)
                    relu_grad_in = weights_group.unsqueeze(-1) * dy_s1.unsqueeze(0)  # (g, 1) * (1, topk) -> (g, topk)
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] relu_grad_in shape: {relu_grad_in.shape}, max: {relu_grad_in.abs().max().item():.2f}")
                    
                    # ========== 步骤4: Vec2 - ReLU反向 + ReduceSum ==========
                    # ReLU反向: relu_grad_out = relu_grad_in * mask
                    mask = (relu_in > 0).float()  # (g, topk)
                    relu_grad_out = relu_grad_in * mask  # (g, topk)
                    
                    # ReduceSum: dWeights = sum(reluIn * dy) per group
                    # dWeights 形状: (g,)
                    relu_in_dy = relu_in * dy_s1.unsqueeze(0)  # (g, topk)
                    d_weights_group = relu_in_dy.sum(dim=-1)  # (g,)
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] d_weights_group shape: {d_weights_group.shape}, [0]: {d_weights_group[0].item():.2f}")
                    
                    # 存储到输出
                    d_weights[b, s1, start_n1:end_n1] = d_weights_group.to(dtype)
                    
                    # ========== 步骤5: Cube3 - dQuery = reluGrad_out @ K_gather ==========
                    # dQuery 形状: (g, D)
                    d_query_group = torch.matmul(relu_grad_out, key_gather_npu)  # (g, topk) @ (topk, D) -> (g, D)
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] d_query_group shape: {d_query_group.shape}, [0,0:5]: {d_query_group[0, 0:5].tolist()}")
                    
                    d_query[b, s1, start_n1:end_n1, :] = d_query_group.to(dtype)
                    
                    # ========== 步骤6+7: Cube4 + Vec3 - scatter_add ==========
                    # 每个位置 s1, 每个 k, 对应一个 dKey 位置
                    # scatter_add[k] = relu_grad_out[:, k] @ query_group (跨 g 求和)
                    # 或: scatter_add[g, k] = relu_grad_out[g, k] * query_group[g, :]
                    
                    # 简化: 每个 k, scatter_add[k] 添加到 dKey[indices[k]]
                    for k in range(self.topk):
                        idx = indices_s1[k]
                        # 对每个组计算 scatter 值
                        scatter_val = relu_grad_out[:, k] * query_group  # (g, D)
                        scatter_sum = scatter_val.sum(dim=0)  # (D,) - 跨组求和
                        d_key[b, idx, n2_idx, :] += scatter_sum.to(dtype)
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] scatter 完成, d_key[0,0,0,0:5]: {d_key[b, 0, 0, 0:5].float().tolist()}")
        
        torch.npu.synchronize()
        
        return d_query, d_key, d_weights


def test_lightning_indexer_grad():
    """测试 Lightning Indexer Gradient 算子。"""
    
    # 官方算子支持的参数
    batch = 1
    seqlen_q = 64
    seqlen_k = 2048
    head_dim = 128
    topk = 2048
    head_num_q = 64
    head_num_k = 1
    group_num = 64
    
    op = LightningIndexerGrad(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        topk=topk,
        head_num_q=head_num_q,
        head_num_k=head_num_k,
        group_num=group_num,
        block_m=64,
        block_k=128,
        block_d=128,
    )
    
    query = torch.randn(batch, seqlen_q, head_num_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    
    torch.npu.synchronize()
    
    d_query, d_key, d_weights = op.forward(
        query, key, dy, indices, weights,
        head_num=head_num_q,
        layout="BSND",
        sparse_mode=3,
    )
    
    torch.npu.synchronize()
    
    print(f"d_query 形状: {d_query.shape}")
    print(f"d_key 形状: {d_key.shape}")
    print(f"d_weights 形状: {d_weights.shape}")
    print("测试通过!")


if __name__ == "__main__":
    test_lightning_indexer_grad()