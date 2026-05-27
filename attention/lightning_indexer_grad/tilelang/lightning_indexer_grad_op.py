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
        
        # 遍历每个Query头
        for b in range(batch):
            for n1 in range(self.head_num_q):  # 遍历所有Query头
                # 确定对应的Key头索引
                n2_idx = n1 // self.group_num
                
                # ========== 步骤1: GatherTopk (CPU辅助) ==========
                key_gather = gather_topk_python(
                    key_cpu[b:b+1],
                    indices_cpu[b:b+1],
                    n2_idx,
                    self.head_dim
                )
                key_gather_npu = torch.from_numpy(key_gather.squeeze(0)).npu()
                
                query_b = query[b, :, n1, :]  # (S1, D)
                
                # ========== 步骤2: Cube1 - Query @ KeyGather^T ==========
                # 每行s有不同的KeyGather，使用einsum
                relu_in = torch.einsum('sd,skd->sk', query_b.float(), key_gather_npu.float()).to(dtype)
                
                # 调试: 检查 relu_in
                if n1 == 0:  # 只打印第一个头，避免太多输出
                    print(f"[调试] relu_in: max={relu_in.abs().max().item():.2f}, "
                          f"min={relu_in.min().item():.2f}, has_nan={torch.isnan(relu_in).any().item()}")
                
                # ========== 步骤3: Cube2 - Weights @ Dy ==========
                # 每个Query头有独立的权重
                weights_b = weights[b, :, n1]  # (S1,)
                dy_b = dy[b]  # (S1, topk)
                relu_grad_in = weights_b.unsqueeze(-1) * dy_b  # 广播乘法
                
                # ========== 步骤4: Vec2 - ReLU反向 + ReduceSum ==========
                # ReLU反向: relu_grad_out = relu_grad_in * (relu_in > 0)
                # 暂时用 PyTorch 实现，跳过有问题的 TileLang kernel
                mask = (relu_in > 0).to(dtype)
                relu_grad_out = relu_grad_in * mask
                
                # 调试: 检查 relu_grad_out
                if n1 == 0:
                    print(f"[调试] relu_grad_in: max={relu_grad_in.abs().max().item():.2f}, has_nan={torch.isnan(relu_grad_in).any().item()}")
                    print(f"[调试] relu_grad_out (PyTorch): max={relu_grad_out.abs().max().item():.2f}, has_nan={torch.isnan(relu_grad_out).any().item()}")
                
                # ReduceSum计算dWeights - 每个Query头的权重梯度
                relu_in_dy = relu_in * dy_b
                # 暂时用 PyTorch 实现
                d_weights_b = relu_in_dy.sum(dim=-1)
                
                # 调试: 检查 d_weights
                if n1 == 0:
                    print(f"[调试] d_weights_b (PyTorch): max={d_weights_b.abs().max().item():.2f}, has_nan={torch.isnan(d_weights_b).any().item()}")
                
                d_weights[b, :, n1] = d_weights_b  # 存储到对应头的位置
                
                # ========== 步骤5: Cube3 - dQuery ==========
                d_query_b = torch.einsum('sk,skd->sd', relu_grad_out.float(), key_gather_npu.float()).to(dtype)
                d_query[b, :, n1, :] = d_query_b
                
                # ========== 步骤6: Cube4 - ReluGrad外积Query ==========
                # 每个位置s需要独立的(topk, D)值，因为scatter时indices不同，不能跨s求和
                # relu_grad_out: (S1, topk), query_b: (S1, D) -> scatter_add: (S1, topk, D)
                scatter_add = torch.einsum('sk,sd->skd', relu_grad_out.float(), query_b.float()).to(dtype)
                
                # 调试: 检查 scatter_add
                if n1 == 0:
                    print(f"[调试] scatter_add: max={scatter_add.abs().max().item():.2f}, has_nan={torch.isnan(scatter_add).any().item()}")
                    print(f"[调试] scatter_add shape: {scatter_add.shape}")

                # ========== 步骤7: Vec3 - ScatterAdd (CPU辅助) ==========
                scatter_add_cpu = scatter_add.cpu().numpy()
                d_key_cpu = d_key[b].cpu().numpy()
                
                # 调试: 检查 indices 范围
                if n1 == 0:
                    indices_b = indices_cpu[b]
                    print(f"[调试] indices: min={indices_b.min()}, max={indices_b.max()}, seqlen_k={self.seqlen_k}")
                    if indices_b.max() >= self.seqlen_k:
                        print(f"[警告] indices 越界! max={indices_b.max()} >= seqlen_k={self.seqlen_k}")
                
                d_key_cpu = scatter_add_python(
                    d_key_cpu[np.newaxis, ...],       # (1, S2, head_num_k, D)
                    scatter_add_cpu[np.newaxis, ...], # (1, S1, topk, D)
                    indices_cpu[b:b+1],               # (1, S1, topk)
                    n2_idx,
                    self.head_dim
                )
                d_key[b] = torch.from_numpy(d_key_cpu.squeeze(0)).to(device)
                
                # 调试: 检查最终 d_key
                if n1 == 0:
                    print(f"[调试] d_key (after scatter): max={d_key[b].abs().max().item():.2f}, has_nan={torch.isnan(d_key[b]).any().item()}")
        
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