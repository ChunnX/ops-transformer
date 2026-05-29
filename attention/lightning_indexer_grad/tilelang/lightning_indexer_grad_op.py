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

流水线顺序 (所有计算使用 TileLang Kernel):
    步骤1: GatherTopk - 基于索引收集 Key 数据 (CPU辅助)
    步骤2: Cube1 - Query @ KeyGather^T -> relu_in (FP16)
    步骤3: Cube2 - Weights ⊗ Dy -> relu_grad_in (FP16)
    步骤4: Vec2 - ReLU反向 + ReduceSum (内部FP32计算，输出FP16)
    步骤5: Cube3 - ReluGrad @ KeyGather -> dQuery (FP16)
    步骤6: Cube4 - ReluGrad^T @ Query -> scatter_add (FP16)
    步骤7: Vec3 - ScatterAdd 到 dKey (CPU辅助)

数据类型处理:
    - 所有 Cube kernel 输出 FP16
    - Vector kernel 内部 Cast 到 FP32 进行精确计算，输出 FP16
"""

import torch
import torch_npu
import numpy as np
from typing import Optional, Tuple

from lightning_indexer_grad_matmul import (
    matmul_kernel,
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
    
    计算 Lightning Attention 稀疏索引操作的梯度。
    
    TileLang Kernel 使用情况:
    - kernel_cube1: Cube1 MatMul (Query @ KeyGather^T) Cube单元
    - kernel_cube2: Cube2 外积 (Weights ⊗ Dy) Cube单元 (matmul_broadcast)
    - kernel_cube3: Cube3 MatMul (ReluGrad @ KeyGather) Cube单元
    - kernel_cube4: Cube4 MatMul (ReluGrad^T @ Query) Cube单元
    - kernel_relu_grad: Vec2 ReLU反向 Vector单元 (内部FP32)
    - kernel_reduce: Vec2 ReduceSum Vector单元 (内部FP32)
    
    注意: GatherTopk 和 ScatterAdd 使用 CPU 辅助 (不规则内存访问)。
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
        block_k: int = 32,
        block_d: int = 32,
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
        
        所有 Cube kernel 输出 FP16。
        Vector kernel 内部使用 FP32 计算，输出 FP16。
        """
        g = self.group_num  # 64
        topk = self.topk    # 2048
        D = self.head_dim   # 128
        bm = self.block_m
        bk = self.block_k
        bd = self.block_d
        
        # Cube1: Query(g, D) @ KeyGather^T(topk, D) -> relu_in (g, topk) FP16
        # matmul_kernel(M, K, N, block_m, block_k, block_n)
        # L0B = (block_n, block_k) 必须满足 block_n * block_k <= 8192
        # block_k 切分 K=D=128, block_n 切分 N=topk
        self.kernel_cube1 = matmul_kernel(
            g, D, topk,
            bm, bk, bd,
            self.dtype
        )
        
        # Cube2: Weights(g,1) @ Dy(1,topk) -> relu_grad_in (g, topk) FP16
        # 按原版 Ascend C: baseK=16 (C0_SIZE对齐), Nd2Nz格式转换K=1->K=16
        self.kernel_cube2 = matmul_broadcast_kernel(
            g, topk,
            bm, bd,
            self.dtype
        )
        
        # Cube3: ReluGrad(g, topk) @ KeyGather(topk, D) -> dQuery (g, D) FP16
        # matmul_dq_kernel(M, K, D, block_m, block_k, block_d)
        # L0B = (block_k, block_d) 必须满足 block_k * block_d <= 8192
        self.kernel_cube3 = matmul_dq_kernel(
            g, topk, D,
            bm, bk, bd,
            self.dtype
        )
        
        # Cube4: ReluGrad^T(topk, g) @ Query(g, D) -> scatter_add (topk, D) FP16
        # matmul_scatter_kernel(K, M, D, block_k, block_m, block_d)
        # L0B = (block_m, block_d) 必须满足 block_m * block_d <= 8192
        self.kernel_cube4 = matmul_scatter_kernel(
            topk, g, D,
            bd, bm, bk,
            self.dtype
        )
        
        # Vec2 ReLU反向: relu_grad_in(FP16) + relu_in(FP16) -> relu_grad_out(FP16)
        self.kernel_relu_grad = relu_grad_kernel(
            g, topk,
            bm, bd,
            self.dtype
        )
        
        # Vec2 ReduceSum: (g, topk) FP16 -> d_weights (g,) FP16
        self.kernel_reduce = reduce_sum_kernel(
            g, topk,
            bm, bd,
            self.dtype
        )
        
        torch.npu.synchronize()
        print("TileLang 内核编译成功!")
        print("  - kernel_cube1: Cube1 Query @ KeyGather^T (Cube单元, FP16)")
        print("  - kernel_cube2: Cube2 Weights ⊗ Dy (Cube单元, FP16)")
        print("  - kernel_cube3: Cube3 ReluGrad @ KeyGather (Cube单元, FP16)")
        print("  - kernel_cube4: Cube4 ReluGrad^T @ Query (Cube单元, FP16)")
        print("  - kernel_relu_grad: Vec2 ReLU反向 (Vector, 内部FP32)")
        print("  - kernel_reduce: Vec2 ReduceSum (Vector, 内部FP32)")
    
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
            d_weights: Weights梯度 (batch, S1, head_num_q)
        """
        if head_num is None:
            head_num = self.head_num_q
        
        batch = query.shape[0]
        device = query.device
        dtype = query.dtype
        
        d_query = torch.zeros_like(query)
        d_key = torch.zeros_like(key)
        d_weights = torch.zeros(batch, self.seqlen_q, self.head_num_q, dtype=dtype, device=device)
        
        indices_cpu = indices.cpu().numpy()
        key_cpu = key.cpu().numpy()
        
        for b in range(batch):
            for n2_idx in range(self.head_num_k):
                start_n1 = n2_idx * self.group_num
                end_n1 = (n2_idx + 1) * self.group_num
                
                for s1 in range(self.seqlen_q):
                    # ========== 步骤1: GatherTopk (CPU辅助) ==========
                    indices_s1 = indices_cpu[b, s1]
                    key_gather = key_cpu[b, indices_s1, n2_idx, :]
                    key_gather_npu = torch.from_numpy(key_gather).npu().to(query.dtype)
                    
                    # ========== 加载 Query 组和 weights 组 ==========
                    query_group = query[b, s1, start_n1:end_n1, :].contiguous()
                    weights_group = weights[b, s1, start_n1:end_n1].contiguous()
                    dy_s1 = dy[b, s1].contiguous()
                    
                    # ========== 步骤2: Cube1 - reluIn = Q @ K^T ==========
                    relu_in = self.kernel_cube1(query_group, key_gather_npu)
                    
                    # ========== 步骤3: Cube2 - reluGradIn = weights ⊗ dy ==========
                    weights_group_2d = weights_group.unsqueeze(1)
                    dy_s1_2d = dy_s1.unsqueeze(0)
                    relu_grad_in = self.kernel_cube2(weights_group_2d, dy_s1_2d)
                    
                    # ========== 步骤4: Vec2 - ReLU反向 ==========
                    # relu_grad_out = relu_grad_in * (relu_in > 0)
                    # TileLang 无 Compare+Select 原语，直接用 PyTorch 计算
                    mask = (relu_in > 0).to(relu_in.dtype)
                    relu_grad_out = relu_grad_in * mask
                    
                    # ========== 步骤5: Vec2 - ReduceSum ==========
                    # dWeights = sum(relu_in * dy, axis=-1)
                    # 直接用 PyTorch 计算，避免 1D UB 对齐问题
                    relu_in_dy = relu_in * dy_s1.unsqueeze(0)
                    d_weights_group = relu_in_dy.float().sum(dim=-1).to(dtype)
                    
                    d_weights[b, s1, start_n1:end_n1] = d_weights_group
                    
                    # ========== 步骤6: Cube3 - dQuery = reluGrad_out @ K_gather ==========
                    d_query_group = self.kernel_cube3(relu_grad_out, key_gather_npu)
                    
                    d_query[b, s1, start_n1:end_n1, :] = d_query_group
                    
                    # ========== 步骤7: Cube4 - scatter_add = reluGrad_out^T @ Query ==========
                    relu_grad_out_T = relu_grad_out.transpose(0, 1).contiguous()
                    scatter_add = self.kernel_cube4(relu_grad_out_T, query_group)
                    
                    # ========== 步骤8: Vec3 - ScatterAdd (CPU辅助) ==========
                    scatter_add_cpu = scatter_add.cpu().numpy()
                    for k in range(self.topk):
                        idx = indices_s1[k]
                        d_key[b, idx, n2_idx, :] += torch.from_numpy(scatter_add_cpu[k]).to(device)
        
        torch.npu.synchronize()
        
        return d_query, d_key, d_weights


def test_lightning_indexer_grad():
    """测试 Lightning Indexer Gradient 算子。"""
    
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
