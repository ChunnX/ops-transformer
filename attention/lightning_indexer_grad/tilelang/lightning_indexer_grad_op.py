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
    - kernel_cube1: Cube1 MatMul (Query @ KeyGather^T) ✓ Cube单元
    - kernel_cube2: Cube2 外积 (Weights ⊗ Dy) ✓ Cube单元 (matmul_broadcast)
    - kernel_cube3: Cube3 MatMul (ReluGrad @ KeyGather) ✓ Cube单元
    - kernel_cube4: Cube4 MatMul (ReluGrad^T @ Query) ✓ Cube单元
    - kernel_relu_grad: Vec2 ReLU反向 ✓ Vector单元 (内部FP32)
    - kernel_reduce: Vec2 ReduceSum ✓ Vector单元 (内部FP32)
    
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
        block_m: int = 32,
        block_n: int = 32,
        block_k_v: int = 32,
        dtype: str = 'float16',
    ):
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.head_dim = head_dim
        self.topk = topk
        self.head_num_q = head_num_q
        self.head_num_k = head_num_k
        self.group_num = group_num
        
        self.block_m = block_m
        self.block_n = block_n
        self.block_k_v = block_k_v  # vector k dimension for Cube3
        self.dtype = dtype
        
        self._compile_kernels()
    
    def _compile_kernels(self):
        """
        提前编译 TileLang 内核。
        
        Ascend L0B 容量限制: 最大 16KB (FP16)
            - L0B 元素数 <= 8192 (FP16)
            - 对于 transpose_B 的 kernel: block_n * block_k <= 8192
            - 对于普通 matmul: block_k * block_d <= 8192
        
        所有 Cube kernel 输出 FP16。
        Vector kernel 内部使用 FP32 计算，输出 FP16。
        """
        g = self.group_num  # 64
        topk = self.topk    # 2048
        D = self.head_dim   # 128
        
        block_m = self.block_m      # 64
        block_n = self.block_n      # 64 (满足 L0B: 64*128=8192)
        block_k_v = self.block_k_v  # 64 (满足 Cube3 L0B: 64*128=8192)
        
        # Cube1: Query(g, D) @ KeyGather^T(topk, D) -> relu_in (g, topk) FP16
        # B_L0B = (block_n, D) = (64, 128) = 8192 元素 = 16KB ✓
        self.kernel_cube1 = matmul_kernel(
            g, D, topk,
            block_m, D, block_n,  # block_m=64, block_k=D=128, block_n=64
            self.dtype
        )
        
        # Cube2: Weights(g,1) @ Dy(1,topk) -> relu_grad_in (g, topk) FP16
        # B_L0B = (1, block_n) = (1, 64) 很小，无容量问题
        self.kernel_cube2 = matmul_broadcast_kernel(
            g, topk,
            block_m, block_n,  # block_m=64, block_n=64
            self.dtype
        )
        
        # Cube3: ReluGrad(g, topk) @ KeyGather(topk, D) -> dQuery (g, D) FP16
        # B_L0B = (block_k_v, D) = (64, 128) = 8192 元素 = 16KB ✓
        self.kernel_cube3 = matmul_dq_kernel(
            g, topk, D,
            block_m, block_k_v, D,  # block_m=64, block_k=64, block_d=128
            self.dtype
        )
        
        # Cube4: ReluGrad^T(topk, g) @ Query(g, D) -> scatter_add (topk, D) FP16
        # B_L0B = (block_m, D) = (64, 128) = 8192 元素 = 16KB ✓
        self.kernel_cube4 = matmul_scatter_kernel(
            topk, g, D,
            D, block_m, D,  # block_k=128, block_m=64, block_d=128
            self.dtype
        )
        
        # Vec2 ReLU反向: relu_grad_in(FP16) + relu_in(FP16) -> relu_grad_out(FP16)
        self.kernel_relu_grad = relu_grad_kernel(
            g, topk,
            block_m, block_n,  # block_m=64, block_k=64
            self.dtype
        )
        
        # Vec2 ReduceSum: (g, topk) FP16 -> d_weights (g,) FP16
        self.kernel_reduce = reduce_sum_kernel(
            g, topk,
            block_m, block_n,  # block_m=64, block_k=64
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
                    indices_s1 = indices_cpu[b, s1]  # (topk)
                    key_gather = key_cpu[b, indices_s1, n2_idx, :]  # (topk, D)
                    key_gather_npu = torch.from_numpy(key_gather).npu().to(query.dtype)  # (topk, D) FP16
                    
                    # ========== 加载 Query 组和 weights 组 ==========
                    query_group = query[b, s1, start_n1:end_n1, :]  # (g, D) FP16
                    weights_group = weights[b, s1, start_n1:end_n1]  # (g,) FP16
                    dy_s1 = dy[b, s1]  # (topk,) FP16
                    
                    # ========== 步骤2: Cube1 - reluIn = Q @ K^T (TileLang) ==========
                    # 输入: FP16, 输出: FP16 (内部 L0C FP32 累加)
                    relu_in = self.kernel_cube1(query_group, key_gather_npu)  # (g, topk) FP16
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] Cube1 relu_in shape: {relu_in.shape}, dtype: {relu_in.dtype}, max: {relu_in.abs().max().item():.2f}")
                    
                    # ========== 步骤3: Cube2 - reluGradIn = weights ⊗ dy (TileLang Cube外积) ==========
                    # 输入: FP16, 需 reshape: weights(g,) -> (g,1), dy(topk,) -> (1,topk)
                    # 输出: FP16
                    weights_group_2d = weights_group.unsqueeze(1)  # (g, 1)
                    dy_s1_2d = dy_s1.unsqueeze(0)  # (1, topk)
                    relu_grad_in = self.kernel_cube2(weights_group_2d, dy_s1_2d)  # (g, topk) FP16
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] Cube2 relu_grad_in shape: {relu_grad_in.shape}, dtype: {relu_grad_in.dtype}, max: {relu_grad_in.abs().max().item():.2f}")
                    
                    # ========== 步骤4: Vec2 - ReLU反向 (TileLang) ==========
                    # 输入: FP16, 输出: FP16 (内部 Cast relu_in 到 FP32 比较)
                    relu_grad_out = self.kernel_relu_grad(relu_grad_in, relu_in)  # (g, topk) FP16
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] Vec2 relu_grad_out shape: {relu_grad_out.shape}, dtype: {relu_grad_out.dtype}, max: {relu_grad_out.abs().max().item():.2f}")
                    
                    # ========== 步骤5: Vec2 - ReduceSum (TileLang) ==========
                    # 计算 dWeights = sum(reluIn * dy, axis=topk)
                    # 输入: FP16, 输出: FP16 (内部 Cast 到 FP32 累加)
                    # 注意: 确保内存连续，避免 MPU address access invalid
                    relu_in_contiguous = relu_in.contiguous()
                    dy_s1_contiguous = dy_s1.contiguous().unsqueeze(0)
                    relu_in_dy = relu_in_contiguous * dy_s1_contiguous  # (g, topk) FP16
                    d_weights_group = self.kernel_reduce(relu_in_dy)  # (g,) FP16
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] ReduceSum d_weights_group shape: {d_weights_group.shape}, dtype: {d_weights_group.dtype}, [0]: {d_weights_group[0].item():.2f}")
                    
                    d_weights[b, s1, start_n1:end_n1] = d_weights_group
                    
                    # ========== 步骤6: Cube3 - dQuery = reluGrad_out @ K_gather (TileLang) ==========
                    # 输入: FP16, 输出: FP16
                    d_query_group = self.kernel_cube3(relu_grad_out, key_gather_npu)  # (g, D) FP16
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] Cube3 d_query_group shape: {d_query_group.shape}, dtype: {d_query_group.dtype}, [0,0:5]: {d_query_group[0, 0:5].tolist()}")
                    
                    d_query[b, s1, start_n1:end_n1, :] = d_query_group
                    
                    # ========== 步骤7: Cube4 - scatter_add = reluGrad_out^T @ Query (TileLang) ==========
                    # 输入: FP16, 输出: FP16 (内部 L0C FP32 累加)
                    relu_grad_out_T = relu_grad_out.transpose(0, 1).contiguous()  # (topk, g) FP16
                    query_group_contiguous = query_group.contiguous()
                    scatter_add = self.kernel_cube4(relu_grad_out_T, query_group_contiguous)  # (topk, D) FP16
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] Cube4 scatter_add shape: {scatter_add.shape}, dtype: {scatter_add.dtype}, [0,0:5]: {scatter_add[0, 0:5].tolist()}")
                    
                    # ========== 步骤8: Vec3 - ScatterAdd (CPU辅助) ==========
                    # scatter_add 是 FP16，直接累加到 d_key (FP16)
                    for k in range(self.topk):
                        idx = indices_s1[k]
                        d_key[b, idx, n2_idx, :] += scatter_add[k]
                    
                    if b == 0 and n2_idx == 0 and s1 == 0:
                        print(f"[调试] Scatter 完成, d_key[0,0,0,0:5]: {d_key[b, 0, 0, 0:5].tolist()}")
        
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
        block_m=32,
        block_n=32,
        block_k_v=32,
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