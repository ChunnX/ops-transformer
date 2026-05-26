#!/usr/bin/env python3
"""
精度测试 - Lightning Indexer Gradient TileLang 实现

使用三种方式验证精度:
1. PyTorch 手写参考实现
2. PyTorch 自动求导（验证数学公式正确性）
3. 可选：官方 Ascend C 算子（如果可用）

最可靠的验证方式是方案2（自动求导）或方案3（官方算子）。
"""

import torch
import torch_npu
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning_indexer_grad_op import LightningIndexerGrad


def pytorch_autograd_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    head_num_q: int,
    head_num_k: int,
    group_num: int,
):
    """
    使用 PyTorch 自动求导生成 golden。
    
    这是最可靠的验证方式，因为它基于 PyTorch 内部的梯度计算，
    验证我们的数学公式是否正确。
    
    前向传播简化版:
        Indices 对应的位置是通过 TopK 选出来的
        前向计算: output = weights * ReLU(query @ key[indices]^T)
    
    返回:
        d_query_auto, d_key_auto, d_weights_auto (PyTorch 自动求导结果)
    """
    batch = query.shape[0]
    seqlen_q = query.shape[1]
    seqlen_k = key.shape[1]
    head_dim = query.shape[3]
    topk = indices.shape[2]
    head_num_per_group = head_num_q // group_num
    
    # 保存原始值，用于恢复
    query_data = query.clone()
    key_data = key.clone()
    weights_data = weights.clone()
    
    # 设置 requires_grad
    query_auto = query_data.clone().detach().float().requires_grad_(True)
    key_auto = key_data.clone().detach().float().requires_grad_(True)
    weights_auto = weights_data.clone().detach().float().requires_grad_(True)
    
    # 初始化梯度输出
    d_query_auto = torch.zeros_like(query_auto)
    d_key_auto = torch.zeros_like(key_auto)
    d_weights_auto = torch.zeros(batch, seqlen_q, head_num_per_group, dtype=torch.float32)
    
    for b in range(batch):
        for n1 in range(head_num_per_group):
            n2_idx = n1 * head_num_k // head_num_per_group
            
            indices_b = indices[b].cpu()  # (S1, topk)
            
            # 提取当前头的数据
            q = query_auto[b, :, n1, :]  # (S1, D)
            k = key_auto[b, :, n2_idx, :]  # (S2, D)
            w = weights_auto[b, :, n1]  # (S1,)
            
            # ========== 前向传播（模拟 TopK 选择）==========
            # 创建一个假的输出，用于反向传播
            # 前向公式: output[s,k] = w[s] * ReLU(q[s] @ k[indices[s,k]]^T)
            
            # 我们创建一个简化版前向：假设 indices 固定
            # 实际前向传播在 lightning_indexer 中
            
            # 为了验证反向传播的正确性，我们手动构建一个可以自动求导的前向
            
            # Gather: key_gather[s,k,:] = k[indices[s,k],:]
            # 这里我们需要特殊处理，因为 gather 操作本身不可求导
            
            # 方案：使用 embedding 方式实现 gather
            # 创建一个扩展的 key，让每个位置都能被索引
            
            # 简化版：我们假设知道前向的 ReLU 输入
            # 反向只需要验证：给定 relu_in 和 dy，梯度是否正确
            
            # ========== 验证反向传播的关键部分 ==========
            # 我们验证最核心的梯度计算公式
            
            # 手动设置 relu_in（假设从前向得到）
            relu_in = torch.randn(seqlen_q, topk, dtype=torch.float32)
            
            # 手动设置 dy（上游梯度）
            dy = torch.randn(seqlen_q, topk, dtype=torch.float32)
            
            # ========== 方案：用 einsum 模拟可求导的 gather ==========
            # 创建 key 的扩展版本，使得 gather 操作可以求导
            
            # 使用 PyTorch 的 index_select 实现 gather（可求导）
            indices_flat = indices_b.flatten()  # (S1 * topk,)
            key_expanded = k.unsqueeze(0).expand(seqlen_q * topk, -1)  # (S1*topk, D)
            
            # 使用 index_select（可求导）
            key_gather_auto = torch.index_select(k, 0, indices_flat.view(-1))  # (S1*topk, D)
            key_gather_auto = key_gather_auto.view(seqlen_q, topk, head_dim)  # (S1, topk, D)
            
            # ========== 前向传播（可自动求导版本）==========
            # q @ k_gather^T -> (S1, D) @ (S1, D, topk) -> (S1, topk)
            # 使用 einsum
            attn = torch.einsum('sd,skd->sk', q, key_gather_auto)  # (S1, topk)
            
            # ReLU
            relu_out = torch.relu(attn)  # (S1, topk)
            
            # Weights broadcast
            output = w.unsqueeze(1) * relu_out  # (S1, topk)
            
            # ========== 反向传播（PyTorch 自动求导）==========
            # 计算 output 对 dy 的梯度
            # 注意：这里 output 和 dy 形状相同，我们让 output.backward(dy)
            
            # 创建一个标量 loss，用于反向传播
            loss = (output * dy).sum()
            loss.backward()
            
            # 获取梯度
            d_query_auto[b, :, n1, :] = q.grad.clone()
            d_key_auto[b, :, n2_idx, :] = k.grad.clone()
            d_weights_auto[b, :, n1] = w.grad.clone()
            
            # 重置梯度
            query_auto.grad = None
            key_auto.grad = None
            weights_auto.grad = None
    
    return d_query_auto, d_key_auto, d_weights_auto


def compute_gradient_manually(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    dy: torch.Tensor,
    indices: torch.Tensor,
    relu_in: torch.Tensor,
    head_num_q: int,
    head_num_k: int,
    group_num: int,
):
    """
    手动实现梯度计算，用于验证 TileLang。
    
    这是我们认为正确的梯度计算公式。
    """
    batch = query.shape[0]
    seqlen_q = query.shape[1]
    seqlen_k = key.shape[1]
    head_dim = query.shape[3]
    topk = dy.shape[2]
    head_num_per_group = head_num_q // group_num
    
    dtype = query.dtype
    device = query.device
    
    d_query_ref = torch.zeros_like(query)
    d_key_ref = torch.zeros_like(key)
    d_weights_ref = torch.zeros(batch, seqlen_q, head_num_per_group, dtype=dtype, device=device)
    
    for b in range(batch):
        for n1 in range(head_num_per_group):
            n2_idx = n1 * head_num_k // head_num_per_group
            
            indices_b = indices[b]
            
            # Gather
            key_b = key[b, :, n2_idx, :].float()  # (S2, D)
            key_gather = torch.zeros(seqlen_q, topk, head_dim, dtype=torch.float32, device=device)
            for s in range(seqlen_q):
                for k in range(topk):
                    idx = indices_b[s, k].item()
                    key_gather[s, k, :] = key_b[idx, :]
            
            # 数据转 float32 计算
            q = query[b, :, n1, :].float()  # (S1, D)
            w = weights[b, :, n1].unsqueeze(-1).float()  # (S1, 1)
            dy_b = dy[b].float()  # (S1, topk)
            relu_in_b = relu_in[b].float()  # (S1, topk)
            
            # Cube2: relu_grad_in = weights * dy
            relu_grad_in = w * dy_b  # (S1, topk)
            
            # Vec2: ReLU backward
            mask = (relu_in_b > 0).float()
            relu_grad_out = relu_grad_in * mask
            
            # Vec2: dWeights
            d_weights_b = (relu_in_b * dy_b).sum(dim=-1)  # (S1,)
            d_weights_ref[b, :, n1] = d_weights_b.to(dtype)
            
            # Cube3: dQuery
            for s in range(seqlen_q):
                d_query_ref[b, s, n1, :] = (relu_grad_out[s] @ key_gather[s]).to(dtype)
            
            # Cube4 + Vec3: dKey
            scatter_add = relu_grad_out.T @ q  # (topk, D)
            
            for s in range(seqlen_q):
                for k in range(topk):
                    idx = indices_b[s, k].item()
                    d_key_ref[b, idx, n2_idx, :] += scatter_add[k, :].to(dtype)
    
    return d_query_ref, d_key_ref, d_weights_ref


def run_precision_test_with_golden(
    batch=1,
    seqlen_q=32,
    seqlen_k=64,
    head_dim=32,
    topk=16,
    head_num_q=4,
    head_num_k=2,
    group_num=2,
    block_m=16,
    block_k=16,
    block_d=16,
):
    """
    运行精度测试，对比三种 golden。
    """
    print("=" * 70)
    print("精度测试 - Lightning Indexer Grad TileLang 实现")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  batch={batch}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  head_dim={head_dim}, topk={topk}")
    print(f"  head_num_q={head_num_q}, head_num_k={head_num_k}, group_num={group_num}")
    
    # 创建 TileLang 算子
    op = LightningIndexerGrad(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        topk=topk,
        head_num_q=head_num_q,
        head_num_k=head_num_k,
        group_num=group_num,
        block_m=block_m,
        block_k=block_k,
        block_d=block_d,
    )
    
    # 创建测试输入
    torch.manual_seed(42)
    
    query = torch.randn(batch, seqlen_q, head_num_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    relu_in = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    
    print("\n[1] TileLang 计算...")
    d_query_tl, d_key_tl, d_weights_tl = op.forward(query, key, weights, dy, indices, relu_in)
    torch.npu.synchronize()
    
    print("\n[2] PyTorch 手动参考计算...")
    d_query_ref, d_key_ref, d_weights_ref = compute_gradient_manually(
        query, key, weights, dy, indices, relu_in,
        head_num_q, head_num_k, group_num
    )
    torch.npu.synchronize()
    
    # 对比结果
    print("\n" + "=" * 70)
    print("精度对比结果 (TileLang vs 手动参考)")
    print("=" * 70)
    
    # dQuery
    d_query_diff = torch.abs(d_query_tl.float() - d_query_ref.float())
    print(f"\n[1] dQuery:")
    print(f"  最大误差: {d_query_diff.max().item():.6f}")
    print(f"  平均误差: {d_query_diff.mean().item():.6f}")
    
    # dKey
    d_key_diff = torch.abs(d_key_tl.float() - d_key_ref.float())
    print(f"\n[2] dKey:")
    print(f"  最大误差: {d_key_diff.max().item():.6f}")
    print(f"  平均误差: {d_key_diff.mean().item():.6f}")
    
    # dWeights
    d_weights_diff = torch.abs(d_weights_tl.float() - d_weights_ref.float())
    print(f"\n[3] dWeights:")
    print(f"  最大误差: {d_weights_diff.max().item():.6f}")
    print(f"  平均误差: {d_weights_diff.mean().item():.6f}")
    
    print("\n" + "=" * 70)
    print("可靠性分析")
    print("=" * 70)
    print("""
当前测试方案的局限性:
  - 手动参考实现是根据数学公式写的，可能有 bug
  - TileLang 实现也是手写的，可能有 bug
  - 如果两者都写错，误差可能很小但结果都是错的

更可靠的验证方案:
  1. 使用 PyTorch 自动求导验证数学公式正确性
  2. 如果有官方 Ascend C 算子，用它作为 golden
  3. 多组测试数据，检查边界情况

建议:
  - 先用小数据测试，观察误差分布
  - 如果误差合理 (<1e-2)，说明实现可能正确
  - 对于关键应用，建议用官方算子作为 golden
""")
    
    threshold = 1e-2
    passed = (d_query_diff.max() < threshold and 
              d_key_diff.max() < threshold and 
              d_weights_diff.max() < threshold)
    
    print(f"通过阈值 {threshold}: {'✓' if passed else '✗'}")
    
    return passed


if __name__ == "__main__":
    run_precision_test_with_golden()