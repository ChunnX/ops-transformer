#!/usr/bin/env python3
"""
精度测试 - Lightning Indexer Gradient TileLang 实现

对比 TileLang 实现与 PyTorch 参考实现的结果，验证精度。
"""

import torch
import torch_npu
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning_indexer_grad_op import LightningIndexerGrad


def pytorch_reference_forward(
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
    PyTorch 参考实现，用于验证 TileLang 结果的正确性。
    
    参数:
        query: (batch, S1, head_num_q, D)
        key: (batch, S2, head_num_k, D)
        weights: (batch, S1, head_num_q)
        dy: (batch, S1, topk)
        indices: (batch, S1, topk) int32
        relu_in: (batch, S1, topk)
        head_num_q: Query 头数量
        head_num_k: Key 头数量
        group_num: 组数量
    
    返回:
        d_query_ref, d_key_ref, d_weights_ref
    """
    batch = query.shape[0]
    seqlen_q = query.shape[1]
    seqlen_k = key.shape[1]
    head_dim = query.shape[3]
    topk = dy.shape[2]
    head_num_per_group = head_num_q // group_num
    
    device = query.device
    dtype = query.dtype
    
    # 初始化输出
    d_query_ref = torch.zeros_like(query)
    d_key_ref = torch.zeros_like(key)
    d_weights_ref = torch.zeros(batch, seqlen_q, head_num_per_group, dtype=dtype, device=device)
    
    # 处理每个批次
    for b in range(batch):
        for n1 in range(head_num_per_group):
            # 确定对应的 Key 头索引
            n2_idx = n1 * head_num_k // head_num_per_group
            
            # ========== 步骤 1: GatherTopk ==========
            # 基于 indices 收集 Key 数据
            # indices[b]: (S1, topk)
            # key[b]: (S2, head_num_k, D)
            indices_b = indices[b]  # (S1, topk)
            key_b = key[b, :, n2_idx, :]  # (S2, D)
            
            # 收集: 对每个 (s, k)，取 key[indices[s, k], :]
            key_gather = torch.zeros(seqlen_q, topk, head_dim, dtype=dtype, device=device)
            for s in range(seqlen_q):
                for k in range(topk):
                    idx = indices_b[s, k].item()
                    key_gather[s, k, :] = key_b[idx, :]
            
            # ========== 步骤 2: Cube2 - Weights @ Dy ==========
            # weights[b, :, n1]: (S1, 1) -> 实际是 (S1,)
            weights_b = weights[b, :, n1].unsqueeze(-1)  # (S1, 1)
            dy_b = dy[b]  # (S1, topk)
            
            # 广播乘法: relu_grad_in = weights * dy
            relu_grad_in = weights_b * dy_b  # (S1, topk)
            
            # ========== 步骤 3: Vec2 - ReLU 反向 + ReduceSum ==========
            # ReLU 反向: relu_grad_out = relu_grad_in * (relu_in > 0)
            relu_in_b = relu_in[b]  # (S1, topk)
            mask = (relu_in_b > 0).to(dtype)
            relu_grad_out = relu_grad_in * mask
            
            # dWeights = ReduceSum(relu_in * dy)
            d_weights_b = (relu_in_b * dy_b).sum(dim=-1)  # (S1,)
            d_weights_ref[b, :, n1] = d_weights_b
            
            # ========== 步骤 4: Cube3 - ReluGrad @ KeyGather ==========
            # dQuery = relu_grad_out @ KeyGather
            # relu_grad_out: (S1, topk), key_gather: (S1, topk, D)
            # 对每个 s: dQuery[s] = relu_grad_out[s] @ key_gather[s]
            for s in range(seqlen_q):
                d_query_ref[b, s, n1, :] = relu_grad_out[s] @ key_gather[s]
            
            # ========== 步骤 5: Cube4 - ReluGrad^T @ Query ==========
            # scatter_add = relu_grad_out^T @ Query
            # relu_grad_out: (S1, topk), query: (S1, D)
            # scatter_add: (topk, D)
            query_b = query[b, :, n1, :]  # (S1, D)
            scatter_add = relu_grad_out.T @ query_b  # (topk, D)
            
            # ========== 步骤 6: Vec3 - ScatterAdd ==========
            # dKey[indices] += scatter_add
            for k in range(topk):
                for s in range(seqlen_q):
                    idx = indices_b[s, k].item()
                    d_key_ref[b, idx, n2_idx, :] += scatter_add[k, :]
    
    return d_query_ref, d_key_ref, d_weights_ref


def run_precision_test(
    batch=1,
    seqlen_q=64,
    seqlen_k=128,
    head_dim=64,
    topk=32,
    head_num_q=8,
    head_num_k=2,
    group_num=4,
    block_m=32,
    block_k=32,
    block_d=32,
):
    """
    运行精度测试，对比 TileLang 和 PyTorch 参考实现。
    """
    print("=" * 70)
    print("精度测试 - Lightning Indexer Grad TileLang 实现")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  batch={batch}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  head_dim={head_dim}, topk={topk}")
    print(f"  head_num_q={head_num_q}, head_num_k={head_num_k}, group_num={group_num}")
    print(f"  block_m={block_m}, block_k={block_k}, block_d={block_d}")
    
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
    
    print("\n开始 TileLang 计算...")
    d_query_tl, d_key_tl, d_weights_tl = op.forward(query, key, weights, dy, indices, relu_in)
    torch.npu.synchronize()
    print("TileLang 计算完成")
    
    print("\n开始 PyTorch 参考计算...")
    d_query_ref, d_key_ref, d_weights_ref = pytorch_reference_forward(
        query, key, weights, dy, indices, relu_in,
        head_num_q, head_num_k, group_num
    )
    torch.npu.synchronize()
    print("PyTorch 参考计算完成")
    
    # 对比结果
    print("\n" + "=" * 70)
    print("精度对比结果")
    print("=" * 70)
    
    # dQuery 对比
    d_query_diff = torch.abs(d_query_tl.float() - d_query_ref.float())
    d_query_max_diff = d_query_diff.max().item()
    d_query_mean_diff = d_query_diff.mean().item()
    print(f"\n[1] dQuery:")
    print(f"  形状: {d_query_tl.shape}")
    print(f"  最大误差: {d_query_max_diff:.6f}")
    print(f"  平均误差: {d_query_mean_diff:.6f}")
    print(f"  通过: {'✓' if d_query_max_diff < 1e-2 else '✗'}")
    
    # dKey 对比
    d_key_diff = torch.abs(d_key_tl.float() - d_key_ref.float())
    d_key_max_diff = d_key_diff.max().item()
    d_key_mean_diff = d_key_diff.mean().item()
    print(f"\n[2] dKey:")
    print(f"  形状: {d_key_tl.shape}")
    print(f"  最大误差: {d_key_max_diff:.6f}")
    print(f"  平均误差: {d_key_mean_diff:.6f}")
    print(f"  通过: {'✓' if d_key_max_diff < 1e-2 else '✗'}")
    
    # dWeights 对比
    d_weights_diff = torch.abs(d_weights_tl.float() - d_weights_ref.float())
    d_weights_max_diff = d_weights_diff.max().item()
    d_weights_mean_diff = d_weights_diff.mean().item()
    print(f"\n[3] dWeights:")
    print(f"  形状: {d_weights_tl.shape}")
    print(f"  最大误差: {d_weights_max_diff:.6f}")
    print(f"  平均误差: {d_weights_mean_diff:.6f}")
    print(f"  通过: {'✓' if d_weights_max_diff < 1e-2 else '✗'}")
    
    # 总结
    all_passed = (d_query_max_diff < 1e-2 and 
                  d_key_max_diff < 1e-2 and 
                  d_weights_max_diff < 1e-2)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有精度测试通过!")
    else:
        print("✗ 精度测试未通过，请检查实现")
    print("=" * 70)
    
    return all_passed


def run_multiple_tests():
    """
    运行多组不同配置的精度测试。
    """
    test_configs = [
        # 小尺寸快速测试
        {"batch": 1, "seqlen_q": 32, "seqlen_k": 64, "head_dim": 32, "topk": 16,
         "head_num_q": 4, "head_num_k": 2, "group_num": 2,
         "block_m": 16, "block_k": 16, "block_d": 16},
        
        # 中等尺寸
        {"batch": 2, "seqlen_q": 64, "seqlen_k": 128, "head_dim": 64, "topk": 32,
         "head_num_q": 8, "head_num_k": 2, "group_num": 4,
         "block_m": 32, "block_k": 32, "block_d": 32},
        
        # 较大尺寸
        {"batch": 1, "seqlen_q": 128, "seqlen_k": 256, "head_dim": 64, "topk": 64,
         "head_num_q": 8, "head_num_k": 2, "group_num": 4,
         "block_m": 64, "block_k": 32, "block_d": 32},
    ]
    
    results = []
    for i, config in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"测试组 {i+1}/{len(test_configs)}")
        print(f"{'='*70}")
        passed = run_precision_test(**config)
        results.append(passed)
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for i, passed in enumerate(results):
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"  测试组 {i+1}: {status}")
    
    all_passed = all(results)
    print(f"\n总体结果: {'✓ 全部通过' if all_passed else '✗ 存在失败'}")
    
    return all_passed


if __name__ == "__main__":
    # 运行单组测试
    # run_precision_test()
    
    # 运行多组测试
    run_multiple_tests()