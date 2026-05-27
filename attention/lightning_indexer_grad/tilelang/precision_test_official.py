#!/usr/bin/env python3
"""
精度测试 - Lightning Indexer Gradient TileLang 实现

对比 TileLang 实现与官方 Ascend C 算子的结果，验证精度。

TileLang Kernel 下发情况:
- kernel_cube4: Cube4 MatMul ✓ 下发NPU
- kernel_reduce: Vec2 ReduceSum ✓ 下发NPU  
- kernel_relu_grad: Vec2 ReLU反向 ✓ 下发NPU
"""

import torch
import torch_npu
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning_indexer_grad_op import LightningIndexerGrad


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
    运行精度测试，对比 TileLang 和官方 Ascend C 算子。
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
    
    torch.npu.synchronize()
    
    print("\n[1] TileLang 计算...")
    d_query_tl, d_key_tl, d_weights_tl = op.forward(
        query, key, dy, indices, weights,
        head_num=head_num_q,
        layout="BSND",
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print("  完成")
    
    print("\n[2] 官方 Ascend C 算子计算...")
    try:
        d_query_off, d_key_off, d_weights_off = torch_npu.npu_lightning_indexer_grad(
            query, key, dy, indices, weights,
            layout="BSND",
            sparse_mode=3,
        )
        torch.npu.synchronize()
        official_available = True
        print("  完成")
    except Exception as e:
        print(f"  失败: {e}")
        official_available = False
    
    if official_available:
        print("\n" + "=" * 70)
        print("精度对比结果")
        print("=" * 70)
        
        # dQuery 对比
        d_query_diff = torch.abs(d_query_tl.float() - d_query_off.float())
        print(f"\n[1] dQuery:")
        print(f"  形状 TileLang: {d_query_tl.shape}")
        print(f"  形状 Official: {d_query_off.shape}")
        print(f"  最大误差: {d_query_diff.max().item():.6f}")
        print(f"  平均误差: {d_query_diff.mean().item():.6f}")
        
        # dKey 对比
        d_key_diff = torch.abs(d_key_tl.float() - d_key_off.float())
        print(f"\n[2] dKey:")
        print(f"  形状 TileLang: {d_key_tl.shape}")
        print(f"  形状 Official: {d_key_off.shape}")
        print(f"  最大误差: {d_key_diff.max().item():.6f}")
        print(f"  平均误差: {d_key_diff.mean().item():.6f}")
        
        # dWeights 对比
        d_weights_diff = torch.abs(d_weights_tl.float() - d_weights_off.float())
        print(f"\n[3] dWeights:")
        print(f"  形状 TileLang: {d_weights_tl.shape}")
        print(f"  形状 Official: {d_weights_off.shape}")
        print(f"  最大误差: {d_weights_diff.max().item():.6f}")
        print(f"  平均误差: {d_weights_diff.mean().item():.6f}")
        
        # 判断是否通过
        threshold = 1e-2
        passed = (d_query_diff.max() < threshold and 
                  d_key_diff.max() < threshold and 
                  d_weights_diff.max() < threshold)
        
        print("\n" + "=" * 70)
        print(f"通过阈值 {threshold}: {'✓' if passed else '✗'}")
        print("=" * 70)
        
        return passed
    else:
        print("\n官方算子不可用，无法进行精度对比")
        return False


def run_multiple_tests():
    """运行多组不同配置的精度测试。"""
    test_configs = [
        # 小尺寸测试
        {"batch": 1, "seqlen_q": 32, "seqlen_k": 64, "head_dim": 32, "topk": 16,
         "head_num_q": 4, "head_num_k": 2, "group_num": 2,
         "block_m": 16, "block_k": 16, "block_d": 16},
        
        # 中等尺寸测试
        {"batch": 2, "seqlen_q": 64, "seqlen_k": 128, "head_dim": 64, "topk": 32,
         "head_num_q": 8, "head_num_k": 2, "group_num": 4,
         "block_m": 32, "block_k": 32, "block_d": 32},
    ]
    
    results = []
    for i, config in enumerate(test_configs):
        print(f"\n{'='*70}")
        print(f"测试组 {i+1}/{len(test_configs)}")
        passed = run_precision_test(**config)
        results.append(passed)
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for i, passed in enumerate(results):
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"  测试组 {i+1}: {status}")
    
    all_passed = all(results)
    print(f"\n总体: {'✓ 全部通过' if all_passed else '✗ 存在失败'}")
    
    return all_passed


if __name__ == "__main__":
    run_multiple_tests()