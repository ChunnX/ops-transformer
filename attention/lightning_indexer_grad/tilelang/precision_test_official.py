#!/usr/bin/env python3
"""
精度测试 - 使用官方 Ascend C 算子作为 Golden

对比 TileLang 实现与官方 torch_npu.npu_lightning_indexer_grad 的结果，
这是最可靠的验证方式。
"""

import torch
import torch_npu
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning_indexer_grad_op import LightningIndexerGrad


def call_official_op(
    query: torch.Tensor,
    key: torch.Tensor,
    dy: torch.Tensor,
    sparse_indices: torch.Tensor,
    weights: torch.Tensor,
    head_num: int,
    layout: str = "BSND",
    sparse_mode: int = 0,
    pre_tokens: int = 65536,
    next_tokens: int = 65536,
    deterministic: bool = False,
):
    """
    调用官方 torch_npu.npu_lightning_indexer_grad 算子。
    
    参数根据官方 C++ 代码推断:
        query: Query 张量
        key: Key 张量
        dy: 上游梯度
        sparse_indices: TopK 索引
        weights: 权重张量
        head_num: 头数量
        layout: 输入布局 ("BSND" 或 "TND")
        sparse_mode: 稀疏模式 (0 或 3)
        pre_tokens: 前向 token 数
        next_tokens: 后向 token 数
        deterministic: 是否确定性计算
    
    返回:
        d_query, d_key, d_weights
    """
    # 尝试不同的调用方式
    try:
        # 方式1: 直接参数调用
        d_query, d_key, d_weights = torch_npu.npu_lightning_indexer_grad(
            query,
            key,
            dy,
            sparse_indices,
            weights,
            head_num=head_num,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            deterministic=deterministic,
        )
        return d_query, d_key, d_weights
    except TypeError as e:
        print(f"方式1失败: {e}")
    
    try:
        # 方式2: 使用 None 作为可选参数
        d_query, d_key, d_weights = torch_npu.npu_lightning_indexer_grad(
            query,
            key,
            dy,
            sparse_indices,
            weights,
            None,  # actual_seq_q_len
            None,  # actual_seq_k_len
            head_num=head_num,
            layout=layout,
            sparse_mode=sparse_mode,
        )
        return d_query, d_key, d_weights
    except TypeError as e:
        print(f"方式2失败: {e}")
    
    try:
        # 方式3: 简化参数
        result = torch_npu.npu_lightning_indexer_grad(
            query, key, dy, sparse_indices, weights,
            head_num
        )
        if isinstance(result, tuple) and len(result) == 3:
            return result
    except TypeError as e:
        print(f"方式3失败: {e}")
    
    raise RuntimeError("无法调用官方算子，请检查参数格式")


def run_precision_test_with_official_golden(
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
    使用官方算子作为 golden 的精度测试。
    """
    print("=" * 70)
    print("精度测试 - TileLang vs 官方 Ascend C 算子")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  batch={batch}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  head_dim={head_dim}, topk={topk}")
    print(f"  head_num_q={head_num_q}, head_num_k={head_num_k}, group_num={group_num}")
    
    # 创建测试输入
    torch.manual_seed(42)
    
    # 注意：官方算子的输入形状可能需要 BSND 布局
    # query: (batch, seqlen_q, head_num_q, head_dim)
    # key: (batch, seqlen_k, head_num_k, head_dim)
    query = torch.randn(batch, seqlen_q, head_num_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    sparse_indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    
    torch.npu.synchronize()
    
    # ========== 官方算子计算 ==========
    print("\n[1] 官方 Ascend C 算子计算...")
    try:
        d_query_official, d_key_official, d_weights_official = call_official_op(
            query, key, dy, sparse_indices, weights,
            head_num=head_num_q,
            layout="BSND",
            sparse_mode=0,
        )
        torch.npu.synchronize()
        print("官方算子计算完成")
        official_available = True
    except Exception as e:
        print(f"官方算子调用失败: {e}")
        print("将跳过官方算子对比")
        official_available = False
        d_query_official = None
        d_key_official = None
        d_weights_official = None
    
    # ========== TileLang 计算 ==========
    print("\n[2] TileLang 计算...")
    
    # TileLang 算子需要 relu_in，这是前向传播的 ReLU 输入
    # 我们需要先运行前向算子得到 relu_in，或者用随机值模拟
    # 这里用随机值模拟（实际应用中应该从前向传播获得）
    relu_in = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    
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
    
    d_query_tl, d_key_tl, d_weights_tl = op.forward(query, key, weights, dy, sparse_indices, relu_in)
    torch.npu.synchronize()
    print("TileLang 计算完成")
    
    # ========== 结果对比 ==========
    if official_available:
        print("\n" + "=" * 70)
        print("精度对比结果")
        print("=" * 70)
        
        # dQuery 对比
        d_query_diff = torch.abs(d_query_tl.float() - d_query_official.float())
        print(f"\n[1] dQuery:")
        print(f"  形状 TileLang: {d_query_tl.shape}")
        print(f"  形状 Official: {d_query_official.shape}")
        print(f"  最大误差: {d_query_diff.max().item():.6f}")
        print(f"  平均误差: {d_query_diff.mean().item():.6f}")
        
        # dKey 对比
        d_key_diff = torch.abs(d_key_tl.float() - d_key_official.float())
        print(f"\n[2] dKey:")
        print(f"  形状 TileLang: {d_key_tl.shape}")
        print(f"  形状 Official: {d_key_official.shape}")
        print(f"  最大误差: {d_key_diff.max().item():.6f}")
        print(f"  平均误差: {d_key_diff.mean().item():.6f}")
        
        # dWeights 对比
        d_weights_diff = torch.abs(d_weights_tl.float() - d_weights_official.float())
        print(f"\n[3] dWeights:")
        print(f"  形状 TileLang: {d_weights_tl.shape}")
        print(f"  形状 Official: {d_weights_official.shape}")
        print(f"  最大误差: {d_weights_diff.max().item():.6f}")
        print(f"  平均误差: {d_weights_diff.mean().item():.6f}")
        
        # 判断是否通过
        threshold = 1e-2
        passed = (d_query_diff.max() < threshold and 
                  d_key_diff.max() < threshold and 
                  d_weights_diff.max() < threshold)
        
        print("\n" + "=" * 70)
        if passed:
            print(f"✓ 精度测试通过 (阈值 {threshold})")
        else:
            print(f"✗ 精度测试未通过 (阈值 {threshold})")
            
            # 显示一些样本值帮助调试
            print("\n调试信息 - 前5个值对比:")
            print(f"  dQuery[0,0,0,:5]:")
            print(f"    TileLang: {d_query_tl[0,0,0,:5].float()}")
            print(f"    Official: {d_query_official[0,0,0,:5].float()}")
        print("=" * 70)
        
        return passed
    else:
        print("\n官方算子不可用，无法进行精度对比")
        return False


def test_official_op_signature():
    """
    测试官方算子的正确调用方式。
    """
    print("=" * 70)
    print("测试官方算子调用方式")
    print("=" * 70)
    
    # 创建小测试数据
    batch = 1
    seqlen_q = 8
    seqlen_k = 16
    head_num_q = 4
    head_num_k = 1
    head_dim = 32
    topk = 8
    
    query = torch.randn(batch, seqlen_q, head_num_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    sparse_indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    
    torch.npu.synchronize()
    
    # 尝试各种调用方式
    call_methods = [
        # 方式1: 带所有参数
        {
            "name": "完整参数",
            "call": lambda: torch_npu.npu_lightning_indexer_grad(
                query, key, dy, sparse_indices, weights,
                head_num=head_num_q,
                layout="BSND",
                sparse_mode=0,
            )
        },
        # 方式2: 最小参数
        {
            "name": "最小参数",
            "call": lambda: torch_npu.npu_lightning_indexer_grad(
                query, key, dy, sparse_indices, weights,
                head_num_q
            )
        },
        # 方式3: 关键字参数
        {
            "name": "关键字参数",
            "call": lambda: torch_npu.npu_lightning_indexer_grad(
                query=query,
                key=key,
                dy=dy,
                sparse_indices=sparse_indices,
                weights=weights,
                headNum=head_num_q,
                inputLayout="BSND",
            )
        },
    ]
    
    for method in call_methods:
        print(f"\n尝试 {method['name']}...")
        try:
            result = method["call"]()
            torch.npu.synchronize()
            if isinstance(result, tuple):
                print(f"  ✓ 成功! 返回 {len(result)} 个张量")
                for i, r in enumerate(result):
                    print(f"    输出{i}: 形状 {r.shape}, 类型 {r.dtype}")
                return result  # 返回成功的调用方式
            else:
                print(f"  返回类型: {type(result)}")
        except TypeError as e:
            print(f"  ✗ TypeError: {e}")
        except RuntimeError as e:
            print(f"  ✗ RuntimeError: {e}")
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
    
    print("\n所有调用方式都失败")
    return None


if __name__ == "__main__":
    # 先测试官方算子调用方式
    test_official_op_signature()
    
    # 然后运行精度测试
    print("\n")
    run_precision_test_with_official_golden()