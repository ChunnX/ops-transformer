#!/usr/bin/env python3
"""
Lightning Indexer Gradient TileLang 内核单元测试。

逐个测试每个内核组件，并与 PyTorch 参考结果进行验证。
"""

import torch
import torch_npu
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_matmul_relu():
    """测试 Cube1 内核: Query @ KeyGather^T 并带 ReLU。"""
    print("\n=== 测试 MatMul ReLU (Cube1) ===")
    
    M, K, D = 128, 64, 128
    block_m, block_k, block_d = 64, 32, 64
    
    kernel = matmul_relu_kernel(M, D, K, block_m, block_d, block_k)
    
    query = torch.randn(M, D, dtype=torch.float16).npu()
    key_gather = torch.randn(K, D, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    output = kernel(query, key_gather)
    torch.npu.synchronize()
    
    # 参考结果: query @ key_gather^T 并带 ReLU
    ref = torch.relu(torch.matmul(query, key_gather.transpose(0, 1)))
    
    # 检查形状
    assert output.shape == (M, K), f"形状不匹配: {output.shape} vs {(M, K)}"
    
    # 检查值
    torch.testing.assert_close(output.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"✓ MatMul ReLU 内核测试通过! 形状: {output.shape}")


def test_matmul_broadcast():
    """测试 Cube2 内核: Weights @ Dy 带广播。"""
    print("\n=== 测试 MatMul Broadcast (Cube2) ===")
    
    M, K = 128, 64
    block_m, block_k = 64, 32
    
    kernel = matmul_broadcast_kernel(M, 1, K, block_m, 1, block_k)
    
    weights = torch.randn(M, dtype=torch.float16).npu()
    dy = torch.randn(K, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    output = kernel(weights, dy)
    torch.npu.synchronize()
    
    # 参考结果: 外积
    ref = torch.outer(weights, dy)
    
    assert output.shape == (M, K), f"形状不匹配: {output.shape} vs {(M, K)}"
    torch.testing.assert_close(output.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"✓ MatMul Broadcast 内核测试通过! 形状: {output.shape}")


def test_matmul_dq():
    """测试 Cube3 内核: ReluGrad @ KeyGather -> dQuery。"""
    print("\n=== 测试 MatMul dQuery (Cube3) ===")
    
    M, K, D = 128, 64, 128
    block_m, block_k, block_d = 64, 32, 64
    
    kernel = matmul_dq_kernel(M, K, D, block_m, block_k, block_d)
    
    relu_grad = torch.randn(M, K, dtype=torch.float16).npu()
    key_gather = torch.randn(K, D, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    d_query = kernel(relu_grad, key_gather)
    torch.npu.synchronize()
    
    # 参考结果: relu_grad @ key_gather
    ref = torch.matmul(relu_grad, key_gather)
    
    assert d_query.shape == (M, D), f"形状不匹配: {d_query.shape} vs {(M, D)}"
    torch.testing.assert_close(d_query.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"✓ MatMul dQuery 内核测试通过! 形状: {d_query.shape}")


def test_matmul_scatter():
    """测试 Cube4 内核: ReluGrad^T @ Query -> scatter_add。"""
    print("\n=== 测试 MatMul Scatter (Cube4) ===")
    
    K, M, D = 64, 128, 128
    block_k, block_m, block_d = 32, 64, 64
    
    kernel = matmul_scatter_kernel(K, M, D, block_k, block_m, block_d)
    
    relu_grad = torch.randn(K, M, dtype=torch.float16).npu()
    query = torch.randn(M, D, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    scatter_add = kernel(relu_grad, query)
    torch.npu.synchronize()
    
    # 参考结果: relu_grad @ query
    ref = torch.matmul(relu_grad, query)
    
    assert scatter_add.shape == (K, D), f"形状不匹配: {scatter_add.shape} vs {(K, D)}"
    torch.testing.assert_close(scatter_add.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"✓ MatMul Scatter 内核测试通过! 形状: {scatter_add.shape}")


def test_reduce_sum():
    """测试 Vec2 内核: ReduceSum 计算 dWeights。"""
    print("\n=== 测试 ReduceSum (Vec2) ===")
    
    M, K = 128, 64
    block_m, block_k = 64, 32
    
    kernel = reduce_sum_kernel(M, K, block_m, block_k)
    
    input_tensor = torch.randn(M, K, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    output = kernel(input_tensor)
    torch.npu.synchronize()
    
    # 参考结果: 沿 dim -1 求和
    ref = input_tensor.sum(dim=-1)
    
    assert output.shape == (M,), f"形状不匹配: {output.shape} vs {(M,)}"
    torch.testing.assert_close(output.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"✓ ReduceSum 内核测试通过! 形状: {output.shape}")


def test_relu_grad():
    """测试 Vec2 内核: ReLU 反向。"""
    print("\n=== 测试 ReluGrad (Vec2) ===")
    
    M, K = 128, 64
    block_m, block_k = 64, 32
    
    kernel = relu_grad_kernel(M, K, block_m, block_k)
    
    relu_grad_in = torch.randn(M, K, dtype=torch.float16).npu()
    relu_in = torch.randn(M, K, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    relu_grad_out = kernel(relu_grad_in, relu_in)
    torch.npu.synchronize()
    
    # 参考结果: relu_grad_in * (relu_in > 0)
    mask = (relu_in > 0).float()
    ref = relu_grad_in * mask.half()
    
    assert relu_grad_out.shape == (M, K), f"形状不匹配: {relu_grad_out.shape} vs {(M, K)}"
    torch.testing.assert_close(relu_grad_out.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"✓ ReluGrad 内核测试通过! 形状: {relu_grad_out.shape}")


def test_gather_topk():
    """测试 GatherTopk 辅助函数。"""
    print("\n=== 测试 GatherTopk ===")
    
    batch, seqlen_q, seqlen_k, head_num_k, topk, head_dim = 2, 16, 32, 4, 8, 64
    
    key = np.random.randn(batch, seqlen_k, head_num_k, head_dim).astype(np.float32)
    indices = np.random.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=np.int32)
    
    n2_idx = 1
    
    key_gather = gather_topk_python(key, indices, n2_idx, head_dim)
    
    # 检查形状
    assert key_gather.shape == (batch, seqlen_q, topk, head_dim), \
        f"形状不匹配: {key_gather.shape} vs {(batch, seqlen_q, topk, head_dim)}"
    
    # 验证值
    for b in range(batch):
        for s in range(seqlen_q):
            for k in range(topk):
                idx = indices[b, s, k]
                np.testing.assert_array_almost_equal(
                    key_gather[b, s, k, :],
                    key[b, idx, n2_idx, :],
                    decimal=5
                )
    
    print(f"✓ GatherTopk 测试通过! 形状: {key_gather.shape}")


def test_scatter_add():
    """测试 ScatterAdd 辅助函数。"""
    print("\n=== 测试 ScatterAdd ===")
    
    batch, seqlen_q, seqlen_k, head_num_k, topk, head_dim = 2, 16, 32, 4, 8, 64
    
    d_key = np.random.randn(batch, seqlen_k, head_num_k, head_dim).astype(np.float32)
    d_key_orig = d_key.copy()
    scatter_add = np.random.randn(batch, seqlen_q, topk, head_dim).astype(np.float32)
    indices = np.random.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=np.int32)
    
    n2_idx = 2
    
    d_key = scatter_add_python(d_key, scatter_add, indices, n2_idx, head_dim)
    
    # 验证累加
    d_key_ref = d_key_orig.copy()
    for b in range(batch):
        for s in range(seqlen_q):
            for k in range(topk):
                idx = indices[b, s, k]
                d_key_ref[b, idx, n2_idx, :] += scatter_add[b, s, k, :]
    
    np.testing.assert_array_almost_equal(d_key, d_key_ref, decimal=5)
    print(f"✓ ScatterAdd 测试通过!")


def test_full_pipeline():
    """使用小尺寸测试完整流水线。"""
    print("\n=== 测试完整流水线 ===")
    
    from lightning_indexer_grad_op import LightningIndexerGrad
    
    batch = 1
    seqlen_q = 32
    seqlen_k = 64
    head_dim = 32
    topk = 16
    head_num_q = 4
    head_num_k = 2
    group_num = 2
    
    op = LightningIndexerGrad(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        topk=topk,
        head_num_q=head_num_q,
        head_num_k=head_num_k,
        group_num=group_num,
        block_m=16,
        block_k=16,
        block_d=16,
    )
    
    query = torch.randn(batch, seqlen_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q // group_num, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()
    relu_in = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    
    torch.npu.synchronize()
    
    d_query, d_key, d_weights = op.forward(query, key, weights, dy, indices, relu_in)
    
    torch.npu.synchronize()
    
    print(f"✓ 完整流水线测试通过!")
    print(f"  d_query 形状: {d_query.shape}")
    print(f"  d_key 形状: {d_key.shape}")
    print(f"  d_weights 形状: {d_weights.shape}")


def run_all_tests():
    """运行所有单元测试。"""
    print("=" * 60)
    print("运行 Lightning Indexer Grad TileLang 内核测试")
    print("=" * 60)
    
    try:
        test_matmul_relu()
        test_matmul_broadcast()
        test_matmul_dq()
        test_matmul_scatter()
        test_reduce_sum()
        test_relu_grad()
        test_gather_topk()
        test_scatter_add()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败，错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)