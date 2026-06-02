#!/usr/bin/env python3
"""
精度测试 - SparseLightningIndexerGradKLLoss TileLang BSND layout

对比 TileLang 实现与官方 Ascend C 算子的结果，使用 BSND layout。
入参严格对齐 debug_official.py 的 BSND 调用方式（TND unsqueeze B=1）。
"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss


def run_precision_test():
    print("=" * 70)
    print("精度测试 - SparseLightningIndexerGradKLLoss TileLang (BSND layout)")
    print("=" * 70)

    # 对齐 debug_official.py 的 BSND 调用方式
    # TND 参数: t1=1, t2=1, n1=128, n2=1, nQueryIndex=64, topK=2048
    # BSND: B=1, S1=t1=1, S2=t2=1, N1=128, N2=1, Nidx1=64, Nidx2=1
    t1 = 1
    t2 = 1
    n1 = 128
    n2 = 1
    n_query_index = 64
    d_query = 512
    d_query_index = 128
    d_rope = 64
    topk = 2048
    G = n1 // n2  # G=128

    # BSND shapes (unsqueeze TND with B=1)
    B = 1
    S1 = t1
    S2 = t2

    print(f"\n测试配置 (对齐 debug_official.py BSND 调用):")
    print(f"  B={B}, S1={S1}, S2={S2}, N1={n1}, N2={n2}, Nidx1={n_query_index}")
    print(f"  d_query={d_query}, d_query_index={d_query_index}, d_rope={d_rope}, topk={topk}, G={G}")

    torch.manual_seed(42)

    # 先构造 TND tensors（与 debug_official.py 一致），再 unsqueeze 得到 BSND
    query_tnd = torch.randn(t1, n1, d_query, dtype=torch.float16).npu()
    key_tnd = torch.randn(t2, n2, d_query, dtype=torch.float16).npu()
    query_index_tnd = torch.randn(t1, n_query_index, d_query_index, dtype=torch.float16).npu()
    key_index_tnd = torch.randn(t2, n2, d_query_index, dtype=torch.float16).npu()
    weights_tnd = torch.randn(t1, n_query_index, dtype=torch.float16).npu()
    query_rope_tnd = torch.randn(t1, n1, d_rope, dtype=torch.float16).npu()
    key_rope_tnd = torch.randn(t2, n2, d_rope, dtype=torch.float16).npu()

    # sparseIndices: TND (T1, N2, topK)
    sparse_indices_tnd = torch.full((t1, n2, topk), -1, dtype=torch.int32).npu()
    for i in range(t2):
        sparse_indices_tnd[0, 0, i] = i

    # softmaxMax/Sum: TND (N2, T1, G)
    softmax_max_tnd = torch.full((n2, t1, G), 25.4483, dtype=torch.float32).npu()
    softmax_sum_tnd = torch.full((n2, t1, G), 1.0, dtype=torch.float32).npu()

    scale_value = 1.0 / math.sqrt(d_query)

    # TND -> BSND: unsqueeze B=1
    query = query_tnd.unsqueeze(0)
    key = key_tnd.unsqueeze(0)
    query_index = query_index_tnd.unsqueeze(0)
    key_index = key_index_tnd.unsqueeze(0)
    weights = weights_tnd.unsqueeze(0)
    sparse_indices = sparse_indices_tnd.unsqueeze(0)
    query_rope = query_rope_tnd.unsqueeze(0)
    key_rope = key_rope_tnd.unsqueeze(0)
    # softmaxMax/Sum: TND (N2,T1,G) -> BSND (B,N2,T1,G)
    softmax_max = softmax_max_tnd.unsqueeze(0)
    softmax_sum = softmax_sum_tnd.unsqueeze(0)

    print(f"\nBSND Tensor shapes:")
    print(f"  query: {query.shape}, key: {key.shape}")
    print(f"  query_index: {query_index.shape}, key_index: {key_index.shape}")
    print(f"  weights: {weights.shape}")
    print(f"  query_rope: {query_rope.shape}, key_rope: {key_rope.shape}")
    print(f"  sparse_indices: {sparse_indices.shape}, dtype={sparse_indices.dtype}")
    print(f"  softmax_max: {softmax_max.shape}, softmax_sum: {softmax_sum.shape}")
    print(f"  scale_value: {scale_value}")

    print("\n[1] TileLang BSND 计算...")
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            layout_query="BSND",
            sparse_mode=3,
        )
        torch.npu.synchronize()
        print("  完成")
        tl_available = True
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
        tl_available = False

    print("\n[2] 官方 Ascend C 算子计算 (BSND layout)...")
    try:
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            layout="BSND",
            sparse_mode=3,
        )
        torch.npu.synchronize()
        print("  完成")
        official_available = True
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
        official_available = False

    if tl_available and official_available:
        print("\n[3] 精度对比...")
        for name, tl_val, off_val in [
            ("dQueryIndex", d_qi_tl, d_qi_off),
            ("dKeyIndex", d_ki_tl, d_ki_off),
            ("dWeights", dw_tl, dw_off),
        ]:
            diff = (tl_val.float() - off_val.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
            print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.6f}")
        loss_diff = abs(loss_tl.float().item() - loss_off.float().item())
        print(f"  Loss: abs_diff={loss_diff:.6f}, tl={loss_tl.item():.6f}, off={loss_off.item():.6f}")
    elif tl_available:
        print("\n[3] 官方算子不可用，仅验证 TileLang 输出非全零...")
        for name, val in [("dQueryIndex", d_qi_tl), ("dKeyIndex", d_ki_tl), ("dWeights", dw_tl)]:
            print(f"  {name}: max={val.abs().max().item():.6f}, nonzero={val.count_nonzero().item()}")
        print(f"  Loss: value={loss_tl.item():.6f}")
    else:
        print("\n[3] TileLang 计算失败，无法对比")


if __name__ == "__main__":
    run_precision_test()
