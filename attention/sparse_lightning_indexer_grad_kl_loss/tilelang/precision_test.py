#!/usr/bin/env python3
"""
精度测试 - SparseLightningIndexerGradKLLoss TileLang BSND layout

对比 TileLang 实现与官方 Ascend C 算子的结果，使用 BSND layout。
直接构造 BSND 格式入参，不经过 TND 转换。
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

    B = 1
    S1 = 1
    S2 = 1
    N1 = 128
    N2 = 1
    Nidx1 = 64
    Nidx2 = 1
    DQuery = 512
    DQueryIndex = 128
    d_rope = 64
    K = 2048
    G = N1 // N2

    print(f"\n测试配置 (BSND layout):")
    print(f"  B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, Nidx1={Nidx1}, Nidx2={Nidx2}")
    print(f"  DQuery={DQuery}, DQueryIndex={DQueryIndex}, d_rope={d_rope}, K={K}, G={G}")

    torch.manual_seed(42)

    query = torch.randn(B, S1, N1, DQuery, dtype=torch.float16).npu()
    key = torch.randn(B, S2, N2, DQuery, dtype=torch.float16).npu()
    query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=torch.float16).npu()
    key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=torch.float16).npu()
    weights = torch.randn(B, S1, Nidx1, dtype=torch.float16).npu()
    query_rope = torch.randn(B, S1, N1, d_rope, dtype=torch.float16).npu()
    key_rope = torch.randn(B, S2, N2, d_rope, dtype=torch.float16).npu()

    sparse_indices = torch.full((B, S1, Nidx2, K), -1, dtype=torch.int32).npu()
    for i in range(S2):
        sparse_indices[0, 0, 0, i] = i

    softmax_max = torch.full((B, N2, S1, G), 25.4483, dtype=torch.float32).npu()
    softmax_sum = torch.full((B, N2, S1, G), 1.0, dtype=torch.float32).npu()

    scale_value = 1.0 / math.sqrt(DQuery)

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
