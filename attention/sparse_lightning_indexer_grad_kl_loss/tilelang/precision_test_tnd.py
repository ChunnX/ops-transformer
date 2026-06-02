#!/usr/bin/env python3
"""
精度测试 - SparseLightningIndexerGradKLLoss TileLang TND layout

对比 TileLang 实现与官方 Ascend C 算子的结果，使用 TND layout。
入参严格对齐 debug_official.py（已验证可成功调用官方算子）。
"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss_tnd


def run_precision_test():
    print("=" * 70)
    print("精度测试 - SparseLightningIndexerGradKLLoss TileLang (TND layout)")
    print("=" * 70)

    # 严格对齐 debug_official.py / C++ 示例
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

    print(f"\n测试配置 (严格对齐 debug_official.py, TND layout):")
    print(f"  t1={t1}, t2={t2}, n1={n1}, n2={n2}, nQueryIndex={n_query_index}")
    print(f"  d_query={d_query}, d_query_index={d_query_index}, d_rope={d_rope}, topk={topk}, G={G}")

    torch.manual_seed(42)

    # TND layout tensors — 与 debug_official.py 完全一致
    query = torch.randn(t1, n1, d_query, dtype=torch.float16).npu()
    key = torch.randn(t2, n2, d_query, dtype=torch.float16).npu()
    query_index = torch.randn(t1, n_query_index, d_query_index, dtype=torch.float16).npu()
    key_index = torch.randn(t2, n2, d_query_index, dtype=torch.float16).npu()
    weights = torch.randn(t1, n_query_index, dtype=torch.float16).npu()

    # queryRope/keyRope — debug_official.py 中提供了
    query_rope = torch.randn(t1, n1, d_rope, dtype=torch.float16).npu()
    key_rope = torch.randn(t2, n2, d_rope, dtype=torch.float16).npu()

    # sparseIndices: TND (T1, N2, topK) = (1, 1, 2048)
    # debug_official.py: 前 t2 个设为 0..t2-1, 其余设为 -1
    sparse_indices = torch.full((t1, n2, topk), -1, dtype=torch.int32).npu()
    for i in range(t2):
        sparse_indices[0, 0, i] = i

    # softmaxMax/Sum: TND (N2, T1, G) = (1, 1, 128), dtype=float32
    # debug_official.py: softmaxMax=25.4483, softmaxSum=1.0
    softmax_max = torch.full((n2, t1, G), 25.4483, dtype=torch.float32).npu()
    softmax_sum = torch.full((n2, t1, G), 1.0, dtype=torch.float32).npu()

    scale_value = 1.0 / math.sqrt(d_query)

    # actual_seq_lengths: list[int]
    actual_seq_qlen = [t1]
    actual_seq_klen = [t2]

    print(f"\nTensor shapes:")
    print(f"  query: {query.shape}, key: {key.shape}")
    print(f"  query_index: {query_index.shape}, key_index: {key_index.shape}")
    print(f"  weights: {weights.shape}")
    print(f"  query_rope: {query_rope.shape}, key_rope: {key_rope.shape}")
    print(f"  sparse_indices: {sparse_indices.shape}, dtype={sparse_indices.dtype}")
    print(f"  softmax_max: {softmax_max.shape}, softmax_sum: {softmax_sum.shape}")
    print(f"  scale_value: {scale_value}")

    print("\n[1] TileLang TND 计算...")
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss_tnd(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_key=actual_seq_klen,
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

    print("\n[2] 官方 Ascend C 算子计算 (TND layout)...")
    try:
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout="TND",
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
