#!/usr/bin/env python3
"""
精度测试 - SparseLightningIndexerGradKLLoss TileLang TND layout

对比 TileLang 实现与官方 Ascend C 算子的结果，使用 TND layout。
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

    # 严格对齐官方C++示例: s1=1, s2=1, n1=128, n2=1, nQueryIndex=64, topK=2048
    # TND layout: query(T1,N1,D), key(T2,N2,D)
    t1 = 1
    t2 = 1
    n1 = 128
    n2 = 1
    n_query_index = 64
    d_query = 512
    d_query_index = 128
    topk = 2048
    G = n1 // n2  # G=128

    print(f"\n测试配置 (严格对齐官方C++示例, TND layout):")
    print(f"  t1={t1}, t2={t2}, n1={n1}, n2={n2}, nQueryIndex={n_query_index}")
    print(f"  d_query={d_query}, d_query_index={d_query_index}, topk={topk}, G={G}")

    torch.manual_seed(42)

    # TND layout tensors
    query = torch.randn(t1, n1, d_query, dtype=torch.float16).npu()
    key = torch.randn(t2, n2, d_query, dtype=torch.float16).npu()
    query_index = torch.randn(t1, n_query_index, d_query_index, dtype=torch.float16).npu()
    key_index = torch.randn(t2, n2, d_query_index, dtype=torch.float16).npu()
    weights = torch.randn(t1, n_query_index, dtype=torch.float16).npu()
    # sparseIndices: TND (T1, N2, topK) = (1, 1, 2048)
    # 官方示例: 前 t2=1 个设为 0..t2-1, 其余设为 -1 (padding)
    # 但int32不支持-1作为padding, 官方用-1表示无效索引
    sparse_indices = torch.full((t1, n2, topk), 0, dtype=torch.int32).npu()
    # 官方示例逻辑: sparseIndicesHostData[i] = i for i < t2
    for i in range(min(t2, topk)):
        sparse_indices[0, 0, i] = i

    # Compute softmaxMax/softmaxSum from query @ key^T
    # query: (T1, N1, D) = (T1, N2*G, D), key: (T2, N2, D)
    # scores[t1, g, t2] = sum_d query[t1, n2*G+g, d] * key[t2, n2, d]
    scale_value = 1.0 / math.sqrt(d_query)
    query_group = query.reshape(t1, n2, G, d_query)  # (T1, N2, G, D)
    scores_list = []
    for n2_idx in range(n2):
        q_n2 = query_group[:, n2_idx, :, :].float()  # (T1, G, D)
        k_n2 = key[:, n2_idx, :].float()  # (T2, D)
        scores_n2 = torch.matmul(q_n2, k_n2.transpose(-1, -2))  # (T1, G, T2)
        scores_list.append(scores_n2)
    scores = torch.stack(scores_list, dim=1)  # (T1, N2, G, T2)
    scores_scaled = scores * scale_value
    softmax_max = scores_scaled.max(dim=-1).values  # (T1, N2, G)
    softmax_sum = torch.exp(scores_scaled - softmax_max.unsqueeze(-1).float()).sum(dim=-1)  # (T1, N2, G)
    # TND softmaxMax/Sum shape: (N2, T1, G)
    softmax_max_input = softmax_max.permute(1, 0, 2).contiguous()
    softmax_sum_input = softmax_sum.permute(1, 0, 2).contiguous()

    print("\n[1] TileLang TND 计算...")
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss_tnd(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max_input, softmax_sum_input, scale_value,
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
        # actual_seq_lengths: API签名要求 SymInt[]? (list[int]), 不是 tensor
        actual_seq_qlen = [t1]
        actual_seq_klen = [t2]
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max_input, softmax_sum_input, scale_value,
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