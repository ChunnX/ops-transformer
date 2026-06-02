#!/usr/bin/env python3
"""
精度测试 - SparseLightningIndexerGradKLLoss TileLang CV 分离架构

对比 TileLang CV 分离实现与官方 Ascend C 算子的结果。
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
    print("精度测试 - SparseLightningIndexerGradKLLoss TileLang CV 分离架构")
    print("=" * 70)

    # 对齐官方示例参数: s1=1, s2=1, n1=128, n2=1, nQueryIndex=64
    # 但BSND layout需要B维度, 使用s1=128以匹配BLOCK_M=128
    batch = 1
    seqlen_q = 128
    seqlen_k = 128
    d_query = 512
    d_query_index = 128
    topk = 2048
    head_num_q = 128
    head_num_k = 1
    head_num_idx = 64
    G = head_num_q // head_num_k  # G=128 (N1/N2)

    print(f"\n测试配置 (对齐官方示例, BSND layout):")
    print(f"  batch={batch}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  d_query={d_query}, d_query_index={d_query_index}, topk={topk}")
    print(f"  head_num_q={head_num_q}, head_num_k={head_num_k}, head_num_idx={head_num_idx}, G={G}")

    torch.manual_seed(42)

    query = torch.randn(batch, seqlen_q, head_num_q, d_query, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, d_query, dtype=torch.float16).npu()
    query_index = torch.randn(batch, seqlen_q, head_num_idx, d_query_index, dtype=torch.float16).npu()
    key_index = torch.randn(batch, seqlen_k, head_num_k, d_query_index, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_idx, dtype=torch.float16).npu()
    # sparse_indices: (B, S1, Nidx2, K) = (B, S1, N2, K), Nidx2=N2=1
    # indices must be in [0, S2)
    sparse_indices = torch.randint(0, seqlen_k, (batch, seqlen_q, head_num_k, topk), dtype=torch.int32).npu()

    # Compute softmaxMax/softmaxSum from query @ key^T
    # For each n2: scores[b,s1,g,s2] = sum_d query[b,s1,n2*G+g,d] * key[b,s2,n2,d]
    scale_value = 1.0 / math.sqrt(d_query)
    query_group = query.reshape(batch, seqlen_q, head_num_k, G, d_query)  # (B, S1, N2, G, D)
    scores_list = []
    for n2 in range(head_num_k):
        q_n2 = query_group[:, :, n2, :, :].float()  # (B, S1, G, D)
        k_n2 = key[:, :, n2, :].float()  # (B, S2, D)
        scores_n2 = torch.matmul(q_n2, k_n2.transpose(-1, -2))  # (B, S1, G, S2)
        scores_list.append(scores_n2)
    scores = torch.stack(scores_list, dim=2)  # (B, S1, N2, G, S2)
    scores_scaled = scores * scale_value
    softmax_max = scores_scaled.max(dim=-1).values  # (B, S1, N2, G)
    softmax_sum = torch.exp(scores_scaled - softmax_max.unsqueeze(-1).float()).sum(dim=-1)  # (B, S1, N2, G)
    # Reshape to (B, N2, S1, G)
    softmax_max_input = softmax_max.permute(0, 2, 1, 3).contiguous()
    softmax_sum_input = softmax_sum.permute(0, 2, 1, 3).contiguous()

    print("\n[1] TileLang CV 分离架构计算...")
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max_input, softmax_sum_input, scale_value,
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

    print("\n[2] 官方 Ascend C 算子计算...")
    try:
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max_input, softmax_sum_input, scale_value,
            layout="BSND",
            sparse_mode=3,
            pre_tokens=2147483647,
            next_tokens=2147483647,
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
        # Loss comparison
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