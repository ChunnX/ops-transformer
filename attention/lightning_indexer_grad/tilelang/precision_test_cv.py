#!/usr/bin/env python3
"""
精度测试 - Lightning Indexer Grad TileLang CV 分离架构

对比 TileLang CV 分离实现与官方 Ascend C 算子的结果。
"""

import torch
import torch_npu
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightning_indexer_grad_cv import LightningIndexerGradCV


def run_precision_test():
    print("=" * 70)
    print("精度测试 - Lightning Indexer Grad TileLang CV 分离架构")
    print("=" * 70)

    batch = 1
    seqlen_q = 64
    seqlen_k = 2048
    head_dim = 128
    topk = 2048
    head_num_q = 64
    head_num_k = 1
    group_num = 64

    block_m = 64
    block_n = 128
    block_d = 128

    print(f"\n测试配置 (官方算子支持):")
    print(f"  batch={batch}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  head_dim={head_dim}, topk={topk}")
    print(f"  head_num_q={head_num_q}, head_num_k={head_num_k}, group_num={group_num}")
    print(f"  block_m={block_m}, block_n={block_n}, block_d={block_d}")

    op = LightningIndexerGradCV(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        topk=topk,
        head_num_q=head_num_q,
        head_num_k=head_num_k,
        group_num=group_num,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
    )

    torch.manual_seed(42)

    query = torch.randn(batch, seqlen_q, head_num_q, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seqlen_k, head_num_k, head_dim, dtype=torch.float16).npu()
    weights = torch.randn(batch, seqlen_q, head_num_q, dtype=torch.float16).npu()
    dy = torch.randn(batch, seqlen_q, topk, dtype=torch.float16).npu()
    indices = torch.randint(0, seqlen_k, (batch, seqlen_q, topk), dtype=torch.int32).npu()

    torch.npu.synchronize()

    print("\n[1] TileLang CV 分离架构计算...")
    try:
        d_query_tl, d_key_tl, d_weights_tl = op.forward(
            query, key, dy, indices, weights,
            head_num=head_num_q,
            layout="BSND",
            sparse_mode=3,
        )
        torch.npu.synchronize()
        print("  完成")
        tl_available = True
    except Exception as e:
        print(f"  失败: {e}")
        tl_available = False

    print("\n[2] 官方 Ascend C 算子计算...")
    try:
        d_query_off, d_key_off, d_weights_off = torch_npu.npu_lightning_indexer_grad(
            query, key, dy, indices, weights,
            layout="BSND",
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
            ("dQuery", d_query_tl, d_query_off),
            ("dKey", d_key_tl, d_key_off),
            ("dWeights", d_weights_tl, d_weights_off),
        ]:
            diff = (tl_val.float() - off_val.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
            print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.6f}")
    elif tl_available:
        print("\n[3] 官方算子不可用，仅验证 TileLang 输出非全零...")
        for name, val in [("dQuery", d_query_tl), ("dKey", d_key_tl), ("dWeights", d_weights_tl)]:
            print(f"  {name}: max={val.abs().max().item():.6f}, nonzero={val.count_nonzero().item()}")
    else:
        print("\n[3] TileLang 计算失败，无法对比")


if __name__ == "__main__":
    run_precision_test()