#!/usr/bin/env python3
"""
泛化精度测试 - SparseLightningIndexerGradKLLoss TileLang

覆盖多种配置：BSND/TND layout、不同 B/S1/S2/N1/Nidx1/K 组合
"""

import torch
import torch_npu
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss


def run_single_test(config_name, B, S1, S2, N1, N2, Nidx1, Nidx2,
                    DQuery, DQueryIndex, d_rope, K, layout="BSND", seed=42):
    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    torch.manual_seed(seed)

    query = torch.randn(B, S1, N1, DQuery, dtype=torch.float16).npu()
    key = torch.randn(B, S2, N2, DQuery, dtype=torch.float16).npu()
    query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=torch.float16).npu()
    key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=torch.float16).npu()
    weights = torch.randn(B, S1, Nidx1, dtype=torch.float16).npu()
    query_rope = torch.randn(B, S1, N1, d_rope, dtype=torch.float16).npu()
    key_rope = torch.randn(B, S2, N2, d_rope, dtype=torch.float16).npu()

    sparse_indices = torch.full((B, S1, Nidx2, K), -1, dtype=torch.int32).npu()
    for b in range(B):
        for s in range(S1):
            for i in range(min(S2, K)):
                sparse_indices[b, s, 0, i] = i

    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        bmm1_full = torch.matmul(q_full, k_full.transpose(-1, -2))
        bmm1_scaled = bmm1_full.float() * scale_value
        sm_max_val = bmm1_scaled.max(dim=-1).values
        sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()

    if layout == "TND":
        query_in = query.squeeze(0) if B == 1 else query
        key_in = key.squeeze(0) if B == 1 else key
        query_index_in = query_index.squeeze(0) if B == 1 else query_index
        key_index_in = key_index.squeeze(0) if B == 1 else key_index
        weights_in = weights.squeeze(0) if B == 1 else weights
        sparse_indices_in = sparse_indices.squeeze(0) if B == 1 else sparse_indices
        softmax_max_in = softmax_max.squeeze(0) if B == 1 else softmax_max
        softmax_sum_in = softmax_sum.squeeze(0) if B == 1 else softmax_sum
        query_rope_in = query_rope.squeeze(0) if B == 1 else query_rope
        key_rope_in = key_rope.squeeze(0) if B == 1 else key_rope
        tl_layout = "TND"
    else:
        query_in = query
        key_in = key
        query_index_in = query_index
        key_index_in = key_index
        weights_in = weights
        sparse_indices_in = sparse_indices
        softmax_max_in = softmax_max
        softmax_sum_in = softmax_sum
        query_rope_in = query_rope
        key_rope_in = key_rope
        tl_layout = "BSND"

    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
            query_in, key_in, query_index_in, key_index_in, weights_in, sparse_indices_in,
            softmax_max_in, softmax_sum_in, scale_value,
            query_rope=query_rope_in,
            key_rope=key_rope_in,
            layout_query=tl_layout,
            sparse_mode=3,
        )
        torch.npu.synchronize()
        tl_ok = True
    except Exception as e:
        print(f"  TileLang 失败: {e}")
        tl_ok = False

    try:
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query_in, key_in, query_index_in, key_index_in, weights_in, sparse_indices_in,
            softmax_max_in, softmax_sum_in, scale_value,
            query_rope=query_rope_in,
            key_rope=key_rope_in,
            layout=tl_layout,
            sparse_mode=3,
        )
        torch.npu.synchronize()
        off_ok = True
    except Exception as e:
        print(f"  官方算子失败: {e}")
        off_ok = False

    if tl_ok and off_ok:
        results = {}
        for name, tl_val, off_val in [
            ("dQueryIndex", d_qi_tl, d_qi_off),
            ("dKeyIndex", d_ki_tl, d_ki_off),
            ("dWeights", dw_tl, dw_off),
        ]:
            diff = (tl_val.float() - off_val.float()).abs()
            results[name] = {
                "max_diff": diff.max().item(),
                "mean_diff": diff.mean().item(),
                "rel_err": (diff / (off_val.float().abs() + 1e-6)).mean().item(),
            }
        loss_diff = abs(loss_tl.float().item() - loss_off.float().item())
        results["Loss"] = {"abs_diff": loss_diff, "tl": loss_tl.item(), "off": loss_off.item()}
        return results
    return None


def print_result(config_name, results):
    if results is None:
        print(f"  {config_name}: 无法对比")
        return
    passed = True
    for name in ["dQueryIndex", "dKeyIndex", "dWeights"]:
        r = results[name]
        ok = r["max_diff"] < 0.01
        if not ok:
            passed = False
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: max_diff={r['max_diff']:.6f}, rel_err={r['rel_err']:.6f} [{status}]")
    r = results["Loss"]
    ok = r["abs_diff"] < 0.01
    if not ok:
        passed = False
    status = "PASS" if ok else "FAIL"
    print(f"  Loss: abs_diff={r['abs_diff']:.6f} [{status}]")
    return passed


if __name__ == "__main__":
    print("=" * 70)
    print("泛化精度测试 - SparseLightningIndexerGradKLLoss TileLang")
    print("=" * 70)

    test_configs = [
        # (name, B, S1, S2, N1, N2, Nidx1, Nidx2, DQuery, DQueryIndex, d_rope, K, layout)
        # BSND - 标准配置
        ("BSND_S1=1_S2=2048_N1=128_Nidx1=64_K=2048",
         1, 1, 2048, 128, 1, 64, 1, 512, 128, 64, 2048, "BSND"),
        # BSND - 小 K
        ("BSND_S1=1_S2=1024_N1=128_Nidx1=64_K=1024",
         1, 1, 1024, 128, 1, 64, 1, 512, 128, 64, 1024, "BSND"),
        # BSND - 大 K
        ("BSND_S1=1_S2=4096_N1=128_Nidx1=64_K=4096",
         1, 1, 4096, 128, 1, 64, 1, 512, 128, 64, 4096, "BSND"),
        # BSND - 小 N1
        ("BSND_S1=1_S2=2048_N1=64_Nidx1=32_K=2048",
         1, 1, 2048, 64, 1, 32, 1, 512, 128, 64, 2048, "BSND"),
        # BSND - 最小 N1
        ("BSND_S1=1_S2=1024_N1=32_Nidx1=16_K=1024",
         1, 1, 1024, 32, 1, 16, 1, 512, 128, 64, 1024, "BSND"),
        # BSND - S1 > 1
        ("BSND_S1=2_S2=2048_N1=128_Nidx1=64_K=2048",
         1, 2, 2048, 128, 1, 64, 1, 512, 128, 64, 2048, "BSND"),
        # BSND - B > 1
        ("BSND_B=2_S1=1_S2=2048_N1=128_Nidx1=64_K=2048",
         2, 1, 2048, 128, 1, 64, 1, 512, 128, 64, 2048, "BSND"),
        # BSND - 小 Nidx1
        ("BSND_S1=1_S2=2048_N1=128_Nidx1=8_K=2048",
         1, 1, 2048, 128, 1, 8, 1, 512, 128, 64, 2048, "BSND"),
        # TND - 标准配置
        ("TND_S2=2048_N1=128_Nidx1=64_K=2048",
         1, 1, 2048, 128, 1, 64, 1, 512, 128, 64, 2048, "TND"),
        # TND - 小 K
        ("TND_S2=1024_N1=64_Nidx1=32_K=1024",
         1, 1, 1024, 64, 1, 32, 1, 512, 128, 64, 1024, "TND"),
        # BSND - K=3072
        ("BSND_S1=1_S2=3072_N1=128_Nidx1=64_K=3072",
         1, 1, 3072, 128, 1, 64, 1, 512, 128, 64, 3072, "BSND"),
        # BSND - K=5120
        ("BSND_S1=1_S2=5120_N1=128_Nidx1=64_K=5120",
         1, 1, 5120, 128, 1, 64, 1, 512, 128, 64, 5120, "BSND"),
        # BSND - K=6144
        ("BSND_S1=1_S2=6144_N1=128_Nidx1=64_K=6144",
         1, 1, 6144, 128, 1, 64, 1, 512, 128, 64, 6144, "BSND"),
        # BSND - K=7168
        ("BSND_S1=1_S2=7168_N1=128_Nidx1=64_K=7168",
         1, 1, 7168, 128, 1, 64, 1, 512, 128, 64, 7168, "BSND"),
        # BSND - K=8192
        ("BSND_S1=1_S2=8192_N1=128_Nidx1=64_K=8192",
         1, 1, 8192, 128, 1, 64, 1, 512, 128, 64, 8192, "BSND"),
    ]

    all_passed = True
    for cfg in test_configs:
        name = cfg[0]
        print(f"\n{'─' * 70}")
        print(f"测试: {name}")
        print(f"  配置: B={cfg[1]}, S1={cfg[2]}, S2={cfg[3]}, N1={cfg[4]}, N2={cfg[5]}, "
              f"Nidx1={cfg[6]}, Nidx2={cfg[7]}, DQuery={cfg[8]}, DQueryIndex={cfg[9]}, "
              f"d_rope={cfg[10]}, K={cfg[11]}, layout={cfg[12]}")
        results = run_single_test(*cfg)
        passed = print_result(name, results)
        if passed is False:
            all_passed = False

    print(f"\n{'=' * 70}")
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查上方输出。")