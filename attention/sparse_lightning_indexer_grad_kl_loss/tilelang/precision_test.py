#!/usr/bin/env python3
"""
精度+性能测试 - SparseLightningIndexerGradKLLoss TileLang BSND layout

对比 TileLang 实现与官方 Ascend C 算子的结果和性能，使用 BSND layout。
直接构造 BSND 格式入参，不经过 TND 转换。
"""

import torch
import torch_npu
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss
from sparse_lightning_indexer_grad_kl_loss_pass import sparse_lightning_indexer_grad_kl_loss_pass


def run_precision_test():
    print("=" * 70)
    print("精度测试 - SparseLightningIndexerGradKLLoss TileLang (BSND layout)")
    print("=" * 70)

    B = 1
    S1 = 1
    S2 = 2048
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

    scale_value = 1.0 / math.sqrt(DQuery)

    # Compute correct softmax_max/sum from actual query/key (full FlashAttention with rope)
    # score = [Query||QueryRope] @ [Key||KeyRope]^T
    with torch.no_grad():
        k_mqa = key.squeeze(2)  # (B, S2, DQuery) for MQA
        k_rope_mqa = key_rope.squeeze(2)  # (B, S2, d_rope)
        q_full = torch.cat([query, query_rope], dim=-1)  # (B, S1, N1, DQuery+d_rope)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)  # (B, S2, DQuery+d_rope)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        bmm1_full = torch.matmul(q_full, k_full_t)  # (B, S1, N1, S2) on NPU
        bmm1_scaled = bmm1_full.float() * scale_value
        sm_max_val = bmm1_scaled.max(dim=-1).values  # (B, S1, N1)
        sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)  # (B, S1, N1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
    print(f"  softmax_max range: [{softmax_max.min().item():.6f}, {softmax_max.max().item():.6f}]")
    print(f"  softmax_sum range: [{softmax_sum.min().item():.6f}, {softmax_sum.max().item():.6f}]")

    print(f"\nBSND Tensor shapes:")
    print(f"  query: {query.shape}, key: {key.shape}")
    print(f"  query_index: {query_index.shape}, key_index: {key_index.shape}")
    print(f"  weights: {weights.shape}")
    print(f"  query_rope: {query_rope.shape}, key_rope: {key_rope.shape}")
    print(f"  sparse_indices: {sparse_indices.shape}, dtype={sparse_indices.dtype}")
    print(f"  softmax_max: {softmax_max.shape}, softmax_sum: {softmax_sum.shape}")
    print(f"  scale_value: {scale_value}")

    print("\n[1] TileLang BSND 计算 (with rope)...")
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
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

    print("\n[2] 官方 Ascend C 算子计算 (BSND layout, with rope)...")
    try:
        d_qi_off, d_ki_off, dw_off, loss_off = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query, key, query_index, key_index, weights, sparse_indices,
            softmax_max, softmax_sum, scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=[S1] * B,
            actual_seq_klen=[S2] * B,
            layout="BSND",
            sparse_mode=3,
        )
        torch.npu.synchronize()
        print("  完成 (with rope)")
        official_available = True
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
        official_available = False

    if tl_available and official_available:
        print("\n[3] 精度对比 (TileLang vs 官方 with rope)...")
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
        loss_tl_val = loss_tl.float().item()
        loss_off_val = loss_off.float().item()
        loss_abs_diff = abs(loss_tl_val - loss_off_val)
        loss_rel_err = loss_abs_diff / (abs(loss_off_val) + 1e-6)
        if math.isnan(loss_rel_err) or math.isinf(loss_rel_err):
            loss_rel_err = float('inf')
        print(f"  Loss: abs_diff={loss_abs_diff:.6f}, rel_err={loss_rel_err:.6f}, tl={loss_tl_val:.6f}, off={loss_off_val:.6f}")
    elif tl_available:
        print("\n[3] 官方算子不可用，仅验证 TileLang 输出非全零...")
        for name, val in [("dQueryIndex", d_qi_tl), ("dKeyIndex", d_ki_tl), ("dWeights", dw_tl)]:
            print(f"  {name}: max={val.abs().max().item():.6f}, nonzero={val.count_nonzero().item()}")
        print(f"  Loss: value={loss_tl.item():.6f}")
    else:
        print("\n[3] TileLang 计算失败，无法对比")


def run_performance_test():
    print("=" * 70)
    print("性能测试 - TileLang vs TileLang_pass vs 官方AscendC")
    print("=" * 70)

    B = 1
    S1 = 1
    S2 = 2048
    N1 = 128
    N2 = 1
    Nidx1 = 64
    Nidx2 = 1
    DQuery = 512
    DQueryIndex = 128
    d_rope = 64
    K = 2048
    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    warmup_iters = 5
    bench_iters = 50

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

    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        bmm1_full = torch.matmul(q_full, k_full_t)
        bmm1_scaled = bmm1_full.float() * scale_value
        sm_max_val = bmm1_scaled.max(dim=-1).values
        sm_sum_val = torch.exp(bmm1_scaled - sm_max_val.unsqueeze(-1)).sum(dim=-1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()

    common_kwargs = dict(
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND", sparse_mode=3,
    )
    off_kwargs = dict(
        query_rope=query_rope, key_rope=key_rope,
        actual_seq_qlen=[S1] * B, actual_seq_klen=[S2] * B,
        layout="BSND", sparse_mode=3,
    )

    results = {}

    # --- TileLang (current) ---
    print(f"\n[1] TileLang (current) - {warmup_iters} warmup + {bench_iters} iters")
    try:
        for _ in range(warmup_iters):
            sparse_lightning_indexer_grad_kl_loss(
                query, key, query_index, key_index, weights, sparse_indices,
                softmax_max, softmax_sum, scale_value, **common_kwargs)
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            sparse_lightning_indexer_grad_kl_loss(
                query, key, query_index, key_index, weights, sparse_indices,
                softmax_max, softmax_sum, scale_value, **common_kwargs)
            torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000
        results["TileLang(current)"] = elapsed
        print(f"  avg: {elapsed:.3f} ms/iter")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- TileLang (pass) ---
    print(f"\n[2] TileLang (pass) - {warmup_iters} warmup + {bench_iters} iters")
    try:
        for _ in range(warmup_iters):
            sparse_lightning_indexer_grad_kl_loss_pass(
                query, key, query_index, key_index, weights, sparse_indices,
                softmax_max, softmax_sum, scale_value, **common_kwargs)
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            sparse_lightning_indexer_grad_kl_loss_pass(
                query, key, query_index, key_index, weights, sparse_indices,
                softmax_max, softmax_sum, scale_value, **common_kwargs)
            torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000
        results["TileLang(pass)"] = elapsed
        print(f"  avg: {elapsed:.3f} ms/iter")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Official AscendC ---
    print(f"\n[3] 官方AscendC - {warmup_iters} warmup + {bench_iters} iters")
    try:
        for _ in range(warmup_iters):
            torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                query, key, query_index, key_index, weights, sparse_indices,
                softmax_max, softmax_sum, scale_value, **off_kwargs)
            torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                query, key, query_index, key_index, weights, sparse_indices,
                softmax_max, softmax_sum, scale_value, **off_kwargs)
            torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000
        results["Official"] = elapsed
        print(f"  avg: {elapsed:.3f} ms/iter")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Summary ---
    if results:
        print("\n" + "=" * 50)
        print("性能对比总结:")
        print("=" * 50)
        baseline = results.get("Official")
        for name, ms in results.items():
            ratio = f"  (vs Official: {ms/baseline:.2f}x)" if baseline and name != "Official" else ""
            print(f"  {name:20s}: {ms:8.3f} ms/iter{ratio}")


if __name__ == "__main__":
    run_precision_test()
    print()
    run_performance_test()
