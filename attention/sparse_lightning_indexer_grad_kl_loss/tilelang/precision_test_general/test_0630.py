#!/usr/bin/env python3
"""
Case 0 测试 - 基于0630.py (gather移入kernel V0 scope)
"""
import torch
import torch_npu
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
mod = import_module("0630")
sparse_lightning_indexer_grad_kl_loss = mod.sparse_lightning_indexer_grad_kl_loss

# Clear all caches to avoid stale compiled kernels
mod._clear_caches()

WARMUP_ITERS = 3
BENCH_ITERS = 5

CASE_CONFIG = {
    "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
    "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
    "d_rope": 64, "K": 2048, "layout": "BSND",
}


def gen_inputs(seed=42):
    B = CASE_CONFIG["B"]
    S1 = CASE_CONFIG["S1"]
    S2 = CASE_CONFIG["S2"]
    N1 = CASE_CONFIG["N1"]
    N2 = CASE_CONFIG["N2"]
    Nidx1 = CASE_CONFIG["Nidx1"]
    Nidx2 = CASE_CONFIG["Nidx2"]
    DQuery = CASE_CONFIG["DQuery"]
    DQueryIndex = CASE_CONFIG["DQueryIndex"]
    d_rope = CASE_CONFIG["d_rope"]
    K = CASE_CONFIG["K"]

    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    torch.manual_seed(seed)
    dtype = torch.float16

    query = torch.randn(B, S1, N1, DQuery, dtype=dtype).npu()
    key = torch.randn(B, S2, N2, DQuery, dtype=dtype).npu()
    query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).npu()
    key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).npu()
    query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).npu()
    key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).npu()
    weights = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
    sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).npu()

    s2_real_len = min(S2, K)
    indices = torch.randperm(S2)[:s2_real_len]
    for s1_idx in range(S1):
        sparse_indices[0, s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
        if s2_real_len < K:
            sparse_indices[0, s1_idx, 0, s2_real_len:] = -1
    torch.npu.synchronize()

    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        sm_max_list = []
        sm_sum_list = []
        for s1_idx in range(S1):
            row = torch.matmul(q_full[:, s1_idx:s1_idx+1], k_full_t)
            row_scaled = row.float() * scale_value
            row_max = row_scaled.max(dim=-1).values
            row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
            sm_max_list.append(row_max)
            sm_sum_list.append(row_sum)
        sm_max_val = torch.cat(sm_max_list, dim=1)
        sm_sum_val = torch.cat(sm_sum_list, dim=1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()

    return {
        "query": query, "key": key,
        "query_index": query_index, "key_index": key_index,
        "weights": weights, "sparse_indices": sparse_indices,
        "softmax_max": softmax_max, "softmax_sum": softmax_sum,
        "scale_value": scale_value,
        "query_rope": query_rope, "key_rope": key_rope,
    }


def call_tl(inputs):
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        layout_query="BSND",
        actual_seq_lengths_query=[CASE_CONFIG["S1"]] * CASE_CONFIG["B"],
        actual_seq_lengths_key=[CASE_CONFIG["S2"]] * CASE_CONFIG["B"],
        sparse_mode=3,
    )


def call_official(inputs):
    return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        actual_seq_qlen=[CASE_CONFIG["S1"]] * CASE_CONFIG["B"],
        actual_seq_klen=[CASE_CONFIG["S2"]] * CASE_CONFIG["B"],
        layout="BSND", sparse_mode=3,
        pre_tokens=65536, next_tokens=65536,
    )


def bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    arr = sorted(times)
    med = arr[len(arr) // 2]
    avg = sum(arr) / len(arr)
    mn = arr[0]
    return avg, med, mn


def main():
    print("=" * 60)
    print("Case 0 - 0630.py (gather移入kernel V0 scope) 测试")
    print("=" * 60)

    inputs = gen_inputs()

    # TileLang
    print("\n[1] TileLang (0630 kernel-gather)...")
    torch.npu.synchronize()
    try:
        d_qi_tl, d_ki_tl, dw_tl, loss_tl = call_tl(inputs)
        torch.npu.synchronize()
        print("  完成")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 官方
    print("\n[2] 官方...")
    torch.npu.synchronize()
    d_qi_off, d_ki_off, dw_off, loss_off = call_official(inputs)
    torch.npu.synchronize()
    print("  完成")

    # 精度
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
        print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, rel_err={rel_err:.6e}")

    loss_abs_diff = abs(loss_tl.float().item() - loss_off.float().item())
    loss_rel_err = loss_abs_diff / (abs(loss_off.float().item()) + 1e-6)
    print(f"  Loss: abs_diff={loss_abs_diff:.6e}, rel_err={loss_rel_err:.6e}")

    # 性能
    print(f"\n[4] 性能测试 (warmup={WARMUP_ITERS}, bench={BENCH_ITERS})...")
    torch.npu.synchronize()
    tl_avg, tl_med, tl_min = bench(lambda: call_tl(inputs))
    torch.npu.synchronize()
    off_avg, off_med, off_min = bench(lambda: call_official(inputs))
    speedup = off_med / tl_med
    print(f"  TileLang: avg={tl_avg*1e3:.3f}ms, med={tl_med*1e3:.3f}ms, min={tl_min*1e3:.3f}ms")
    print(f"  官方:     avg={off_avg*1e3:.3f}ms, med={off_med*1e3:.3f}ms, min={off_min*1e3:.3f}ms")
    print(f"  加速比:   {speedup:.2f}x (基于中位数)")


if __name__ == "__main__":
    main()
