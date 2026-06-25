#!/usr/bin/env python3
"""Precision and performance tests for Plan A (K-parallel kernel + Python postprocessing)."""

import math
import sys
import os
import time
import itertools

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilelang_rewrite.sparse_lightning_indexer_grad_kl_loss_planA import (
    sparse_lightning_indexer_grad_kl_loss,
    _clear_caches,
)

WARMUP_ITERS = 3
BENCH_ITERS = 5

DEFAULT_CONFIG = {
    "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
    "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
    "d_rope": 64, "K": 2048,
}


def gen_softmax_max_sum(query, query_rope, key, key_rope, scale_value, B, S1, N1, N2, G):
    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        sm_max_list = []
        sm_sum_list = []
        for s in range(S1):
            row = torch.matmul(q_full[:, s:s + 1], k_full_t)
            row_scaled = row.float() * scale_value
            row_max = row_scaled.max(dim=-1).values
            row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
            sm_max_list.append(row_max)
            sm_sum_list.append(row_sum)
        sm_max_val = torch.cat(sm_max_list, dim=1)
        sm_sum_val = torch.cat(sm_sum_list, dim=1)
        sm_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        sm_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
    return sm_max, sm_sum


def gen_inputs(config, seed=42):
    B, S1, S2 = config["B"], config["S1"], config["S2"]
    N1, N2 = config["N1"], config["N2"]
    Nidx1, Nidx2 = config["Nidx1"], config["Nidx2"]
    DQuery, DQueryIndex = config["DQuery"], config["DQueryIndex"]
    d_rope, K = config["d_rope"], config["K"]
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

    softmax_max, softmax_sum = gen_softmax_max_sum(
        query, query_rope, key, key_rope, scale_value, B, S1, N1, N2, G,
    )

    return {
        "query": query, "key": key,
        "query_index": query_index, "key_index": key_index,
        "weights": weights, "sparse_indices": sparse_indices,
        "softmax_max": softmax_max, "softmax_sum": softmax_sum,
        "scale_value": scale_value,
        "query_rope": query_rope, "key_rope": key_rope,
    }


def call_tl(inputs, config):
    _clear_caches()
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        layout_query="BSND",
        actual_seq_lengths_query=[config["S1"]] * config["B"],
        actual_seq_lengths_key=[config["S2"]] * config["B"],
        sparse_mode=3,
    )


def call_official(inputs, config):
    return torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        actual_seq_qlen=[config["S1"]] * config["B"],
        actual_seq_klen=[config["S2"]] * config["B"],
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


def check_precision(name, tl_val, off_val, tol_rel=0.05):
    diff = (tl_val.float() - off_val.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = (diff / (off_val.float().abs() + 1e-6)).mean().item()
    passed = rel_err < tol_rel
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name}: max={max_diff:.4e} mean={mean_diff:.4e} rel={rel_err:.4e} (tol={tol_rel})")
    return passed


def check_loss_precision(tl_loss, off_loss, tol_rel=0.01):
    abs_diff = abs(tl_loss.float().item() - off_loss.float().item())
    rel_err = abs_diff / (abs(off_loss.float().item()) + 1e-6)
    passed = rel_err < tol_rel
    status = "PASS" if passed else "FAIL"
    print(f"  {status} Loss: abs={abs_diff:.4e} rel={rel_err:.4e} (tol={tol_rel})")
    return passed


def test_k_sweep():
    print("\n" + "=" * 60)
    print("Plan A - Test Suite: K sweep")
    print("=" * 60)
    all_pass = True
    for K in [1024, 2048, 4096, 8192]:
        cfg = {**DEFAULT_CONFIG, "K": K, "S2": K}
        print(f"\n--- K={K} ---")
        inputs = gen_inputs(cfg)
        try:
            d_qi, d_ki, dw, loss = call_tl(inputs, cfg)
            torch.npu.synchronize()
        except Exception as e:
            print(f"  ERROR: {e}")
            all_pass = False
            continue
        d_qi_o, d_ki_o, dw_o, loss_o = call_official(inputs, cfg)
        torch.npu.synchronize()
        all_pass &= check_precision("dQueryIndex", d_qi, d_qi_o)
        all_pass &= check_precision("dKeyIndex", d_ki, d_ki_o)
        all_pass &= check_precision("dWeights", dw, dw_o)
        all_pass &= check_loss_precision(loss, loss_o)
    return all_pass


def test_performance():
    print("\n" + "=" * 60)
    print("Plan A - Performance")
    print("=" * 60)

    configs = [
        {**DEFAULT_CONFIG, "desc": "Base (K=2048)"},
        {**DEFAULT_CONFIG, "K": 4096, "S2": 4096, "desc": "K=4096"},
        {**DEFAULT_CONFIG, "K": 8192, "S2": 8192, "desc": "K=8192"},
        {**DEFAULT_CONFIG, "S1": 16, "desc": "S1=16, K=2048"},
    ]

    for cfg in configs:
        desc = cfg.pop("desc")
        print(f"\n--- {desc} ---")
        inputs = gen_inputs(cfg)
        torch.npu.synchronize()
        tl_avg, tl_med, tl_min = bench(lambda: call_tl(inputs, cfg))
        torch.npu.synchronize()
        off_avg, off_med, off_min = bench(lambda: call_official(inputs, cfg))
        speedup = off_med / tl_med if tl_med > 0 else float("inf")
        print(f"  TileLang: avg={tl_avg*1e3:.3f}ms  med={tl_med*1e3:.3f}ms  min={tl_min*1e3:.3f}ms")
        print(f"  Official: avg={off_avg*1e3:.3f}ms  med={off_med*1e3:.3f}ms  min={off_min*1e3:.3f}ms")
        print(f"  Speedup:  {speedup:.2f}x (median)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Plan A (K-parallel + Python)")
    parser.add_argument("--precision-only", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick test: single config")
    args = parser.parse_args()

    if args.quick:
        print("Plan A - Quick test: single config")
        cfg = DEFAULT_CONFIG.copy()
        inputs = gen_inputs(cfg)
        print("[1] TileLang Plan A...")
        d_qi, d_ki, dw, loss = call_tl(inputs, cfg)
        torch.npu.synchronize()
        print("[2] Official...")
        d_qi_o, d_ki_o, dw_o, loss_o = call_official(inputs, cfg)
        torch.npu.synchronize()
        print("[3] Precision:")
        check_precision("dQueryIndex", d_qi, d_qi_o)
        check_precision("dKeyIndex", d_ki, d_ki_o)
        check_precision("dWeights", dw, dw_o)
        check_loss_precision(loss, loss_o)
        print("[4] Performance:")
        tl_avg, tl_med, tl_min = bench(lambda: call_tl(inputs, cfg))
        off_avg, off_med, off_min = bench(lambda: call_official(inputs, cfg))
        print(f"  TileLang: med={tl_med*1e3:.3f}ms  Official: med={off_med*1e3:.3f}ms  speedup={off_med/tl_med:.2f}x")
        return

    if args.perf_only:
        test_performance()
    elif args.precision_only:
        test_k_sweep()
    else:
        test_k_sweep()
        test_performance()


if __name__ == "__main__":
    main()
