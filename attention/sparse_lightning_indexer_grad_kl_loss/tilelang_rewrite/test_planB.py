#!/usr/bin/env python3
"""Precision and performance tests for Plan B (Dual-kernel + Python bridge)."""

import math
import sys
import os
import time

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilelang_rewrite.sparse_lightning_indexer_grad_kl_loss_planB import (
    sparse_lightning_indexer_grad_kl_loss,
    _clear_caches,
    _profile_result,
    set_block_config,
    reset_block_config,
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


def call_tl(inputs, config, profile=False):
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
        profile=profile,
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
    print("Plan B - Test Suite: K sweep")
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
            import traceback
            traceback.print_exc()
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
    print("Plan B - Performance")
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


def test_profile():
    """Warmup then profile to show steady-state stage timing."""
    print("\n" + "=" * 60)
    print("Plan B - Stage Profiling")
    print("=" * 60)
    cfg = DEFAULT_CONFIG.copy()
    inputs = gen_inputs(cfg)

    # Warmup: trigger JIT compilation without profiling
    print("Warmup (compiling kernels)...")
    _clear_caches()
    call_tl(inputs, cfg, profile=False)
    torch.npu.synchronize()

    # Now profile steady-state execution
    print("Profiling steady-state...")
    call_tl(inputs, cfg, profile=True)
    torch.npu.synchronize()

    pt = _profile_result
    if not pt:
        print("  WARNING: No profile data collected.")
        return

    total = pt["total"]
    print(f"\n  Stage timing breakdown (steady-state, total={total:.3f}ms):")
    print(f"  {'Stage':<25} {'Time(ms)':>10} {'Pct':>8}")
    print(f"  {'-'*43}")
    for name in ["gather_compile", "kernel1", "kernel2", "python"]:
        t = pt[name]
        pct = t / total * 100 if total > 0 else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  {name:<25} {t:>10.3f} {pct:>7.1f}% {bar}")


def test_tune_blocks():
    """Sweep BLOCK parameters and report performance for each."""
    print("\n" + "=" * 60)
    print("Plan B - BLOCK Parameter Tuning")
    print("=" * 60)

    from tilelang_rewrite.sparse_lightning_indexer_grad_kl_loss_planB import (
        BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N1, BLOCK_NIDX1,
    )

    cfg = DEFAULT_CONFIG.copy()
    print(f"\n  Default config: K={cfg['K']}, S1={cfg['S1']}, N1={cfg['N1']}, Nidx1={cfg['Nidx1']}")
    print(f"  Current BLOCK: M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}, N1={BLOCK_N1}, NIDX1={BLOCK_NIDX1}")

    sweeps = [
        # (param, values_to_try)
        ("BLOCK_N", [64, 128, 256]),
        ("BLOCK_N1", [16, 32, 64]),
        ("BLOCK_NIDX1", [8, 16, 32]),
    ]

    default_map = {"BLOCK_N": 128, "BLOCK_N1": 32, "BLOCK_NIDX1": 16}

    for param_name, values in sweeps:
        print(f"\n  --- Sweep {param_name} ---")
        print(f"  {'Value':>6}  {'n_num':>6}  {'Avg(ms)':>10}  {'Med(ms)':>10}  {'Min(ms)':>10}  Speedup")
        print(f"  {'-'*60}")

        baseline_med = None
        for val in values:
            reset_block_config()
            _clear_caches()
            if param_name == "BLOCK_N":
                set_block_config(n=val)
            elif param_name == "BLOCK_N1":
                set_block_config(n1=val)
            elif param_name == "BLOCK_NIDX1":
                set_block_config(nidx1=val)

            n_num = cfg["K"] // BLOCK_N
            inputs = gen_inputs(cfg)
            torch.npu.synchronize()

            try:
                avg, med, mn = bench(lambda: call_tl(inputs, cfg))
                torch.npu.synchronize()
            except Exception as e:
                print(f"  {val:>6}  ERROR: {e}")
                continue

            if baseline_med is None:
                baseline_med = med

            speedup = baseline_med / med if med > 0 else float("inf")
            is_default = val == default_map.get(param_name, -1)

            print(f"  {val:>6}  {n_num:>6}  {avg*1e3:>10.3f}  {med*1e3:>10.3f}  {mn*1e3:>10.3f}  {speedup:>6.2f}x"
                  f"{' (default)' if is_default else ''}")

    # Restore defaults
    reset_block_config()
    _clear_caches()
    print("\n  Restored default BLOCK config.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Plan B (Dual-kernel + Python bridge)")
    parser.add_argument("--precision-only", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick test: single config")
    parser.add_argument("--profile", action="store_true", help="Profile: show stage-by-stage timing breakdown")
    parser.add_argument("--tune-blocks", action="store_true", help="Sweep BLOCK_N, BLOCK_N1, BLOCK_NIDX1 for best perf")
    args = parser.parse_args()

    if args.profile:
        test_profile()
        return

    if args.tune_blocks:
        test_tune_blocks()
        return

    if args.quick:
        print("Plan B - Quick test: single config")
        cfg = DEFAULT_CONFIG.copy()
        inputs = gen_inputs(cfg)
        print("[1] TileLang Plan B...")
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
