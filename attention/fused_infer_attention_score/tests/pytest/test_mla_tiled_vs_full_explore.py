#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Explore tiled vs full golden: memory & time tradeoff across S1/S2 matrix.

Memory model (per head, float32 on NPU):
  full:  3 * Sq * Skv * 4B  (scores + exp_scores + probs)
  tiled: ~2 * TILE^2 * 4B + TILE*dv*4B  (fixed, independent of Sq*Skv)

Goal: find crossover where tiled saves memory without major time loss.
"""

import sys
import os
import math
import time
import gc
import argparse

import torch
import torch_npu
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fused_infer_attention_score_golden import FusedInferAttentionScoreGolden

QK_D = 512
ROPE_D = 64
V_D = 512
TILE_SIZE = FusedInferAttentionScoreGolden.TILE_SIZE  # 2048


def _make_inputs(b, n1, n2, s1, s2, dtype, device="npu"):
    torch.manual_seed(42)
    np.random.seed(42)
    query = torch.randn(b, n1, s1, QK_D, dtype=dtype, device=device)
    query_rope = torch.randn(b, n1, s1, ROPE_D, dtype=dtype, device=device)
    kv = torch.randn(b, n2, s2, QK_D, dtype=dtype, device=device)
    key_rope = torch.randn(b, n2, s2, ROPE_D, dtype=dtype, device=device)
    return query, query_rope, kv, key_rope


def _npu_mem_mb():
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated() / (1024 * 1024)
    reserved = torch.npu.memory_reserved() / (1024 * 1024)
    return allocated, reserved


def _run_golden(query, query_rope, kv, key_rope, n1, n2, compute_mode, tile_size=None):
    scale = 1.0 / math.sqrt(QK_D)
    golden = FusedInferAttentionScoreGolden(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=QK_D + ROPE_D,
        softmax_scale=scale,
        tile_size=tile_size,
    )
    golden_out, _ = golden.forward_golden(
        query, kv, kv,
        query_rope=query_rope,
        key_rope=key_rope,
        softmax_scale=scale,
        layout_q="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
        device_mode="npu",
        compute_mode=compute_mode,
    )
    return golden_out


def _bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    mem_before = torch.npu.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    mem_after = torch.npu.max_memory_allocated()
    avg_ms = (t1 - t0) / iters * 1000
    return avg_ms, mem_after / (1024 * 1024)


def _theoretical_mem_full(s1, s2, n1):
    """Peak intermediate memory for full mode (per head): 3 * S1 * S2 * 4B"""
    per_head = 3 * s1 * s2 * 4
    # Processed per head, so peak is 1 head's worth
    return per_head / (1024 * 1024)


def _theoretical_mem_tiled(tile_size=TILE_SIZE):
    """Peak intermediate memory for tiled mode: 2 * TILE^2 * 4B + TILE*dv*4B"""
    return (2 * tile_size * tile_size * 4 + tile_size * V_D * 4) / (1024 * 1024)


def run_case(b, n1, n2, s1, s2, dtype_str="fp16"):
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16

    header = f"B={b} N1={n1} N2={n2} S1={s1} S2={s2}"
    print(f"\n{'─'*80}")
    print(f"  {header}")
    print(f"{'─'*80}")

    # Theoretical analysis
    mem_full_theory = _theoretical_mem_full(s1, s2, n1)
    mem_tiled_theory = _theoretical_mem_tiled()
    attn_matrix_mb = s1 * s2 * 4 / (1024 * 1024)
    print(f"  Attn matrix (S1*S2*4B):  {attn_matrix_mb:.1f} MB")
    print(f"  Full peak inter (1hd):   {mem_full_theory:.1f} MB")
    print(f"  Tiled peak inter:        {mem_tiled_theory:.1f} MB  (fixed, TILE={TILE_SIZE})")
    print(f"  Mem saving:              {mem_full_theory - mem_tiled_theory:.1f} MB  "
          f"({(1 - mem_tiled_theory/max(mem_full_theory,0.01))*100:.0f}%)")

    query, query_rope, kv, key_rope = _make_inputs(b, n1, n2, s1, s2, dtype)

    # Input tensor memory
    input_mem = sum(t.nelement() * t.element_size() for t in [query, query_rope, kv, key_rope]) / (1024*1024)
    print(f"  Input tensors:           {input_mem:.1f} MB")

    results = {}

    # --- Full mode ---
    torch.npu.reset_peak_memory_stats()
    try:
        torch.npu.synchronize()
        full_ms, full_peak = _bench(lambda: _run_golden(query, query_rope, kv, key_rope, n1, n2, "full"))
        print(f"  FULL:  {full_ms:.1f} ms  peak_mem={full_peak:.1f} MB")
        results["full_ms"] = full_ms
        results["full_peak"] = full_peak
    except Exception as e:
        print(f"  FULL:  OOM/ERROR ({type(e).__name__})")
        results["full_ms"] = None
        results["full_peak"] = None

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    # --- Tiled mode (default TILE=2048) ---
    try:
        torch.npu.synchronize()
        tiled_ms, tiled_peak = _bench(lambda: _run_golden(query, query_rope, kv, key_rope, n1, n2, "tiled"))
        print(f"  TILED: {tiled_ms:.1f} ms  peak_mem={tiled_peak:.1f} MB  (TILE={TILE_SIZE})")
        results["tiled_ms"] = tiled_ms
        results["tiled_peak"] = tiled_peak
    except Exception as e:
        print(f"  TILED: OOM/ERROR ({type(e).__name__})")
        results["tiled_ms"] = None
        results["tiled_peak"] = None

    gc.collect()
    torch.npu.empty_cache()

    # --- Comparison ---
    if results["full_ms"] and results["tiled_ms"]:
        ratio = results["tiled_ms"] / results["full_ms"]
        mem_saved = results["full_peak"] - results["tiled_peak"]
        verdict = ""
        if ratio < 1.5 and mem_saved > 50:
            verdict = "*** GOOD: tiled saves mem with minimal time loss"
        elif ratio < 3.0 and mem_saved > 200:
            verdict = "**  ACCEPTABLE: tiled saves significant mem, moderate time loss"
        elif results["full_ms"] is None:
            verdict = "*** MUST USE TILED: full OOMs"
        else:
            verdict = "-   NOT WORTH: tiled too slow or no mem saving"
        print(f"  RATIO: tiled/full = {ratio:.2f}x  mem_saved = {mem_saved:.1f} MB")
        print(f"  {verdict}")
    elif results["full_ms"] is None and results["tiled_ms"] is not None:
        print(f"  *** MUST USE TILED: full OOMs, tiled works")

    del query, query_rope, kv, key_rope
    gc.collect()
    torch.npu.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLA golden tiled vs full memory/time exploration")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device_id)

    print(f"{'#'*80}")
    print(f"# MLA D512 Golden: tiled vs full — Memory & Time Tradeoff")
    print(f"# TILE_SIZE = {TILE_SIZE},  tiled peak inter = {_theoretical_mem_tiled():.1f} MB (fixed)")
    print(f"{'#'*80}")
    print(f"\nLegend:")
    print(f"  Attn matrix = S1*S2*4B (one head's score matrix, fp32)")
    print(f"  Full peak inter = 3 * S1 * S2 * 4B (scores + exp + probs)")
    print(f"  Tiled peak inter = 2*TILE^2*4B + TILE*dv*4B (fixed)")

    # Scan S1 x S2 matrix
    # Rows: S1 from small to large
    # Cols: S2 from small to large
    cases = [
        # (label, B, N1, N2, S1, S2)
        # --- S1 small (current MLA decode scenario) ---
        ("S1=1    S2=512",    1, 2, 1, 1,    512),
        ("S1=1    S2=4K",     1, 2, 1, 1,    4096),
        ("S1=1    S2=16K",    1, 2, 1, 1,    16384),
        ("S1=1    S2=32K",    1, 2, 1, 1,    32768),

        # --- S1 medium ---
        ("S1=128  S2=512",    1, 2, 1, 128,  512),
        ("S1=128  S2=2K",     1, 2, 1, 128,  2048),
        ("S1=128  S2=8K",     1, 2, 1, 128,  8192),
        ("S1=128  S2=16K",    1, 2, 1, 128,  16384),

        # --- S1 large (prefill scenario) ---
        ("S1=1K   S2=1K",     1, 2, 1, 1024, 1024),
        ("S1=1K   S2=4K",     1, 2, 1, 1024, 4096),
        ("S1=1K   S2=8K",     1, 2, 1, 1024, 8192),
        ("S1=2K   S2=2K",     1, 2, 1, 2048, 2048),
        ("S1=2K   S2=8K",     1, 2, 1, 2048, 8192),
        ("S1=4K   S2=4K",     1, 2, 1, 4096, 4096),

        # --- Large N1 (many heads, amplifies per-head overhead) ---
        ("S1=1    S2=16K  N=128", 1, 128, 1, 1,   16384),
        ("S1=128  S2=8K   N=128", 1, 128, 1, 128, 8192),
        ("S1=1K   S2=4K   N=128", 1, 128, 1, 1024, 4096),
        ("S1=2K   S2=2K   N=128", 1, 128, 1, 2048, 2048),
    ]

    print(f"\n{'='*80}")
    print(f"Running {len(cases)} cases...")
    print(f"{'='*80}")

    all_results = []
    for label, b, n1, n2, s1, s2 in cases:
        print(f"\n>>> {label}")
        r = run_case(b, n1, n2, s1, s2)
        all_results.append((label, r))

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"{'SUMMARY':^90}")
    print(f"{'='*90}")
    print(f"{'Case':<28} {'Full ms':>8} {'Tiled ms':>9} {'Ratio':>7} {'Full MB':>8} {'Tiled MB':>9} {'Saved MB':>9}")
    print(f"{'─'*90}")
    for label, r in all_results:
        f_ms = f"{r['full_ms']:.1f}" if r['full_ms'] else "OOM"
        t_ms = f"{r['tiled_ms']:.1f}" if r['tiled_ms'] else "OOM"
        ratio = f"{r['tiled_ms']/r['full_ms']:.2f}x" if r['full_ms'] and r['tiled_ms'] else "N/A"
        f_mb = f"{r['full_peak']:.0f}" if r['full_peak'] else "OOM"
        t_mb = f"{r['tiled_peak']:.0f}" if r['tiled_peak'] else "OOM"
        saved = f"{r['full_peak']-r['tiled_peak']:.0f}" if r['full_peak'] and r['tiled_peak'] else "N/A"
        print(f"{label:<28} {f_ms:>8} {t_ms:>9} {ratio:>7} {f_mb:>8} {t_mb:>9} {saved:>9}")
    print(f"{'='*90}")
