#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`report` mode: run all PERF_CASE_NAMES and print a comparison table."""

import sys

from ...model.hardware import AscendHWSpec
from ...model.orchestration import estimate_all_stages
from ...pipeline.dag import build_pipeline_dag
from ...pipeline.kernel_time import estimate_kernel_time
from ...tiling.config import compute_tiling
from ..formatting import print_header, params_from_testcase


def run(hw: AscendHWSpec):
    """Mode 6: Run all performance test cases and print a comparison table."""
    print_header(f"PERFORMANCE REPORT — All Perf Test Cases ({hw.name})")

    try:
        from testcases import TEST_PARAMS, PERF_CASE_NAMES
    except ImportError:
        print("Error: cannot import testcases.py from ../pytest/", file=sys.stderr)
        sys.exit(1)

    # Table header
    cols = (
        f"{'Case':<28} {'B':>3} {'S':>4} {'T':>5} {'N':>4} {'He':>5} "
        f"{'Quant':<12} {'Step':>4} {'Steps':>5} "
        f"{'MM1':>7} {'MM2':>7} {'MM3':>7} {'MM4':>7} "
        f"{'Total':>9} {'Bound':<10}"
    )
    print(cols)
    print("-" * len(cols))

    results = []
    for name in PERF_CASE_NAMES:
        if name not in TEST_PARAMS:
            print(f"  WARNING: '{name}' not found in TEST_PARAMS, skipping")
            continue

        params = params_from_testcase(name)
        tiling = compute_tiling(params, hw)
        stages = estimate_all_stages(params, tiling, hw)
        pipeline = build_pipeline_dag(params, tiling, hw, stages)
        total_us = estimate_kernel_time(params, tiling, hw)

        # Identify which MM dominates
        mm_times = {
            "MM1": stages["MM1_Cq"].total_us,
            "MM2": stages["MM2_CkvKr"].total_us,
            "MM3": stages["MM3_QcQr"].total_us,
            "MM4": stages["MM4_Qn"].total_us,
        }
        dominant = max(mm_times, key=mm_times.get)
        dominant_bound = stages[{"MM1": "MM1_Cq", "MM2": "MM2_CkvKr",
                                 "MM3": "MM3_QcQr", "MM4": "MM4_Qn"}[dominant]].bound

        quant_label = params.quant_mode.name
        # Shorten for display
        quant_short = {
            "NO_QUANT": "BF16",
            "PARTIAL_QUANT_KV_NO_QUANT": "INT8-P",
            "PARTIAL_QUANT_KV_QUANT_PER_CHANNEL": "INT8-PCh",
            "FULL_QUANT_KV_NO_QUANT": "INT8-F",
            "FULL_QUANT_KV_QUANT_PER_TENSOR": "INT8-FT",
            "FULL_QUANT_KV_QUANT_PERTILE": "INT8-FTi",
            "MXFP8_FULL_QUANT_KV_NO_QUANT": "FP8",
            "MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR": "FP8-T",
            "MXFP8_FULL_QUANT_KV_QUANT_PER_TILE": "FP8-Ti",
        }.get(quant_label, quant_label[:10])

        num_steps = (params.T + tiling.step_batch_size - 1) // tiling.step_batch_size

        print(
            f"{name:<28} {params.batch_size:>3} {params.seq_len:>4} {params.T:>5} {params.N:>4} {params.He:>5} "
            f"{quant_short:<12} {tiling.step_batch_size:>4} {num_steps:>5} "
            f"{mm_times['MM1']:>7.2f} {mm_times['MM2']:>7.2f} {mm_times['MM3']:>7.2f} {mm_times['MM4']:>7.2f} "
            f"{total_us:>8.2f}u {dominant}:{dominant_bound:<6}"
        )

        results.append({
            "name": name,
            "params": params,
            "tiling": tiling,
            "total_us": total_us,
            "pipeline": pipeline,
            "stages": stages,
            "dominant": dominant,
            "dominant_bound": dominant_bound,
        })

    # Summary
    print()
    print(f"{'— Summary —':^{len(cols)}}")
    if results:
        fastest = min(results, key=lambda r: r["total_us"])
        slowest = max(results, key=lambda r: r["total_us"])
        print(f"  Fastest: {fastest['name']} ({fastest['total_us']:.2f} us)")
        print(f"  Slowest: {slowest['name']} ({slowest['total_us']:.2f} us)")

        # Bound distribution
        from collections import Counter
        bound_counts = Counter(r["dominant_bound"] for r in results)
        bound_str = ", ".join(f"{b}: {c}" for b, c in bound_counts.most_common())
        print(f"  Bottleneck distribution: {bound_str}")

        # Quant comparison (decode bs=1 across quant modes)
        decode_cases = [r for r in results if r["params"].batch_size == 1
                        and r["params"].seq_len == 1 and r["params"].N == 128]
        if len(decode_cases) > 1:
            print()
            print("  Decode BS=1 quant comparison:")
            baseline = next((r for r in decode_cases if r["params"].weight_quant_mode == 0), None)
            for r in decode_cases:
                speedup = ""
                if baseline and r is not baseline:
                    speedup = f" ({baseline['total_us'] / r['total_us']:.2f}x vs BF16)"
                print(f"    {r['name']:<28} {r['total_us']:>8.2f} us{speedup}")

    return results
