#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`split-kn` mode: 2D N×K split search with full pipeline evaluation."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...tiling.config import TilingConfig
from ...tiling.split_kn_search import search_split_kn
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Search 2D N×K split configs with full pipeline evaluation."""
    print_header("2D SPLIT (N×K) SEARCH — Full Pipeline Evaluation")

    if not hw.has_cycle_spec:
        print("  Requires per-cycle HW spec. Use --chip 950.")
        return

    results = search_split_kn(params, tiling, hw)
    if not results:
        print("  No valid configurations found.")
        return

    baseline = next((r for r in results if r.label == "baseline"), results[-1])

    # Summary header
    print(f"  Parameters: T={tiling.step_batch_size}, He={params.He}, N={params.N}")
    print(f"  Baseline pipeline: {baseline.pipeline_us:.2f} us "
          f"(MM1={baseline.mm1_cube_us:.2f} MM2={baseline.mm2_cube_us:.2f} "
          f"MM3={baseline.mm3_cube_us:.2f} MM4={baseline.mm4_cube_us:.2f})")
    print()

    # Full results table
    header = (f"  {'Rank':>4} {'Config':<28} | {'Pipeline':>8} {'MM1':>7} {'MM2':>7} "
              f"{'MM3':>7} {'MM4':>7} {'Accum':>7} | {'CubeUtil':>8} {'AIC':>7} {'AIV':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, r in enumerate(results[:15], 1):
        marker = " <--" if i == 1 else ""
        accum = r.mm1_accum_us + r.mm2_accum_us
        print(f"  {i:>4} {r.label:<28} | {r.pipeline_us:>7.2f} {r.mm1_cube_us:>6.2f} "
              f"{r.mm2_cube_us:>6.2f} {r.mm3_cube_us:>6.2f} {r.mm4_cube_us:>6.2f} "
              f"{accum:>6.2f} | {r.cube_utilization:>7.1%} {r.aic_busy_us:>6.2f} "
              f"{r.aiv_busy_us:>6.2f}{marker}")

    best = results[0]
    if best.label != "baseline":
        delta = baseline.pipeline_us - best.pipeline_us
        pct = delta / baseline.pipeline_us * 100
        print(f"\n  Best: {best.label} -> {best.pipeline_us:.2f} us "
              f"(saves {delta:.2f} us / {pct:.1f}% vs baseline)")
    else:
        print(f"\n  Baseline is already optimal at {baseline.pipeline_us:.2f} us")
