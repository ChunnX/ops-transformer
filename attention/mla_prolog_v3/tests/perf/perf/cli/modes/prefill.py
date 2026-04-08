#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`prefill` mode: K-outer loop search for cube-bound prefill."""

import math

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import estimate_all_stages
from ...pipeline.dag import build_pipeline_dag
from ...tiling.config import TilingConfig
from ...tiling.prefill_search import search_prefill_optimal
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """K-outer loop analysis for prefill: B pinned in L1 across M-blocks."""
    print_header("PREFILL K-OUTER — B Pinned in L1, Cube-Bound Target")

    if not hw.has_cycle_spec:
        print("  Requires per-cycle HW spec. Use --chip 950.")
        return

    if params.T <= 1:
        print(f"  T={params.T}: prefill analysis requires T > 1.")
        return

    results = search_prefill_optimal(params, tiling, hw)
    if not results:
        print("  No valid configurations found.")
        return

    # Multi-step split-N baseline
    baseline_stages = estimate_all_stages(params, tiling, hw)
    baseline_pipeline = build_pipeline_dag(params, tiling, hw, baseline_stages)
    baseline_per_step = baseline_pipeline.total_us
    step_bs = tiling.step_batch_size
    n_steps = math.ceil(params.T / step_bs)
    baseline_total = n_steps * baseline_per_step

    print(f"  Parameters: T={params.T}, He={params.He}, N={params.N}, AIC={hw.aic_num}")
    print(f"  Split-N baseline: {baseline_per_step:.2f} us/step × {n_steps} steps "
          f"= {baseline_total:.2f} us total")
    print(f"    MM1={baseline_stages['MM1_Cq'].total_us:.2f} "
          f"MM2={baseline_stages['MM2_CkvKr'].total_us:.2f} "
          f"MM3={baseline_stages['MM3_QcQr'].total_us:.2f}({baseline_stages['MM3_QcQr'].bound}) "
          f"MM4={baseline_stages['MM4_Qn'].total_us:.2f}")
    print()

    header = (f"  {'Rank':>4} {'Config':<18} | {'Total(us)':>10} {'MM1':>8} {'MM2':>8} "
              f"{'MM3':>8} {'MM4':>8} {'Vec':>7} | {'MM3 Bound':<10} {'vs Base':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, r in enumerate(results[:15], 1):
        marker = " <--" if i == 1 else ""
        delta = baseline_total - r.pipeline_us
        print(f"  {i:>4} {r.label:<18} | {r.pipeline_us:>9.2f} {r.mm1_cube_us:>7.2f} "
              f"{r.mm2_cube_us:>7.2f} {r.mm3_cube_us:>7.2f} {r.mm4_cube_us:>7.2f} "
              f"{r.vec_total_us:>6.2f} | {r.mm3_bound:<10} {delta:>+7.2f}{marker}")

    best = results[0]
    delta = baseline_total - best.pipeline_us
    pct = delta / baseline_total * 100 if baseline_total > 0 else 0
    if delta > 0:
        print(f"\n  Best: {best.label} -> {best.pipeline_us:.2f} us "
              f"(saves {delta:.2f} us / {pct:.1f}% vs split-N {baseline_total:.2f} us)")
    else:
        print(f"\n  Split-N baseline ({baseline_total:.2f} us) is still optimal")

    # Cube utilization for best
    if best.cube_utilization > 0:
        print(f"  Cube utilization: {best.cube_utilization:.1%} "
              f"(AIC={best.aic_busy_us:.2f} AIV={best.aiv_busy_us:.2f})")
