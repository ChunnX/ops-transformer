#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`prefill-msplit` mode: M-Split + K-outer search for cube-bound prefill.

Searches m_groups × n_groups configurations to eliminate full-core barriers
and achieve cube-bound execution. Reports sync reduction, workspace size,
and comparison vs split-N baseline.
"""

import math

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import estimate_all_stages
from ...pipeline.dag import build_pipeline_dag
from ...tiling.config import TilingConfig
from ...tiling.msplit_search import search_msplit_optimal
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """M-Split prefill analysis: eliminate full-core barriers, target cube-bound."""
    print_header("PREFILL M-SPLIT — Eliminate Full-Core Barriers, Cube-Bound Target")

    if not hw.has_cycle_spec:
        print("  Requires per-cycle HW spec. Use --chip 950.")
        return

    if params.T <= 1:
        print(f"  T={params.T}: prefill-msplit analysis requires T > 1.")
        return

    # Split-N baseline (current kernel behavior)
    baseline_stages = estimate_all_stages(params, tiling, hw)
    baseline_pipeline = build_pipeline_dag(params, tiling, hw, baseline_stages)
    baseline_per_step = baseline_pipeline.total_us
    step_bs = tiling.step_batch_size
    n_steps = math.ceil(params.T / step_bs)
    baseline_total = n_steps * baseline_per_step
    baseline_sync = baseline_pipeline.sync_overhead_us * n_steps

    print(f"  Parameters: T={params.T}, He={params.He}, N={params.N}, "
          f"D={params.D}, Dr={params.Dr}, AIC={hw.aic_num}")
    print(f"  Split-N baseline: {baseline_per_step:.2f} us/step × {n_steps} steps "
          f"= {baseline_total:.2f} us total")
    print(f"    Sync: {baseline_pipeline.sync_overhead_us:.2f} us/step × {n_steps} "
          f"= {baseline_sync:.2f} us total ({baseline_sync/baseline_total*100:.1f}%)")
    print(f"    MM1={baseline_stages['MM1_Cq'].total_us:.2f} "
          f"MM2={baseline_stages['MM2_CkvKr'].total_us:.2f} "
          f"MM3={baseline_stages['MM3_QcQr'].total_us:.2f}({baseline_stages['MM3_QcQr'].bound}) "
          f"MM4={baseline_stages['MM4_Qn'].total_us:.2f}")
    print()

    # Search M-Split configurations
    results = search_msplit_optimal(params, tiling, hw)
    if not results:
        print("  No valid M-Split configurations found.")
        return

    header = (f"  {'Rank':>4} {'Config':<20} | {'Total':>9} {'MM1':>7} {'MM2':>7} "
              f"{'MM3':>7} {'MM4':>7} {'Vec':>6} {'Sync':>6} | "
              f"{'MM3Bnd':<7} {'mLoops':>6} {'WS(MB)':>7} {'vs Base':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, r in enumerate(results[:20], 1):
        marker = " <--" if i == 1 else ""
        delta = baseline_total - r.pipeline_us
        pct = delta / baseline_total * 100 if baseline_total > 0 else 0
        print(f"  {i:>4} {r.label:<20} | {r.pipeline_us:>8.2f} {r.mm1_us:>6.2f} "
              f"{r.mm2_us:>6.2f} {r.mm3_us:>6.2f} {r.mm4_us:>6.2f} "
              f"{r.vec_us:>5.2f} {r.sync_us:>5.2f} | "
              f"{r.mm3_bound:<7} {r.m_loops:>6} {r.workspace_mb:>6.1f} "
              f"{delta:>+7.1f}{marker}")

    best = results[0]
    delta = baseline_total - best.pipeline_us
    pct = delta / baseline_total * 100 if baseline_total > 0 else 0

    print()
    if delta > 0:
        print(f"  Best: {best.label}")
        print(f"    Pipeline: {best.pipeline_us:.2f} us "
              f"(saves {delta:.2f} us / {pct:.1f}% vs split-N {baseline_total:.2f} us)")
        print(f"    Speedup: {baseline_total / best.pipeline_us:.2f}×")
    else:
        print(f"  Split-N baseline ({baseline_total:.2f} us) is still optimal")

    print(f"    Cube util: {best.cube_util:.1%}")
    print(f"    Sync: {best.sync_us:.2f} us ({best.n_sync_events} events) "
          f"vs baseline {baseline_sync:.2f} us "
          f"(reduction: {(1 - best.sync_us/baseline_sync)*100:.0f}%)" if baseline_sync > 0 else "")
    print(f"    Workspace: {best.workspace_mb:.1f} MB")
    print(f"    Config: m_groups={best.m_groups}, n_groups={best.n_groups}, "
          f"M/core={best.M_per_core}, N/core(MM3)={best.N_per_core_mm3}, "
          f"baseM={best.baseM}, m_loops={best.m_loops}")

    # Sync analysis summary
    print()
    print("  Sync Reduction Analysis:")
    print(f"    Current: {n_steps} steps × ~9 events/step = ~{n_steps*9} total sync events")
    print(f"      Includes {n_steps*4} full-AIV WaitAllCore (64-core barriers)")
    print(f"    M-Split: 1 pass × {best.n_sync_events} events = {best.n_sync_events} total")
    if best.n_groups > 1:
        print(f"      LocalBarrier scope: {best.n_groups} cores (was 32 AIC / 64 AIV)")
    else:
        print(f"      Zero barriers (fully independent cores)")
