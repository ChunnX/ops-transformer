#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`split-m` mode: M-axis split search for prefill scenarios."""

import math

from ...model.hardware import AscendHWSpec
from ...model.matmul_split_m import SplitMSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import (
    estimate_all_stages, estimate_all_stages_split_m,
)
from ...model.timing_types import VectorOpTiming
from ...pipeline.dag import build_pipeline_dag
from ...tiling.config import TilingConfig
from ...tiling.split_m_search import search_split_m
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Search M-axis split configs for prefill scenarios."""
    print_header("SPLIT-M SEARCH — M-axis Parallelism for Prefill")

    if not hw.has_cycle_spec:
        print("  Requires per-cycle HW spec. Use --chip 950.")
        return

    if params.T <= 1:
        print(f"  T={params.T}: split-M is not beneficial for decode (T=1).")
        print(f"  Use --mode split-kn for decode optimization.")
        return

    results = search_split_m(params, tiling, hw)
    if not results:
        print("  No valid configurations found.")
        return

    # Split-N baseline: processes stepBatchSize tokens per invocation
    # For total T tokens: ceil(T / stepBatchSize) sequential invocations
    baseline_stages = estimate_all_stages(params, tiling, hw)
    baseline_pipeline = build_pipeline_dag(params, tiling, hw, baseline_stages)
    baseline_per_step = baseline_pipeline.total_us
    step_bs = tiling.step_batch_size
    n_steps = math.ceil(params.T / step_bs)
    baseline_total = n_steps * baseline_per_step
    baseline_vec = sum(s.total_us for s in baseline_stages.values()
                       if isinstance(s, VectorOpTiming))

    print(f"  Parameters: T={params.T}, He={params.He}, N={params.N}, AIC={hw.aic_num}")
    print(f"  Split-N baseline: {baseline_per_step:.2f} us/step × {n_steps} steps "
          f"= {baseline_total:.2f} us total")
    print(f"    Per step ({step_bs} tokens): "
          f"MM1={baseline_stages['MM1_Cq'].total_us:.2f} "
          f"MM2={baseline_stages['MM2_CkvKr'].total_us:.2f} "
          f"MM3={baseline_stages['MM3_QcQr'].total_us:.2f} "
          f"MM4={baseline_stages['MM4_Qn'].total_us:.2f} "
          f"Vec={baseline_vec:.2f}")
    print()

    # Split-M: processes all T tokens in 1 invocation
    header = (f"  {'Rank':>4} {'Config':<8} {'M/core':>7} | {'Total(us)':>10} {'MM1':>8} {'MM2':>8} "
              f"{'MM3':>8} {'MM4':>8} {'Vec':>7} | {'CubeUtil':>8} {'vs SplitN':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, r in enumerate(results[:15], 1):
        marker = " <--" if i == 1 else ""
        delta = baseline_total - r.pipeline_us
        print(f"  {i:>4} {r.label:<8} {r.single_m:>7} | {r.pipeline_us:>9.2f} {r.mm1_cube_us:>7.2f} "
              f"{r.mm2_cube_us:>7.2f} {r.mm3_cube_us:>7.2f} {r.mm4_cube_us:>7.2f} "
              f"{r.vec_total_us:>6.2f} | {r.cube_utilization:>7.1%} {delta:>+8.2f}{marker}")

    best = results[0]
    delta = baseline_total - best.pipeline_us
    pct = delta / baseline_total * 100 if baseline_total > 0 else 0
    if delta > 0:
        print(f"\n  Best: m_groups={best.m_groups} ({best.single_m} tokens/core) "
              f"-> {best.pipeline_us:.2f} us "
              f"(saves {delta:.2f} us / {pct:.1f}% vs split-N total {baseline_total:.2f} us)")
    else:
        print(f"\n  Split-N baseline ({baseline_total:.2f} us for {params.T} tokens) "
              f"is better than all split-M configs")

    # Show per-stage bound for best config
    if results:
        best = results[0]
        spec = SplitMSpec("split_m", best.m_groups, best.single_m)
        stages = estimate_all_stages_split_m(params, tiling, hw, spec)
        print(f"\n  Best split-M stage details (m_groups={best.m_groups}, "
              f"M_per_core={best.single_m}):")
        for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
            s = stages[name]
            print(f"    {name}: {s.total_us:.2f} us, bound={s.bound}, inner={s.inner_bound}")
        vec_names = [n for n in stages if isinstance(stages[n], VectorOpTiming)]
        if vec_names:
            print(f"    Vector (per core, {hw.aiv_num // hw.aic_num} vec cores):")
            for vn in vec_names:
                v = stages[vn]
                if v.total_us > 0.001:
                    print(f"      {vn}: {v.total_us:.3f} us, bound={v.bound}")
