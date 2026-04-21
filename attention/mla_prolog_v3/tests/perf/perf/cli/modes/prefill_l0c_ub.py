#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`prefill-l0c-ub` mode: L0C→UB core-internal pipeline for prefill."""

import math

from ...model.hardware import AscendHWSpec
from ...model.matmul_split_m import SplitMSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import (
    compute_ub_constrained_baseM,
    estimate_all_stages,
    estimate_all_stages_l0c_ub,
)
from ...model.timing_types import VectorOpTiming
from ...pipeline.dag import build_pipeline_dag, build_pipeline_dag_l0c_ub
from ...tiling.config import TilingConfig
from ...tiling.l0c_ub_search import search_l0c_ub
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Search L0C→UB core-internal pipeline configs for prefill."""
    print_header("L0C→UB CORE-INTERNAL PIPELINE — Prefill Optimization")

    if not hw.has_cycle_spec:
        print("  Requires per-cycle HW spec. Use --chip 950.")
        return

    if params.T <= 1:
        print(f"  T={params.T}: L0C→UB split-M is not beneficial for decode (T=1).")
        return

    baseM = compute_ub_constrained_baseM(params.Hcq, hw.cache.ub_size)
    ub_used = baseM * params.Hcq * 2 + 17 * 1024
    ub_pct = ub_used / hw.cache.ub_size * 100

    print(f"  Parameters: T={params.T}, He={params.He}, Hcq={params.Hcq}, N={params.N}, AIC={hw.aic_num}")
    print(f"  UB constraint: baseM={baseM} (Hcq={params.Hcq}, "
          f"UB={hw.cache.ub_size // 1024}KB, used={ub_used // 1024}KB/{ub_pct:.0f}%)")
    print()

    results = search_l0c_ub(params, tiling, hw)
    if not results:
        print("  No valid configurations found.")
        return

    # Split-N baseline
    baseline_stages = estimate_all_stages(params, tiling, hw)
    baseline_pipeline = build_pipeline_dag(params, tiling, hw, baseline_stages)
    baseline_per_step = baseline_pipeline.total_us
    step_bs = tiling.step_batch_size
    n_steps = math.ceil(params.T / step_bs)
    baseline_total = n_steps * baseline_per_step

    print(f"  Split-N baseline: {baseline_per_step:.2f} us/step x {n_steps} steps "
          f"= {baseline_total:.2f} us total")
    print()

    header = (f"  {'Rank':>4} {'Config':<16} {'M/core':>7} {'m_loops':>7} | "
              f"{'Total(us)':>10} {'MM1':>8} {'MM2':>8} {'MM3':>8} {'MM4':>8} "
              f"{'Vec':>7} {'UB→L1':>7} | {'CubeUtil':>8} {'vs SplitN':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, r in enumerate(results[:10], 1):
        marker = " <--" if i == 1 else ""
        delta = baseline_total - r.pipeline_us
        print(f"  {i:>4} {r.label:<16} {r.single_m:>7} {r.m_loops:>7} | "
              f"{r.pipeline_us:>9.2f} {r.mm1_cube_us:>7.2f} {r.mm2_cube_us:>7.2f} "
              f"{r.mm3_cube_us:>7.2f} {r.mm4_cube_us:>7.2f} "
              f"{r.vec_total_us:>6.2f} {r.ub_l1_total_us:>6.2f} | "
              f"{r.cube_utilization:>7.1%} {delta:>+8.2f}{marker}")

    best = results[0]
    delta = baseline_total - best.pipeline_us
    pct = delta / baseline_total * 100 if baseline_total > 0 else 0

    print()
    if delta > 0:
        print(f"  Best: m_groups={best.m_groups}, baseM={best.baseM} "
              f"({best.single_m} tokens/core, {best.m_loops} m_loops)")
        print(f"  -> {best.pipeline_us:.2f} us "
              f"(saves {delta:.2f} us / {pct:.1f}% vs split-N {baseline_total:.2f} us)")
    else:
        print(f"  Split-N baseline ({baseline_total:.2f} us) is better than all L0C→UB configs")

    # Show per-stage details for best config
    spec = SplitMSpec("l0c_ub", best.m_groups, best.single_m)
    from ...tiling.l0c_ub_search import _get_l0c_ub_block_specs
    block_specs = _get_l0c_ub_block_specs(params, hw, best.baseM)
    stages = estimate_all_stages_l0c_ub(params, tiling, hw, spec,
                                        block_specs=block_specs)

    print(f"\n  Best L0C→UB stage details (m_groups={best.m_groups}, "
          f"M_per_core={best.single_m}, baseM={best.baseM}):")
    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        s = stages[name]
        a_src = getattr(s, 'a_source', '')
        print(f"    {name}: {s.total_us:.2f} us, bound={s.bound}, a_source={a_src}")
    for name in ["UB_to_L1_Cq", "UB_to_L1_Qc"]:
        if name in stages:
            print(f"    {name}: {stages[name].total_us:.3f} us")
    vec_names = [n for n in stages if isinstance(stages[n], VectorOpTiming)
                 and not n.startswith("UB_to_L1")]
    if vec_names:
        print(f"    Vector (per core, {hw.aiv_num // hw.aic_num} vec cores, compute-only):")
        for vn in vec_names:
            v = stages[vn]
            if v.total_us > 0.001:
                mte2_note = " [HBM]" if v.mte2_load_us > 0 else ""
                print(f"      {vn}: {v.total_us:.3f} us, bound={v.bound}{mte2_note}")
