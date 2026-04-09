#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`split-kn-fused` mode: estimate fused K-accumulation + RmsNorm savings.

A *supplement* to `--mode split-kn`. For every configuration in the existing
split-kn search space with k_groups > 1, this mode also computes the timing
of the fused K-accum + RmsNorm variant (where the V core does the K
accumulation directly into a UB FP32 buffer and runs RmsNorm on it without a
GM round-trip), and reports the savings vs the baseline (separate Accum
stage with a new full-V-core barrier).

Existing kernel barriers at kernel_mla_prolog_split_n.h:854,859,867 are
unrelated to this optimization and not affected.
"""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...model.split_kn_fused import (
    STATIC_AIV_UB_BYTES, SplitKnFusedBudget, UB_BUDGET_WARN_FRACTION,
    estimate_split_kn_fused,
)
from ...tiling.config import TilingConfig
from ...tiling.split_kn_search import search_split_kn
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode: split-kn fused K-accum + RmsNorm vs baseline accum comparison."""
    print_header("SPLIT-KN-FUSED — Fused K-Accumulation + RmsNorm Analysis")

    if not hw.has_cycle_spec:
        print(f"  split-kn-fused mode requires per-cycle hardware spec (Ascend 950+); "
              f"current chip is {hw.name}.")
        return

    ub_avail = hw.cache.ub_size - STATIC_AIV_UB_BYTES
    print(f"\n  Parameters: T={tiling.step_batch_size}, He={params.He}, "
          f"Hcq={params.Hcq}, Hckv+Dr={params.Hckv + params.Dr}, N={params.N}")
    print(f"  UB budget: {hw.cache.ub_size} bytes total, "
          f"{STATIC_AIV_UB_BYTES} bytes static AIV, "
          f"{ub_avail} bytes available for fused step")
    print(f"  Vector cores: {tiling.vector_block_num}")

    # === Run the existing split-kn search ===
    results = search_split_kn(params, tiling, hw)
    if not results:
        print("\n  No valid split-kn configurations found.")
        return

    # Find configurations with at least one MM having k_groups > 1
    eligible = []
    for r in results:
        mm1_k = r.mm1_spec.k_groups if r.mm1_spec else 1
        mm2_k = r.mm2_spec.k_groups if r.mm2_spec else 1
        if mm1_k > 1 or mm2_k > 1:
            eligible.append(r)

    if not eligible:
        print("\n  No eligible split-kn configurations (all have k_groups == 1).")
        print("  The fused-accum optimization only applies when K is split across cores.")
        return

    print(f"\n  Found {len(eligible)} configurations with k_groups > 1.")

    # === Compute fused timing for each eligible configuration ===
    rows = []
    for r in eligible:
        mm1_fused = None
        mm2_fused = None
        if r.mm1_spec and r.mm1_spec.k_groups > 1:
            mm1_fused = estimate_split_kn_fused(
                params, tiling, hw, r.mm1_spec, "MM1_Cq")
        if r.mm2_spec and r.mm2_spec.k_groups > 1:
            mm2_fused = estimate_split_kn_fused(
                params, tiling, hw, r.mm2_spec, "MM2_CkvKr")

        # Per-config delta = sum of MM1 + MM2 deltas (independent stages)
        delta = 0.0
        if mm1_fused and mm1_fused.applicable:
            delta += mm1_fused.delta_us
        if mm2_fused and mm2_fused.applicable:
            delta += mm2_fused.delta_us

        # The fused pipeline total: subtract delta from the baseline pipeline
        # (the baseline DAG already accounts for AccumCq/AccumCkvKr but does
        # NOT account for the new barrier; the fused mode subtracts both the
        # GM round-trip cost and the barrier).
        fused_pipeline_us = r.pipeline_us - delta

        rows.append((r, mm1_fused, mm2_fused, fused_pipeline_us, delta))

    # Sort by fused pipeline time (best first)
    rows.sort(key=lambda x: x[3])

    # === Print results table ===
    header = (f"  {'Rank':>4} {'Config':<28} | {'Baseline':>9} {'Fused':>9} "
              f"{'Δ':>8} | {'MM1 UB':>10} {'MM2 UB':>10}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (r, mm1_f, mm2_f, fused_us, delta) in enumerate(rows[:15], 1):
        marker = " <--" if i == 1 else ""
        mm1_ub_str = "-"
        mm2_ub_str = "-"
        if mm1_f and mm1_f.applicable:
            b = SplitKnFusedBudget.from_timing(mm1_f, hw)
            warn = "⚠" if b.warning else ""
            mm1_ub_str = f"{b.ub_used_bytes // 1024}KB({b.fraction:.0%}){warn}"
        if mm2_f and mm2_f.applicable:
            b = SplitKnFusedBudget.from_timing(mm2_f, hw)
            warn = "⚠" if b.warning else ""
            mm2_ub_str = f"{b.ub_used_bytes // 1024}KB({b.fraction:.0%}){warn}"

        print(f"  {i:>4} {r.label:<28} | {r.pipeline_us:>8.2f} {fused_us:>8.2f} "
              f"{-delta:>+7.2f} | {mm1_ub_str:>10} {mm2_ub_str:>10}{marker}")

    # === Best summary ===
    best = rows[0]
    r, mm1_f, mm2_f, fused_us, delta = best
    print(f"\n  Best (fused): {r.label} -> {fused_us:.2f} us")
    print(f"    Baseline split-kn pipeline: {r.pipeline_us:.2f} us")
    print(f"    Fused saving:               {delta:.2f} us  ({delta/r.pipeline_us*100:+.1f}%)")

    # Per-MM breakdown for the best config
    print(f"\n  Per-MM breakdown for best config:")
    for fused, label in [(mm1_f, "MM1 (Cq)"), (mm2_f, "MM2 (CkvKr)")]:
        if fused is None or not fused.applicable:
            continue
        budget = SplitKnFusedBudget.from_timing(fused, hw)
        print(f"    {label}  n_groups={fused.n_groups} k_groups={fused.k_groups}")
        print(f"      per_v_m={fused.per_v_m}  chunks={fused.chunks}  chunk_m={fused.chunk_m}")
        print(f"      Fused: accum_load={fused.accum_load_us:.3f}us  "
              f"accum_compute={fused.accum_compute_us:.3f}us  "
              f"rmsnorm={fused.rmsnorm_compute_us:.3f}us  "
              f"store={fused.output_store_us:.3f}us  "
              f"total={fused.total_fused_us:.3f}us  bound={fused.bound}")
        print(f"      Baseline: accum={fused.baseline_accum_us:.3f}us + "
              f"new_barrier={fused.new_barrier_us:.3f}us + "
              f"rmsnorm={fused.baseline_rmsnorm_us:.3f}us = "
              f"{fused.baseline_total_us:.3f}us")
        print(f"      Δ (saved): {fused.delta_us:+.3f}us")
        warn = "  WARNING: > {:.0%} of UB cap".format(UB_BUDGET_WARN_FRACTION) \
            if budget.warning else ""
        print(f"      UB used: {budget.ub_used_bytes} / {budget.ub_avail_bytes} bytes "
              f"({budget.fraction:.1%}){warn}")
        if not budget.fits:
            print(f"      ERROR: budget exceeds available UB — chunking required")
