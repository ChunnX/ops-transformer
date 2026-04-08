#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`dequant-opt` mode: estimate UB-pipelined dequant+rope savings vs baseline.

Compares the baseline pipeline (3 cross-core sync points + GM round-trips for
DequantQc and RopeQr) against the dequantOpt path (L0C → UB direct, in-UB
RoPE, UB → L1 direct, intra-core HardEvent only).

Available only on chips with the per-cycle hardware spec (Ascend 950+).
"""

from ...model.dequant_opt import (
    compute_budget, estimate_dequant_opt_pipeline,
    is_dequant_opt_applicable,
)
from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import estimate_all_stages
from ...pipeline.dag import build_pipeline_dag, build_pipeline_dag_dequant_opt
from ...tiling.config import TilingConfig
from ..formatting import print_header

# Flag UB usage above this fraction of the cap as a warning.
UB_BUDGET_WARN_FRACTION = 0.70


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode: dequantOpt UB pipeline vs baseline comparison."""
    print_header("DEQUANT-OPT — UB-Pipelined Dequant+RoPE Analysis")

    if not hw.has_cycle_spec:
        print(f"  dequant-opt mode requires per-cycle hardware spec (Ascend 950+); "
              f"current chip is {hw.name}.")
        return

    applicable = is_dequant_opt_applicable(params, tiling)
    if not applicable:
        print(f"  dequantOpt is not applicable for this configuration:")
        print(f"    quant_mode = {params.quant_mode.name} (needs FP8 / INT8 / MXFP8)")
        print(f"    head_num   = {params.N} (needs >= 8)")
        print(f"  Reporting modeled timing anyway for hypothetical comparison.\n")

    # === Build both pipelines using a single shared stage_timings dict ===
    stage_timings = estimate_all_stages(params, tiling, hw)
    baseline = build_pipeline_dag(params, tiling, hw, stage_timings)

    fused = estimate_dequant_opt_pipeline(params, tiling, hw)
    optimized = build_pipeline_dag_dequant_opt(
        params, tiling, hw, stage_timings, dequant_opt_timing=fused)

    # === Compute the per-stage delta around the MM3→MM4 segment ===
    sync_us = hw.sync_overhead_us

    def base_us(name: str) -> float:
        s = stage_timings.get(name)
        return s.total_us if s is not None else 0.0

    base_mm3 = base_us("MM3_QcQr")
    base_dequant = base_us("DequantQc")
    base_rope_qr = base_us("RopeQr")
    base_mm4 = base_us("MM4_Qn")
    base_segment = base_mm3 + sync_us * 2 + base_dequant + base_rope_qr + sync_us * 2 + base_mm4
    opt_segment = base_mm3 + fused.total_us + base_mm4

    print(f"\nMM3 → DequantRope → MM4 segment comparison")
    print(f"{'Stage':<22}{'Baseline (us)':>16}{'dequantOpt (us)':>20}{'Δ':>14}")
    print("  " + "-" * 70)
    print(f"  {'MM3_QcQr':<20}{base_mm3:>16.3f}{base_mm3:>20.3f}{0.0:>14.3f}")
    print(f"  {'sync→AIV':<20}{sync_us:>16.3f}{0.0:>20.3f}{-sync_us:>14.3f}")
    print(f"  {'DequantQc':<20}{base_dequant:>16.3f}{0.0:>20.3f}{-base_dequant:>14.3f}")
    print(f"  {'RopeQr':<20}{base_rope_qr:>16.3f}{0.0:>20.3f}{-base_rope_qr:>14.3f}")
    print(f"  {'sync→AIC':<20}{sync_us:>16.3f}{0.0:>20.3f}{-sync_us:>14.3f}")
    print(f"  {'DequantRopeUB':<20}{0.0:>16.3f}{fused.total_us:>20.3f}{fused.total_us:>14.3f}")
    print(f"  {'MM4_Qn':<20}{base_mm4:>16.3f}{base_mm4:>20.3f}{0.0:>14.3f}")
    print("  " + "-" * 70)
    seg_delta = opt_segment - base_segment
    seg_pct = (seg_delta / base_segment * 100.0) if base_segment > 0 else 0.0
    print(f"  {'Segment total':<20}{base_segment:>16.3f}{opt_segment:>20.3f}"
          f"{seg_delta:>14.3f}  ({seg_pct:+.1f}%)")

    # === Whole-pipeline comparison ===
    pipe_delta = optimized.total_us - baseline.total_us
    pipe_pct = (pipe_delta / baseline.total_us * 100.0) if baseline.total_us > 0 else 0.0
    print(f"\nWhole pipeline (DAG critical path)")
    print(f"  Baseline       : {baseline.total_us:.3f} us")
    print(f"  dequantOpt     : {optimized.total_us:.3f} us")
    print(f"  Δ              : {pipe_delta:+.3f} us  ({pipe_pct:+.1f}%)")

    # === DequantRopeUB stage breakdown ===
    print(f"\nDequantRopeUB stage breakdown (per AI core, per stepBatchSize={tiling.step_batch_size})")
    print(f"  L0C → UB         : {fused.l0c_to_ub_us:.3f} us")
    print(f"  UB compute       : {fused.ub_compute_us:.3f} us")
    print(f"  UB → L1          : {fused.ub_to_l1_us:.3f} us")
    print(f"  intra-core sync  : {fused.intra_core_sync_us:.3f} us  (assumed 0 per H6)")
    print(f"  total            : {fused.total_us:.3f} us")
    print(f"  bound            : {fused.bound}")
    print(f"  per-core N       : {fused.per_core_n}  (active cores: {fused.active_cores})")

    # === UB / L1 budget ===
    budget = compute_budget(fused, hw)
    print(f"\nResource budget (per AI core)")
    ub_warn = "  WARNING: > {:.0%} of UB cap".format(UB_BUDGET_WARN_FRACTION) \
        if budget.ub_fraction > UB_BUDGET_WARN_FRACTION else ""
    print(f"  UB used          : {budget.ub_used_bytes:>7d} / {budget.ub_total_bytes} bytes  "
          f"({budget.ub_fraction:.1%}){ub_warn}")
    print(f"  L1 used          : {budget.l1_used_bytes:>7d} / {budget.l1_total_bytes} bytes  "
          f"({budget.l1_fraction:.1%})")
    if not budget.fits:
        print(f"  ERROR: budget exceeds cap — UB-pipeline not feasible for this configuration")
