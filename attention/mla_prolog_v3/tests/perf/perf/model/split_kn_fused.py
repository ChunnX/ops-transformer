#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split-KN fused K-accumulation + RmsNorm model.

A *supplement* to the existing split-kn analysis (perf/model/matmul_split_kn.py).
When split-kn is enabled with k_groups > 1, the K-axis partials produced by the
cube cores must be reduced before downstream stages can use them. The naive
implementation is a separate AIV stage that reads all k_groups partials from
GM, sums them, writes the reduced result back to GM, then a separate RmsNorm
stage reads it back from GM.

This module models the *fused* alternative: each V core, in its existing
M-row range, reads its M-slice from each k_groups partial, sums them in a UB
FP32 accumulator, then immediately runs RmsNorm on the accumulator (still in
UB) and writes the final BF16 output to GM. This avoids:

  1. A new full-V-core barrier that the separate accumulation step would need
     (so all k_groups cubes finish before any V core starts reading partials)
  2. The UB → GM → UB round-trip between the accumulation output and the
     RmsNorm input

Existing kernel barriers at kernel_mla_prolog_split_n.h:854,859,867 are
unrelated to this optimization and stay in place.

This module is purely additive — it does not modify any existing perf model
function. The bit-identical regression invariant is preserved.
"""

import math
from dataclasses import dataclass

from .hardware import AscendHWSpec
from .matmul_split_kn import SplitKNSpec, estimate_matmul_split_kn
from .operator_params import OperatorParams
from .vector_ops import estimate_vector_op


# Static AIV UB occupancy (gamma + dequant scales + sin/cos + share buffer).
# Drawn from the Phase 1 exploration of kernel_mla_prolog_split_n.h:510-578.
# Conservative ~34 KB; the budget for the fused step is `ub_size - STATIC`.
STATIC_AIV_UB_BYTES = 34 * 1024

# Threshold above which the perf model raises a budget warning.
UB_BUDGET_WARN_FRACTION = 0.70


@dataclass
class SplitKnFusedTiming:
    """Timing for the fused K-accum + RmsNorm step (per V core, all chunks)."""
    mm_name: str               # "MM1_Cq" or "MM2_CkvKr"
    n_groups: int
    k_groups: int
    per_v_m: int               # rows per V core (existing M-split partitioning)
    chunks: int                # > 1 if M had to be sub-batched for UB fit
    chunk_m: int               # M rows per chunk

    # Fused stage breakdown (summed across all chunks)
    accum_load_us: float       # k_groups * MTE2 reads, one partial per iteration
    accum_compute_us: float    # k_groups * Add ops in UB
    rmsnorm_compute_us: float  # PIPE_V on accumulator, in UB
    output_store_us: float     # MTE3 to final GM destination
    total_fused_us: float      # max() of components per chunk × chunks

    # Baseline (separate accum + rmsnorm) for comparison
    baseline_accum_us: float
    baseline_rmsnorm_us: float
    new_barrier_us: float      # the would-be new barrier (= sync_overhead_us)
    baseline_total_us: float   # accum + barrier + rmsnorm

    delta_us: float            # baseline_total - total_fused (positive = win)
    bound: str                 # "MTE2" | "VECTOR" | "MTE3"

    # Resource accounting
    ub_bytes: int              # peak per-V-core UB usage in fused step
    applicable: bool           # False when k_groups <= 1


@dataclass
class SplitKnFusedBudget:
    """UB budget summary for one configuration."""
    ub_used_bytes: int
    ub_total_bytes: int
    ub_avail_bytes: int        # cap minus static AIV usage
    fraction: float            # used / avail
    fits: bool
    warning: bool              # True if > UB_BUDGET_WARN_FRACTION

    @classmethod
    def from_timing(cls, timing: SplitKnFusedTiming, hw: AscendHWSpec):
        avail = hw.cache.ub_size - STATIC_AIV_UB_BYTES
        used = timing.ub_bytes
        frac = used / avail if avail > 0 else 0.0
        return cls(
            ub_used_bytes=used,
            ub_total_bytes=hw.cache.ub_size,
            ub_avail_bytes=avail,
            fraction=frac,
            fits=used <= avail,
            warning=frac > UB_BUDGET_WARN_FRACTION,
        )


def _ub_bytes_per_chunk(chunk_m: int, n_full: int, out_size: int) -> int:
    """UB bytes for one chunk's working set.

    Layout (FP32 partials, FP32 accumulator, FP32 RmsNorm work, BF16 output):
      - 2 × chunk_m × n_full × 4   (FP32 partial input, double-buffered)
      - 1 × chunk_m × n_full × 4   (FP32 accumulator)
      - 1 × chunk_m × n_full × 4   (FP32 RmsNorm working buffer)
      - 2 × chunk_m × n_full × out_size  (BF16 output, double-buffered)
    """
    return chunk_m * n_full * (4 * 2 + 4 + 4 + out_size * 2)


def _compute_chunk_layout(per_v_m: int, n_full: int, out_size: int,
                          ub_avail: int) -> tuple:
    """Return (chunk_m, chunks) such that one chunk fits in 70% of ub_avail."""
    target = int(ub_avail * UB_BUDGET_WARN_FRACTION)
    per_row_bytes = n_full * (4 * 2 + 4 + 4 + out_size * 2)
    if per_row_bytes <= 0:
        return per_v_m, 1
    chunk_m_max = max(1, target // per_row_bytes)
    chunk_m = min(per_v_m, chunk_m_max)
    chunks = math.ceil(per_v_m / chunk_m) if chunk_m > 0 else 1
    return chunk_m, chunks


def estimate_split_kn_fused(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    spec: SplitKNSpec,
    mm_name: str,
) -> SplitKnFusedTiming:
    """Compare fused-accum-rmsnorm vs baseline accum+rmsnorm for one split-kn config.

    Only meaningful when spec.k_groups > 1; for k_groups == 1 the baseline has
    no AccumCq stage and the fused step is identical to the existing
    RmsNorm — returns SplitKnFusedTiming(applicable=False, delta_us=0).
    """
    if spec.k_groups <= 1:
        return SplitKnFusedTiming(
            mm_name=mm_name,
            n_groups=spec.n_groups,
            k_groups=spec.k_groups,
            per_v_m=0, chunks=0, chunk_m=0,
            accum_load_us=0.0, accum_compute_us=0.0,
            rmsnorm_compute_us=0.0, output_store_us=0.0,
            total_fused_us=0.0,
            baseline_accum_us=0.0, baseline_rmsnorm_us=0.0,
            new_barrier_us=0.0, baseline_total_us=0.0,
            delta_us=0.0, bound="N/A",
            ub_bytes=0, applicable=False,
        )

    # === Configuration ===
    M = tiling.step_batch_size
    if mm_name == "MM1_Cq":
        N_full = params.Hcq
    elif mm_name == "MM2_CkvKr":
        N_full = params.Hckv + params.Dr
    else:
        raise ValueError(f"unknown mm_name {mm_name!r}")
    out_size = 2  # BF16 final output to GM

    # Existing M-split partitioning: per_v_m = ceil(M / vector_block_num)
    aiv_total = max(tiling.vector_block_num, 1)
    per_v_m = max(1, math.ceil(M / aiv_total))

    # Chunking decision based on UB budget
    ub_avail = hw.cache.ub_size - STATIC_AIV_UB_BYTES
    chunk_m, chunks = _compute_chunk_layout(per_v_m, N_full, out_size, ub_avail)

    # === Fused step modeling (per V core, summed across chunks) ===
    # Per chunk byte counts
    partial_load_bytes_per_kg = chunk_m * N_full * 4   # FP32 partial
    accum_compute_elements = chunk_m * N_full          # one Add per element per kg
    rmsnorm_compute_elements = chunk_m * N_full * 3    # mean + scale + multiply
    output_store_bytes = chunk_m * N_full * out_size

    # Active cores for the fused step: same vector cores doing existing
    # M-split RmsNorm. Use min(M, aiv_total) to reflect underutilization
    # when M < aiv_total (e.g., decode bs1).
    active_cores = max(min(M, aiv_total), 1)

    # MTE2: k_groups partial reads. Bandwidth shared across active V cores.
    # Use chip-total HBM divided by active_cores (mirrors estimate_vector_op).
    mte2_bw_per_core = hw.mte2_bw_bytes_per_sec / active_cores
    accum_load_us_per_chunk = (
        spec.k_groups * partial_load_bytes_per_kg / mte2_bw_per_core
        if mte2_bw_per_core > 0 else 0.0
    ) * 1e6

    # PIPE_V: k_groups Add ops, each chunk_m * N_full elements
    vec_per_core = hw.vector_ops_per_sec_bf16 / hw.aiv_num
    accum_compute_us_per_chunk = (
        spec.k_groups * accum_compute_elements / vec_per_core
        if vec_per_core > 0 else 0.0
    ) * 1e6

    # PIPE_V: RmsNorm on accumulator (already in UB, no MTE2/MTE3)
    # Reuse estimate_vector_op compute portion semantics: ~3 ops/element.
    rmsnorm_compute_us_per_chunk = (
        rmsnorm_compute_elements / vec_per_core if vec_per_core > 0 else 0.0
    ) * 1e6

    # MTE3: final BF16 output to GM
    mte3_bw_per_core = hw.mte3_bw_bytes_per_sec / active_cores
    output_store_us_per_chunk = (
        output_store_bytes / mte3_bw_per_core if mte3_bw_per_core > 0 else 0.0
    ) * 1e6

    # Total per chunk: sum of components (no internal pingpong overlap).
    # Conservative — does not assume MTE2/PIPE_V overlap across iterations.
    total_per_chunk = (accum_load_us_per_chunk + accum_compute_us_per_chunk
                       + rmsnorm_compute_us_per_chunk + output_store_us_per_chunk)
    total_fused_us = total_per_chunk * chunks

    # Bound classification (which component dominates a single chunk)
    candidates = {
        "MTE2": accum_load_us_per_chunk,
        "VECTOR": accum_compute_us_per_chunk + rmsnorm_compute_us_per_chunk,
        "MTE3": output_store_us_per_chunk,
    }
    bound = max(candidates, key=candidates.get)

    # === Baseline modeling: separate accum + rmsnorm + new barrier ===
    # Use estimate_matmul_split_kn's existing accum timing
    from .block_spec import MatmulBlockSpec
    from .orchestration import get_default_block_specs
    block_specs = get_default_block_specs(params, hw)
    block = block_specs[mm_name]

    if mm_name == "MM1_Cq":
        K_full = params.He
    else:
        K_full = params.He
    _, accum_timing = estimate_matmul_split_kn(
        mm_name, M, K_full, N_full, spec,
        hw=hw, block=block, a_reused=False,
        vec_cores=tiling.vector_block_num, out_size=out_size,
    )
    baseline_accum_us = accum_timing.total_us

    # Baseline RmsNorm: standard estimate_vector_op with full M × N_full GM read
    rmsnorm_load = M * N_full * out_size
    rmsnorm_compute = M * N_full * 3
    rmsnorm_store = M * N_full * out_size
    rmsnorm_timing = estimate_vector_op(
        f"RmsNorm{'Cq' if mm_name == 'MM1_Cq' else 'CkvKr'}_baseline",
        load_bytes=rmsnorm_load,
        compute_elements=rmsnorm_compute,
        store_bytes=rmsnorm_store,
        hw=hw,
        active_cores=tiling.vector_block_num,
    )
    baseline_rmsnorm_us = rmsnorm_timing.total_us

    new_barrier_us = hw.sync_overhead_us
    baseline_total_us = baseline_accum_us + new_barrier_us + baseline_rmsnorm_us
    delta_us = baseline_total_us - total_fused_us

    return SplitKnFusedTiming(
        mm_name=mm_name,
        n_groups=spec.n_groups,
        k_groups=spec.k_groups,
        per_v_m=per_v_m,
        chunks=chunks,
        chunk_m=chunk_m,
        accum_load_us=accum_load_us_per_chunk * chunks,
        accum_compute_us=accum_compute_us_per_chunk * chunks,
        rmsnorm_compute_us=rmsnorm_compute_us_per_chunk * chunks,
        output_store_us=output_store_us_per_chunk * chunks,
        total_fused_us=total_fused_us,
        baseline_accum_us=baseline_accum_us,
        baseline_rmsnorm_us=baseline_rmsnorm_us,
        new_barrier_us=new_barrier_us,
        baseline_total_us=baseline_total_us,
        delta_us=delta_us,
        bound=bound,
        ub_bytes=_ub_bytes_per_chunk(chunk_m, N_full, out_size),
        applicable=True,
    )
