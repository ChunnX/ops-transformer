#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D Split-KN matmul model.

Models matmul where cores are organized in an n_groups × k_groups grid.
N-split cores partition the output columns; K-split cores partition the
reduction axis and produce float32 partial sums that the vector unit
sequentially accumulates.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from .block_spec import MatmulBlockSpec, derive_stepK
from .hardware import AscendHWSpec
from .matmul_detailed import estimate_matmul_detailed
from .timing_types import MatMulTiming, VectorOpTiming
from .vector_ops import estimate_vector_op


@dataclass
class SplitKNSpec:
    """2D core split: n_groups cores on N axis × k_groups cores on K axis."""
    name: str
    n_groups: int
    k_groups: int
    single_n: int        # N per N-group
    single_k: int        # K per K-group

    @property
    def total_cores(self) -> int:
        return self.n_groups * self.k_groups


def estimate_matmul_split_kn(
    name: str, M: int, K: int, N: int,
    spec: SplitKNSpec,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool,
    vec_cores: int,
    out_size: int = 2,
) -> Tuple[MatMulTiming, VectorOpTiming]:
    """Model matmul with 2D split: N-axis × K-axis across cores.

    Returns chip-total wall-clock timing for one (k_group_idx, n_group_idx)
    tile, under the same chip-total / contention convention as
    estimate_matmul_detailed, estimate_matmul_split_m, estimate_matmul_k_outer,
    estimate_matmul_legacy, and estimate_vector_op (5 of 6 timing models in
    this perf module). All timing functions divide chip-total bandwidth across
    the active cores doing the work.

      - cube_timing:  one cube core's work (M × K_per_core × N_per_core), with
                      bandwidth = chip_bw / total_cores (the n_groups × k_groups
                      cores all contend for HBM)
      - accum_timing: vector accumulation of k_groups partials, distributed
                      across vec_cores V cores (chip-total)

    Pipeline overlap between accum_timing (on AIV) and subsequent AIC matmul
    stages is NOT applied here — the caller (build_pipeline_dag) places
    AccumCq/AccumCkvKr as separate AIV nodes with dependencies, and the DAG
    forward pass hides accum under whatever AIC stage runs in parallel.

    K-cores within one n-group all work on the same N slice (N_per_core
    columns); they only split K. Per_core_N = N_per_core (n-split), not
    N_per_core / k_groups (the historical bug from before commit 9d17f206b).

    "Per-AI-core" in the result is the per-AI-core slice of work, not the
    per-AI-core peak bandwidth — bandwidth is divided across the cores
    actually contending for HBM in this configuration.
    """
    K_per_core = spec.single_k
    N_per_core = spec.single_n
    total_cores = spec.total_cores  # n_groups × k_groups

    # Cube: override block for reduced K and N, float output if K-split
    c_size = 4 if spec.k_groups > 1 else out_size
    block_kn = MatmulBlockSpec(
        name, block.baseM, block.baseN, block.baseK, block.stepK,
        mode="split_k",
        dtype_a_size=block.dtype_a_size,
        dtype_b_size=block.dtype_b_size,
        dtype_c_size=c_size,
    )
    # Re-derive stepK for the reduced K
    derived = derive_stepK(block_kn, K_per_core, hw, N_per_core, M=M)
    if derived > 0:
        block_kn.stepK = derived

    # Within a K-group, the n_groups cores share the same A slice (first core
    # cold HBM, rest warm L2). Pass a_shared_cores=n_groups so the detailed
    # model uses effective_load_bw for the A-matrix (harmonic HBM+L2).
    # a_reused is orthogonal: it signals cross-matmul L2 warm (e.g. MM2 tokenX
    # from MM1), in which case A is fully L2 regardless of n_groups.
    #
    # Per-AI-core slice with chip-total contention (5/6 models convention).
    # Each (k_group_idx, n_group_idx) cube core processes M × K_per_core × N_per_core,
    # but all total_cores cores compete for chip-total HBM bandwidth.
    # Pass N = N_per_core × total_cores so estimate_matmul_detailed's
    # per_core_N = ceil(N / C) = N_per_core (correct slice), and bandwidth is
    # divided across total_cores cores (correct contention).
    # FixPipe destination: when k_groups > 1, each cube core writes FP32
    # partial sums to GM for V-core accumulation — bw is HBM-write-contended,
    # not core-local l0c_to_out peak.
    fp_dest = "GM" if spec.k_groups > 1 else "UB"
    cube_timing = estimate_matmul_detailed(
        name, M, K_per_core, N_per_core * total_cores,
        active_cores=total_cores, hw=hw, block=block_kn, a_reused=a_reused,
        a_shared_cores=spec.n_groups,
        fixpipe_dest=fp_dest,
    )

    # Vector accumulation (if k_groups > 1) — chip-total convention
    # Total accum work across all n_groups N-groups: each N-group has k_groups
    # partials of size (M × N_per_core × 4). Aggregate work is
    # k_groups × M × N_full × 4 where N_full = N_per_core × n_groups.
    # All vec_cores V cores share the work in parallel.
    #
    # Note: estimate_vector_op divides both load and bandwidth by C internally,
    # so the bandwidth-bound time is invariant to the choice of C. Only the
    # compute-bound time depends on C; for typical k-accum workloads with small
    # N_per_core, the load (MTE2) dominates so the C choice has minimal impact.
    if spec.k_groups > 1:
        N_full = N_per_core * spec.n_groups
        chip_load_bytes = spec.k_groups * M * N_full * 4
        chip_store_bytes = M * N_full * out_size
        # active_cores = vec_cores: all V cores share the chip-total accum work
        # n_instructions = k_groups - 1: one element-wise Add per partial
        accum_timing = estimate_vector_op(
            f"Accum{name.split('_')[0]}",
            load_bytes=chip_load_bytes,
            compute_elements=M * N_full,
            store_bytes=chip_store_bytes,
            hw=hw, active_cores=vec_cores,
            n_instructions=spec.k_groups - 1,
        )
    else:
        accum_timing = VectorOpTiming(name=f"Accum{name.split('_')[0]}")

    return cube_timing, accum_timing


def estimate_mm2_split_k_cross_core(
    M: int, K: int, N: int, k_cores: int,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool,
    vec_cores: int,
    out_size: int = 2,
) -> Tuple[MatMulTiming, VectorOpTiming]:
    """Model MM2 with K-axis split across cores + vector accumulation.

    Each cube core computes a partial sum on K/k_cores of K for the full N.
    FixPipe outputs float32 (4 bytes) for partial sums.
    Vector side accumulates k_cores partial sums in order, then stores as out_size.
    """
    K_per_core = math.ceil(K / k_cores)

    # Cube: override block with float output and reduced K
    block_kc = MatmulBlockSpec(
        block.name, block.baseM, block.baseN, block.baseK, block.stepK,
        mode="split_k",
        dtype_a_size=block.dtype_a_size,
        dtype_b_size=block.dtype_b_size,
        dtype_c_size=4,  # float32 partial sum output
    )
    # Re-derive stepK for the reduced K
    derived = derive_stepK(block_kc, K_per_core, hw, N, M=M)
    if derived > 0:
        block_kc.stepK = derived

    # k_cores > 1 always (this is the K-split cross-core path): FP32 partials
    # go to GM for V-core accumulation.
    fp_dest = "GM" if k_cores > 1 else "UB"
    cube_timing = estimate_matmul_detailed(
        "MM2_CkvKr", M, K_per_core, N, k_cores, hw, block_kc, a_reused,
        fixpipe_dest=fp_dest)

    # Vector accumulation: load all partial sums, reduce, store final
    accum_timing = estimate_vector_op(
        "AccumCkvKr",
        load_bytes=k_cores * M * N * 4,
        compute_elements=M * N,
        store_bytes=M * N * out_size,
        hw=hw, active_cores=vec_cores,
        n_instructions=k_cores - 1,
    )

    return cube_timing, accum_timing
