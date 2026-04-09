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

    Returns per-AI-core timing for one (k_group_idx, n_group_idx) tile:
      - cube_timing:  one cube core processing M × K_per_core × N_per_core
                      (includes the FP32 partial-output FixPipe pressure when
                      k_groups > 1, since c_size becomes 4 bytes)
      - accum_timing: 2 vector cores reducing k_groups partials for one
                      (M × N_per_core) tile

    Pipeline overlap between accum_timing (on AIV) and subsequent AIC matmul
    stages is NOT applied here — the caller (build_pipeline_dag) places
    AccumCq/AccumCkvKr as separate AIV nodes with dependencies, and the DAG
    forward pass hides accum under whatever AIC stage runs in parallel.

    Per-AI-core scope (1 cube + 2 vectors per AI core, 950 1:2 ratio):
      - Cube call uses active_cores=1 (full per-core HBM/L2 bandwidth)
      - Accum call uses active_cores=2 (2 V cores per AI core)
      - K-cores within one n-group all work on the SAME N slice (N_per_core
        columns), they only split K — they do NOT split N further.
    """
    K_per_core = spec.single_k
    N_per_core = spec.single_n

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

    # Within a K-group, the n_groups cores share the same A slice (loaded once
    # from HBM and reused via L2 by the other n_groups-1 cores). Model A as
    # L2-reused when n_groups > 1, on top of any cross-matmul reuse.
    a_is_l2 = a_reused or spec.n_groups > 1

    # Per-AI-core cube modeling: one (k_group, n_group) cube core processes
    # M × K_per_core × N_per_core. Pass per_core_N directly with active_cores=1
    # so estimate_matmul_detailed sees the right tile and uses full per-core
    # bandwidth (not chip-total / total_cores). The per-core FixPipe pressure
    # is now correct (k_groups× larger than the previous buggy value, because
    # per_core_N is no longer divided by k_groups).
    cube_timing = estimate_matmul_detailed(
        name, M, K_per_core, N_per_core,
        active_cores=1, hw=hw, block=block_kn, a_reused=a_is_l2,
    )

    # Vector accumulation (if k_groups > 1)
    # Per AI core: 2 vector cores reduce k_groups partials for one (M × N_per_core)
    # tile. Each partial is FP32 (4 bytes), the final result is BF16 (out_size).
    # Pipeline overlap with subsequent AIC matmul is handled by build_pipeline_dag,
    # not here — this function only reports the per-AI-core duration.
    if spec.k_groups > 1:
        per_tile_load_bytes = spec.k_groups * M * N_per_core * 4
        per_tile_compute_elements = M * N_per_core * (spec.k_groups - 1)
        per_tile_store_bytes = M * N_per_core * out_size
        # 2 V cores per AI core (950 ratio: 32 AIC : 64 AIV = 1:2)
        accum_timing = estimate_vector_op(
            f"Accum{name.split('_')[0]}",
            load_bytes=per_tile_load_bytes,
            compute_elements=per_tile_compute_elements,
            store_bytes=per_tile_store_bytes,
            hw=hw, active_cores=2,
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

    cube_timing = estimate_matmul_detailed(
        "MM2_CkvKr", M, K_per_core, N, k_cores, hw, block_kc, a_reused)

    # Vector accumulation: load all partial sums, reduce, store final
    accum_timing = estimate_vector_op(
        "AccumCkvKr",
        load_bytes=k_cores * M * N * 4,
        compute_elements=M * N * (k_cores - 1),
        store_bytes=M * N * out_size,
        hw=hw, active_cores=vec_cores,
    )

    return cube_timing, accum_timing
