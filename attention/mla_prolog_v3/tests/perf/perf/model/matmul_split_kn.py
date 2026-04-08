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

    Cores are organized as (n_groups × k_groups) grid.
    - n_groups cores split N: each handles single_n columns
    - k_groups cores split K: each handles single_k of K reduction
    - If k_groups > 1: FixPipe outputs float32, vector accumulates partial sums
    - A matrix: within same K-group, n_groups cores share A slice (L2 reuse)
    """
    K_per_core = spec.single_k
    N_per_core = spec.single_n
    total_cores = spec.total_cores

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

    # All total_cores contend on HBM, but within a K-group the n_groups cores
    # share the same A slice via L2. Model: pass total_cores for contention,
    # a_reused reflects cross-matmul reuse (e.g., MM2 reuses tokenX from MM1).
    # Within a K-group, A is loaded once from HBM and reused via L2 by n_groups-1
    # other cores. We model this by treating A as L2-reused when n_groups > 1.
    a_is_l2 = a_reused or spec.n_groups > 1

    cube_timing = estimate_matmul_detailed(
        name, M, K_per_core, N_per_core * spec.n_groups,
        active_cores=total_cores, hw=hw, block=block_kn, a_reused=a_is_l2,
    )

    # Vector accumulation (if k_groups > 1)
    if spec.k_groups > 1:
        # Per N-group: load k_groups partial sums, reduce, store
        # All n_groups accumulations run in parallel on different vec cores
        per_group_load = spec.k_groups * M * N_per_core * 4
        per_group_compute = M * N_per_core * (spec.k_groups - 1)
        per_group_store = M * N_per_core * out_size
        # vec_cores shared across n_groups
        cores_per_group = max(vec_cores // spec.n_groups, 1)
        accum_timing = estimate_vector_op(
            f"Accum{name.split('_')[0]}",
            load_bytes=per_group_load,
            compute_elements=per_group_compute,
            store_bytes=per_group_store,
            hw=hw, active_cores=cores_per_group,
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
