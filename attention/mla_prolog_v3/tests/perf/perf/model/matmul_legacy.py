#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chip-total matmul timing model (legacy 910B path).

Uses chip-total bandwidth and TFLOPS aggregated over all cores. Less detailed
than the per-cycle model in matmul_detailed.py but works without per-cycle
hardware specs.
"""

import math

from .hardware import AscendHWSpec
from .quant import QuantMode, is_mxfp8
from .timing_types import MatMulTiming


def estimate_matmul(
    name: str,
    M: int, K: int, N: int,
    active_cores: int,
    hw: AscendHWSpec,
    dtype_a_size: int = 2,
    dtype_b_size: int = 2,
    dtype_c_size: int = 2,
    a_reused: bool = False,
) -> MatMulTiming:
    """Estimate matmul timing with HBM/L2 bandwidth separation.

    In split-N mode, all cores load the same A matrix but different B slices.
    A is shared (L2 reusable), B is unique per core (always HBM).
    """
    C = max(active_cores, 1)
    per_core_N = math.ceil(N / C)

    # A matrix load
    load_A_bytes = M * K * dtype_a_size
    load_A_hbm_time_s = load_A_bytes / hw.hbm_bw_bytes_per_sec
    # All C cores compete for bandwidth; the bottleneck bus limits all cores
    load_A_l2_time_s = load_A_bytes / (hw.l2_bw_bytes_per_sec / max(C, 1))

    if a_reused:
        # A already in L2 from prior matmul
        a_source = "L2"
        load_A_effective_s = load_A_l2_time_s
    else:
        a_source = "HBM"
        # First core pays HBM cost; but all cores contend on whichever bus saturates first
        load_A_effective_s = load_A_bytes / min(hw.hbm_bw_bytes_per_sec, hw.l2_bw_bytes_per_sec / C)

    # B matrix load (unique per core, all from HBM)
    load_B_bytes_per_core = K * per_core_N * dtype_b_size
    load_B_time_s = load_B_bytes_per_core / (hw.hbm_bw_bytes_per_sec / C)

    # MTE1 time (double-buffered overlap of A and B)
    mte1_time_s = max(load_A_effective_s, load_B_time_s)

    # Compute
    total_flops = 2.0 * M * per_core_N * K
    if dtype_a_size == 1:
        cube_power = hw.cube_flops_fp8 if is_mxfp8(QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT) else hw.cube_ops_int8
    else:
        cube_power = hw.cube_flops_bf16
    cube_time_s = total_flops / (cube_power / hw.aic_num)

    # Store via FixPipe
    store_C_bytes = M * per_core_N * dtype_c_size
    fixpipe_time_s = store_C_bytes / (hw.fixpipe_bw_bytes_per_sec / C)

    # Total with double-buffering (overlap MTE1, Cube, FixPipe)
    total_s = max(mte1_time_s, cube_time_s, fixpipe_time_s)

    # Determine bound
    if mte1_time_s >= cube_time_s and mte1_time_s >= fixpipe_time_s:
        if load_A_effective_s >= load_B_time_s:
            bound = "HBM" if a_source == "HBM" else "L2"
        else:
            bound = "HBM"
    elif cube_time_s >= fixpipe_time_s:
        bound = "CUBE"
    else:
        bound = "FIXPIPE"

    us = 1e6
    return MatMulTiming(
        name=name,
        M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=load_A_hbm_time_s * us,
        load_A_l2_us=load_A_l2_time_s * us,
        load_B_hbm_us=load_B_time_s * us,
        cube_us=cube_time_s * us,
        fixpipe_us=fixpipe_time_s * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
    )
