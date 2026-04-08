#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-cycle inner-loop matmul model (Ascend 950 detailed path).

Models the L1/L0 cache hierarchy:
- Outer loop: HBM/L2 -> L1 (MTE1, double-buffered across kL1 iterations)
- Inner loop: L1 -> L0A/L0B + MMAD (double-buffered across stepK iterations)
- Output: L0C -> FixPipe

Two modes are supported:
- "split_k" (default): partition K into kL1 slabs streaming through L1
- "full_load": A loaded once, only N is iterated (used for the small MM4 matmul)
"""

import math

from .block_spec import MatmulBlockSpec
from .hardware import AscendHWSpec
from .timing_types import MatMulTiming


def estimate_matmul_detailed(
    name: str,
    M: int, K: int, N: int,
    active_cores: int,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool = False,
) -> MatMulTiming:
    """Estimate matmul timing using per-cycle inner-loop model.

    Models the L1/L0 cache hierarchy:
    - Outer loop: HBM/L2 -> L1 (MTE1, double-buffered across kL1 iterations)
    - Inner loop: L1 -> L0A/L0B + MMAD (double-buffered across stepK iterations)
    - Output: L0C -> FixPipe
    """
    C = max(active_cores, 1)
    per_core_N = math.ceil(N / C)
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size

    us = 1e6

    if block.mode == "full_load":
        return _estimate_full_load(name, M, K, per_core_N, C, hw, block, a_reused)

    # --- split_k mode ---
    baseM = block.baseM
    baseK = block.baseK
    stepK = block.stepK
    kL1StepSize = baseK * stepK
    kL1Loops = math.ceil(K / kL1StepSize)

    # Bandwidth values (bytes/sec)
    hbm_bw = hw.per_core_bw("hbm", C)
    l2_bw = hw.per_core_bw("l2", C)
    l0a_bw = hw.per_core_bw("l1_to_l0a", C)
    l0b_bw = hw.per_core_bw("l1_to_l0b", C)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", C)
    mmad_tput = hw.mmad_throughput(dtype_a)  # ops/sec per core

    # --- Outer loop: GM/L2 -> L1 per kL1 iteration ---
    load_A_l1_bytes = baseM * kL1StepSize * dtype_a
    load_B_l1_bytes = kL1StepSize * per_core_N * dtype_b
    if a_reused:
        a_source = "L2"
        load_A_time_s = load_A_l1_bytes / l2_bw if l2_bw > 0 else 0
    else:
        a_source = "HBM"
        load_A_time_s = load_A_l1_bytes / hbm_bw if hbm_bw > 0 else 0
    load_B_time_s = load_B_l1_bytes / hbm_bw if hbm_bw > 0 else 0
    mte1_time_s = max(load_A_time_s, load_B_time_s)

    # --- Inner loop: L1 -> L0 + MMAD per stepK iteration ---
    l0a_bytes = baseM * baseK * dtype_a
    l0b_bytes = baseK * per_core_N * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    # MMAD: baseM * per_core_N * baseK MACs = baseM * per_core_N * baseK * 2 FLOPs
    mmad_ops = baseM * per_core_N * baseK
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    # Double-buffered: L0A/L0B loads overlap with MMAD
    inner_iter_s = max(l0a_time_s, l0b_time_s, mmad_time_s)
    total_inner_s = stepK * inner_iter_s

    # --- FixPipe: L0C -> output per kL1 iteration ---
    fixpipe_bytes = baseM * per_core_N * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    # Pipelined: MTE1, inner compute, and FixPipe overlap across kL1 iterations
    per_kl1_s = max(mte1_time_s, total_inner_s, fixpipe_time_s)
    total_s = kL1Loops * per_kl1_s

    # Determine inner bound
    candidates = {
        "MTE1_A": load_A_time_s,
        "MTE1_B": load_B_time_s,
        "L0A": l0a_time_s * stepK,
        "L0B": l0b_time_s * stepK,
        "MMAD": mmad_time_s * stepK,
        "FIXPIPE": fixpipe_time_s,
    }
    inner_bound = max(candidates, key=candidates.get)
    # Simplified bound name
    if per_kl1_s == mte1_time_s:
        # MTE1-bound: A could be coming from L2 (cross-matmul reuse) or HBM.
        # When A is in L2 and dominates the load, label the bound as L2.
        if a_source == "L2" and load_A_time_s >= load_B_time_s:
            bound = "L2"
        else:
            bound = "HBM"
    elif per_kl1_s == total_inner_s:
        if inner_iter_s == mmad_time_s:
            bound = "CUBE"
        elif inner_iter_s == l0a_time_s:
            bound = "L0A"
        else:
            bound = "L0B"
    else:
        bound = "FIXPIPE"

    return MatMulTiming(
        name=name, M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=(load_A_l1_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(load_A_l1_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=load_B_time_s * us,
        cube_us=mmad_time_s * stepK * kL1Loops * us,
        fixpipe_us=fixpipe_time_s * kL1Loops * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=mte1_time_s * us,
        kL1_loops=kL1Loops,
        inner_bound=inner_bound,
    )


def _estimate_full_load(
    name: str, M: int, K: int, per_core_N: int,
    active_cores: int, hw: AscendHWSpec, block: MatmulBlockSpec,
    a_reused: bool,
) -> MatMulTiming:
    """Inner-loop model for full_load mode (MM4): A fully loaded, N-split only."""
    us = 1e6
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size
    nSplitSize = block.baseN

    hbm_bw = hw.per_core_bw("hbm", active_cores)
    l2_bw = hw.per_core_bw("l2", active_cores)
    l0a_bw = hw.per_core_bw("l1_to_l0a", active_cores)
    l0b_bw = hw.per_core_bw("l1_to_l0b", active_cores)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", active_cores)
    mmad_tput = hw.mmad_throughput(dtype_a)

    a_source = "L2" if a_reused else "HBM"
    a_bw = l2_bw if a_reused else hbm_bw

    # Load A once
    load_A_bytes = M * K * dtype_a
    load_A_time_s = load_A_bytes / a_bw if a_bw > 0 else 0

    nSplits = math.ceil(per_core_N / nSplitSize) if nSplitSize > 0 else 1

    # Per N-split iteration
    load_B_bytes = K * nSplitSize * dtype_b
    load_B_time_s = load_B_bytes / hbm_bw if hbm_bw > 0 else 0
    l0a_bytes = M * K * dtype_a
    l0b_bytes = K * nSplitSize * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    mmad_ops = M * nSplitSize * K
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    fixpipe_bytes = M * nSplitSize * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    per_split_s = max(load_B_time_s, max(l0a_time_s, l0b_time_s, mmad_time_s), fixpipe_time_s)
    total_s = load_A_time_s + nSplits * per_split_s

    # Determine bound
    if per_split_s == load_B_time_s:
        bound = "HBM"
    elif max(l0a_time_s, l0b_time_s, mmad_time_s) >= max(load_B_time_s, fixpipe_time_s):
        if mmad_time_s >= l0a_time_s and mmad_time_s >= l0b_time_s:
            bound = "CUBE"
        elif l0a_time_s >= l0b_time_s:
            bound = "L0A"
        else:
            bound = "L0B"
    else:
        bound = "FIXPIPE"

    return MatMulTiming(
        name=name, M=M, K=K, N=per_core_N * max(active_cores, 1),
        active_cores=active_cores,
        load_A_hbm_us=(load_A_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(load_A_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=load_B_time_s * us,
        cube_us=mmad_time_s * nSplits * us,
        fixpipe_us=fixpipe_time_s * nSplits * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=load_A_time_s * us,
        kL1_loops=nSplits,
        inner_bound=bound,
    )
