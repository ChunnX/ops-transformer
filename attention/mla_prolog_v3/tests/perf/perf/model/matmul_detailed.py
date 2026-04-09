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
    baseN = block.baseN
    baseK = block.baseK
    stepK = block.stepK
    kL1StepSize = baseK * stepK
    kL1Loops = math.ceil(K / kL1StepSize)

    # N-blocking: when per_core_N > baseN, iterate N in blocks within each core.
    # Each N-block requires A to be present in L1 alongside B.
    # A is loaded n_blocks times per kL1: first from HBM, rest from L2.
    n_blocks = math.ceil(per_core_N / baseN) if baseN > 0 else 1
    # Actual N per block (may be smaller than baseN for last block or small per_core_N)
    actual_baseN = min(baseN, per_core_N)

    # Bandwidth values (bytes/sec)
    hbm_bw = hw.per_core_bw("hbm", C)
    l2_bw = hw.per_core_bw("l2", C)
    l0a_bw = hw.per_core_bw("l1_to_l0a", C)
    l0b_bw = hw.per_core_bw("l1_to_l0b", C)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", C)
    mmad_tput = hw.mmad_throughput(dtype_a)  # ops/sec per core

    a_source = "L2" if a_reused else "HBM"

    # --- A slab: baseM × kL1StepSize, loaded per N-block ---
    A_slab_bytes = baseM * kL1StepSize * dtype_a
    # First N-block: A from HBM (or L2 if cross-matmul reused)
    A_first_bw = l2_bw if a_reused else hbm_bw
    A_first_s = A_slab_bytes / A_first_bw if A_first_bw > 0 else 0
    # Subsequent N-blocks: A already in L2 (warm)
    A_reuse_s = A_slab_bytes / l2_bw if l2_bw > 0 else 0

    # --- B slab: kL1StepSize × actual_baseN, different N offset per block ---
    B_slab_bytes = kL1StepSize * actual_baseN * dtype_b
    B_load_s = B_slab_bytes / hbm_bw if hbm_bw > 0 else 0

    # --- Inner loop: L1 -> L0 + MMAD per stepK iteration ---
    l0a_bytes = baseM * baseK * dtype_a
    l0b_bytes = baseK * actual_baseN * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    mmad_ops = baseM * actual_baseN * baseK
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    # Double-buffered: L0A/L0B loads overlap with MMAD
    inner_iter_s = max(l0a_time_s, l0b_time_s, mmad_time_s)
    total_inner_s = stepK * inner_iter_s

    # --- FixPipe: L0C -> output per N-block ---
    fixpipe_bytes = baseM * actual_baseN * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    # --- Per kL1: N-blocking with A L2 reuse ---
    # MTE1 for A and B are serialized (same engine)
    mte1_first_nb_s = max(A_first_s, B_load_s)
    mte1_rest_nb_s = max(A_reuse_s, B_load_s)

    per_nb_first_s = max(mte1_first_nb_s, total_inner_s, fixpipe_time_s)
    per_nb_rest_s = max(mte1_rest_nb_s, total_inner_s, fixpipe_time_s)

    per_kl1_s = per_nb_first_s + max(n_blocks - 1, 0) * per_nb_rest_s
    total_s = kL1Loops * per_kl1_s

    # For backward-compatible reporting: effective per-kl1 MTE1 time
    mte1_time_s = per_kl1_s  # entire kL1 is now the outer unit

    # Determine inner bound (based on steady-state N-block)
    steady_nb_s = per_nb_rest_s if n_blocks > 1 else per_nb_first_s
    if steady_nb_s == total_inner_s:
        if inner_iter_s == mmad_time_s:
            bound = "CUBE"
        elif inner_iter_s == l0b_time_s:
            bound = "L0B"
        else:
            bound = "L0A"
    elif steady_nb_s >= mte1_rest_nb_s:
        if a_source == "L2" and A_reuse_s >= B_load_s:
            bound = "L2"
        else:
            bound = "HBM"
    else:
        bound = "FIXPIPE"

    # Detailed inner bound
    candidates = {
        "MTE1_A": A_first_s,
        "MTE1_B": B_load_s * n_blocks,
        "L0A": l0a_time_s * stepK * n_blocks,
        "L0B": l0b_time_s * stepK * n_blocks,
        "MMAD": mmad_time_s * stepK * n_blocks,
        "FIXPIPE": fixpipe_time_s * n_blocks,
    }
    inner_bound = max(candidates, key=candidates.get)

    return MatMulTiming(
        name=name, M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=(A_slab_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(A_slab_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=B_load_s * us,
        cube_us=mmad_time_s * stepK * n_blocks * kL1Loops * us,
        fixpipe_us=fixpipe_time_s * n_blocks * kL1Loops * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=per_kl1_s * us,
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
    a_first_bw = l2_bw if a_reused else hbm_bw

    # A: loaded per N-split (must be in L1 alongside B).
    # First split from HBM (or L2 if reused), subsequent from L2.
    load_A_bytes = M * K * dtype_a
    load_A_first_s = load_A_bytes / a_first_bw if a_first_bw > 0 else 0
    load_A_reuse_s = load_A_bytes / l2_bw if l2_bw > 0 else 0

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

    # First N-split: A from HBM/L2
    mte1_first_s = max(load_A_first_s, load_B_time_s)
    per_split_first_s = max(mte1_first_s, max(l0a_time_s, l0b_time_s, mmad_time_s), fixpipe_time_s)
    # Subsequent N-splits: A from L2 (warm)
    mte1_rest_s = max(load_A_reuse_s, load_B_time_s)
    per_split_rest_s = max(mte1_rest_s, max(l0a_time_s, l0b_time_s, mmad_time_s), fixpipe_time_s)

    total_s = per_split_first_s + max(nSplits - 1, 0) * per_split_rest_s

    # Determine bound (based on steady-state N-split)
    steady_s = per_split_rest_s if nSplits > 1 else per_split_first_s
    if steady_s == max(l0a_time_s, l0b_time_s, mmad_time_s):
        if mmad_time_s >= l0a_time_s and mmad_time_s >= l0b_time_s:
            bound = "CUBE"
        elif l0a_time_s >= l0b_time_s:
            bound = "L0A"
        else:
            bound = "L0B"
    elif steady_s >= mte1_rest_s:
        bound = "HBM"
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
        mte1_outer_us=load_A_first_s * us,
        kL1_loops=nSplits,
        inner_bound=bound,
    )
