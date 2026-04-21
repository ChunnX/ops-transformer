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
    a_shared_cores: int = 1,
    fixpipe_dest: str = "UB",
) -> MatMulTiming:
    """Estimate matmul timing using per-cycle inner-loop model.

    Models the L1/L0 cache hierarchy:
    - Outer loop: HBM/L2 -> L1 (MTE1, double-buffered across kL1 iterations)
    - Inner loop: L1 -> L0A/L0B + MMAD (double-buffered across stepK iterations)
    - Output: L0C -> FixPipe

    A-matrix bandwidth is selected by data origin:
    - a_reused=True               : data already in L2 -> l2_bw
    - a_shared_cores > 1          : first core cold HBM, rest warm L2 (harmonic)
                                    -> effective_load_bw(active_cores, reuse_count=a_shared_cores)
    - a_shared_cores == 1 (default): unique A per core -> hbm_bw
    a_reused and a_shared_cores are orthogonal; a_reused takes precedence.

    fixpipe_dest selects the FixPipe output destination bandwidth:
    - "UB"/"L1" (default): core-local l0c_to_out peak
    - "GM": HBM-write chip-total, contended by active_cores (used for
            Split-KN FP32 partial sums written to GM for V-core accumulation)
    """
    C = max(active_cores, 1)
    per_core_N = math.ceil(N / C)
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size

    us = 1e6

    if block.mode == "full_load":
        return _estimate_full_load(
            name, M, K, per_core_N, C, hw, block, a_reused, a_shared_cores,
            fixpipe_dest=fixpipe_dest,
        )

    # --- split_k mode ---
    baseM = block.baseM
    baseN = block.baseN
    baseK = block.baseK
    stepK = block.stepK
    kL1StepSize = baseK * stepK
    kL1Loops = math.ceil(K / kL1StepSize)

    # M-blocking: when M > baseM, iterate M in blocks.
    # Each m-block processes baseM rows independently.
    m_loops = math.ceil(M / baseM) if baseM > 0 else 1

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
    fixpipe_bw = hw.fixpipe_bw(C, dest=fixpipe_dest)
    mmad_tput = hw.mmad_throughput(dtype_a)  # ops/sec per core

    # A-matrix origin -> bandwidth (see docstring)
    if a_reused:
        A_first_bw = l2_bw
        a_source = "L2"
    elif a_shared_cores > 1:
        A_first_bw = hw.effective_load_bw(C, reuse_count=a_shared_cores)
        a_source = "MIX"
    else:
        A_first_bw = hbm_bw
        a_source = "HBM"

    # --- A slab: baseM × kL1StepSize, loaded per N-block ---
    A_slab_bytes = baseM * kL1StepSize * dtype_a
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

    # L0C double-buffer feasibility: L0C always stores FP32 internally
    # regardless of output dtype_c, so tile = baseM × baseN × 4.
    l0c_tile_bytes = baseM * actual_baseN * 4
    l0c_double_buffered = (2 * l0c_tile_bytes <= hw.cache.l0c_size)
    # With double-buffer: fixpipe(tile k) runs in parallel with MMAD(tile k+1).
    # With single-buffer: fixpipe must serialize after the inner MMAD on the
    # same L0C tile; MTE1 still overlaps with (inner + fixpipe).
    inner_plus_fix_s = total_inner_s if l0c_double_buffered else (total_inner_s + fixpipe_time_s)

    # --- Per kL1: N-blocking with A L2 reuse ---
    # MTE1 for A and B are serialized (same engine)
    mte1_first_nb_s = A_first_s + B_load_s
    mte1_rest_nb_s = A_reuse_s + B_load_s

    if l0c_double_buffered:
        per_nb_first_s = max(mte1_first_nb_s, total_inner_s, fixpipe_time_s)
        per_nb_rest_s = max(mte1_rest_nb_s, total_inner_s, fixpipe_time_s)
    else:
        per_nb_first_s = max(mte1_first_nb_s, inner_plus_fix_s)
        per_nb_rest_s = max(mte1_rest_nb_s, inner_plus_fix_s)

    per_kl1_s = per_nb_first_s + max(n_blocks - 1, 0) * per_nb_rest_s
    per_m_block_s = kL1Loops * per_kl1_s
    total_s = m_loops * per_m_block_s

    # For backward-compatible reporting: effective per-kl1 MTE1 time
    mte1_time_s = per_kl1_s  # entire kL1 is now the outer unit

    # Determine inner bound (based on steady-state N-block)
    steady_nb_s = per_nb_rest_s if n_blocks > 1 else per_nb_first_s
    if l0c_double_buffered:
        if steady_nb_s == total_inner_s:
            if inner_iter_s == mmad_time_s:
                bound = "CUBE"
            elif inner_iter_s == l0b_time_s:
                bound = "L0B"
            else:
                bound = "L0A"
        elif steady_nb_s >= mte1_rest_nb_s:
            if a_source in ("L2", "MIX") and A_reuse_s >= B_load_s:
                bound = "L2"
            else:
                bound = "HBM"
        else:
            bound = "FIXPIPE"
    else:
        # Single-buffer: inner and fixpipe are serialized on the same tile.
        if steady_nb_s == inner_plus_fix_s:
            # The dominant sub-component decides the label.
            if fixpipe_time_s >= total_inner_s:
                bound = "FIXPIPE"
            elif inner_iter_s == mmad_time_s:
                bound = "CUBE"
            elif inner_iter_s == l0b_time_s:
                bound = "L0B"
            else:
                bound = "L0A"
        else:
            if a_source in ("L2", "MIX") and A_reuse_s >= B_load_s:
                bound = "L2"
            else:
                bound = "HBM"

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
        cube_us=mmad_time_s * stepK * n_blocks * kL1Loops * m_loops * us,
        # FixPipe triggers once per (m_block, n_block) — L0C accumulates
        # across all kL1 iterations internally. Do NOT multiply by kL1Loops.
        fixpipe_us=fixpipe_time_s * n_blocks * m_loops * us,
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
    a_shared_cores: int = 1,
    fixpipe_dest: str = "UB",
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
    fixpipe_bw = hw.fixpipe_bw(active_cores, dest=fixpipe_dest)
    mmad_tput = hw.mmad_throughput(dtype_a)

    # A-matrix origin -> bandwidth (see estimate_matmul_detailed docstring)
    if a_reused:
        a_first_bw = l2_bw
        a_source = "L2"
    elif a_shared_cores > 1:
        a_first_bw = hw.effective_load_bw(active_cores, reuse_count=a_shared_cores)
        a_source = "MIX"
    else:
        a_first_bw = hbm_bw
        a_source = "HBM"

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

    # L0C double-buffer feasibility (always FP32 internal storage).
    l0c_tile_bytes = M * nSplitSize * 4
    l0c_double_buffered = (2 * l0c_tile_bytes <= hw.cache.l0c_size)
    inner_s = max(l0a_time_s, l0b_time_s, mmad_time_s)
    inner_plus_fix_s = inner_s if l0c_double_buffered else (inner_s + fixpipe_time_s)

    # First N-split: A from HBM/L2
    mte1_first_s = load_A_first_s + load_B_time_s
    if l0c_double_buffered:
        per_split_first_s = max(mte1_first_s, inner_s, fixpipe_time_s)
    else:
        per_split_first_s = max(mte1_first_s, inner_plus_fix_s)
    # Subsequent N-splits: A from L2 (warm)
    mte1_rest_s = load_A_reuse_s + load_B_time_s
    if l0c_double_buffered:
        per_split_rest_s = max(mte1_rest_s, inner_s, fixpipe_time_s)
    else:
        per_split_rest_s = max(mte1_rest_s, inner_plus_fix_s)

    total_s = per_split_first_s + max(nSplits - 1, 0) * per_split_rest_s

    # Determine bound (based on steady-state N-split)
    steady_s = per_split_rest_s if nSplits > 1 else per_split_first_s
    if l0c_double_buffered:
        if steady_s == inner_s:
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
    else:
        if steady_s == inner_plus_fix_s:
            if fixpipe_time_s >= inner_s:
                bound = "FIXPIPE"
            elif mmad_time_s >= l0a_time_s and mmad_time_s >= l0b_time_s:
                bound = "CUBE"
            elif l0a_time_s >= l0b_time_s:
                bound = "L0A"
            else:
                bound = "L0B"
        else:
            bound = "HBM"

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
