#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-outer loop matmul model (cube-bound prefill optimization).

Reorders the matmul loop to put K outermost so that B is loaded once per kL1
slab and pinned in L1 across all M-blocks. Amortizes B-loading over m_loops
M-blocks, enabling cube-bound operation when baseM is large enough for MMAD
to dominate L0B loading.

Falls back to estimate_matmul_detailed × m_loops when the B slab is too
large to pin in L1 alongside the streaming A buffer.
"""

import math

from .block_spec import MatmulBlockSpec, derive_stepK
from .hardware import AscendHWSpec
from .matmul_detailed import estimate_matmul_detailed
from .timing_types import MatMulTiming


def estimate_matmul_k_outer(
    name: str,
    M: int, K: int, N: int,
    active_cores: int,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool = False,
    fixpipe_dest: str = "UB",
) -> MatMulTiming:
    """Estimate matmul with K-outer loop order for prefill.

    Loop order: kL1_loops × m_loops × stepK × inner
    B is loaded ONCE per kL1 slab and pinned in L1 across M-blocks.
    A is loaded per M-block (streaming from HBM or L2).

    This amortizes B loading over m_loops M-blocks, enabling cube-bound
    operation when baseM is large enough for MMAD to dominate L0B loading.

    Falls back to standard estimate_matmul_detailed when B slab doesn't
    fit in L1 alongside A.
    """
    C = max(active_cores, 1)
    per_core_N = math.ceil(N / C)
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size
    us = 1e6

    baseM = block.baseM
    baseK = block.baseK

    # Derive stepK for per_core_N (same as split-N)
    derived = derive_stepK(block, K, hw, n_per_core=per_core_N, M=M)
    stepK = derived if derived > 0 else block.stepK
    kL1StepSize = baseK * stepK
    kL1_loops = math.ceil(K / kL1StepSize) if kL1StepSize > 0 else 1

    m_loops = math.ceil(M / baseM) if baseM > 0 else 1

    # Check if B slab fits in L1 alongside A (B pinned + A double-buffered)
    B_slab = kL1StepSize * per_core_N * dtype_b
    A_slab = baseM * kL1StepSize * dtype_a
    l1_needed = B_slab + A_slab * 2  # B pinned (1×) + A streaming (2× double buf)

    if l1_needed > hw.cache.l1_size:
        # B slab too large to pin — fall back to standard model with M-loop
        # (standard model but multiply by m_loops)
        single_m_timing = estimate_matmul_detailed(
            name, M, K, N, active_cores, hw, block, a_reused,
            fixpipe_dest=fixpipe_dest)
        # The standard model only covers baseM rows; multiply by m_loops
        return MatMulTiming(
            name=name, M=M, K=K, N=N,
            active_cores=C,
            load_A_hbm_us=single_m_timing.load_A_hbm_us,
            load_A_l2_us=single_m_timing.load_A_l2_us,
            load_B_hbm_us=single_m_timing.load_B_hbm_us,
            cube_us=single_m_timing.cube_us * m_loops,
            fixpipe_us=single_m_timing.fixpipe_us * m_loops,
            total_us=single_m_timing.total_us * m_loops,
            bound=single_m_timing.bound,
            a_source=single_m_timing.a_source,
            l0a_us=single_m_timing.l0a_us,
            l0b_us=single_m_timing.l0b_us,
            mmad_us=single_m_timing.mmad_us,
            mte1_outer_us=single_m_timing.mte1_outer_us,
            kL1_loops=kL1_loops,
            inner_bound=single_m_timing.inner_bound + " (fallback)",
        )

    # --- K-outer loop model: B pinned in L1, with N-blocking ---
    hbm_bw = hw.per_core_bw("hbm", C)
    l2_bw = hw.per_core_bw("l2", C)
    l0a_bw = hw.per_core_bw("l1_to_l0a", C)
    l0b_bw = hw.per_core_bw("l1_to_l0b", C)
    fixpipe_bw = hw.fixpipe_bw(C, dest=fixpipe_dest)
    mmad_tput = hw.mmad_throughput(dtype_a)

    a_source = "L2" if a_reused else "HBM"
    baseN = block.baseN

    # N-blocking within each core
    n_blocks = math.ceil(per_core_N / baseN) if baseN > 0 else 1

    # Per-tile sizes (use baseN, not per_core_N)
    l0a_bytes = baseM * baseK * dtype_a
    l0b_bytes = baseK * baseN * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    mmad_ops = baseM * baseN * baseK
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    inner_iter_s = max(l0a_time_s, l0b_time_s, mmad_time_s)
    total_inner_s = stepK * inner_iter_s

    fixpipe_bytes = baseM * baseN * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    # L0C double-buffer feasibility (always FP32 internal storage).
    l0c_tile_bytes = baseM * baseN * 4
    l0c_double_buffered = (2 * l0c_tile_bytes <= hw.cache.l0c_size)
    inner_plus_fix_s = total_inner_s if l0c_double_buffered else (total_inner_s + fixpipe_time_s)

    # A and B slab sizes for L1
    A_slab_bytes = baseM * kL1StepSize * dtype_a
    B_slab_bytes = kL1StepSize * baseN * dtype_b

    # K-outer: per (kL1, nb, mb)
    # B: mb=0 from HBM, mb>=1 from L2 (pinned across m_blocks)
    # A: nb=0 from HBM (or L2 if a_reused), nb>=1 from L2 (repeated load)
    per_kl1_s = 0.0
    for nb in range(n_blocks):
        A_load_s = A_slab_bytes / l2_bw if (nb > 0 or a_reused) else A_slab_bytes / hbm_bw
        if l2_bw <= 0 or hbm_bw <= 0:
            A_load_s = 0.0
        B_first_s = B_slab_bytes / hbm_bw if hbm_bw > 0 else 0
        B_rest_s = B_slab_bytes / l2_bw if l2_bw > 0 else 0

        mte1_first_mb = B_first_s + A_load_s
        mte1_rest_mb = B_rest_s + A_load_s
        if l0c_double_buffered:
            per_mb_first = max(mte1_first_mb, total_inner_s, fixpipe_time_s)
            per_mb_rest = max(mte1_rest_mb, total_inner_s, fixpipe_time_s)
        else:
            per_mb_first = max(mte1_first_mb, inner_plus_fix_s)
            per_mb_rest = max(mte1_rest_mb, inner_plus_fix_s)

        per_nb_s = per_mb_first + max(m_loops - 1, 0) * per_mb_rest
        per_kl1_s += per_nb_s

    total_s = kL1_loops * per_kl1_s

    # Determine bound (based on steady-state: last nb, last mb)
    A_steady_s = A_slab_bytes / l2_bw if l2_bw > 0 else 0
    B_steady_s = B_slab_bytes / l2_bw if l2_bw > 0 else 0
    mte1_steady = B_steady_s + A_steady_s
    if l0c_double_buffered:
        per_mb_steady = max(mte1_steady, total_inner_s, fixpipe_time_s)
        if per_mb_steady == total_inner_s:
            if inner_iter_s == mmad_time_s:
                bound = "CUBE"
            elif inner_iter_s == l0b_time_s:
                bound = "L0B"
            else:
                bound = "L0A"
        elif per_mb_steady >= mte1_steady:
            bound = "HBM" if not a_reused else "L2"
        else:
            bound = "FIXPIPE"
    else:
        per_mb_steady = max(mte1_steady, inner_plus_fix_s)
        if per_mb_steady == inner_plus_fix_s:
            if fixpipe_time_s >= total_inner_s:
                bound = "FIXPIPE"
            elif inner_iter_s == mmad_time_s:
                bound = "CUBE"
            elif inner_iter_s == l0b_time_s:
                bound = "L0B"
            else:
                bound = "L0A"
        else:
            bound = "HBM" if not a_reused else "L2"

    inner_bound = bound

    return MatMulTiming(
        name=name, M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=(A_slab_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(A_slab_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=(B_slab_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        cube_us=mmad_time_s * stepK * n_blocks * kL1_loops * m_loops * us,
        # FixPipe fires once per (m_block, n_block); L0C accumulates across kL1.
        fixpipe_us=fixpipe_time_s * n_blocks * m_loops * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=per_kl1_s / max(n_blocks, 1) * us,
        kL1_loops=kL1_loops,
        inner_bound=inner_bound,
    )
