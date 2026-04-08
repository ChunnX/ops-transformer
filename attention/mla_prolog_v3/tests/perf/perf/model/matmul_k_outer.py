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
            name, M, K, N, active_cores, hw, block, a_reused)
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

    # --- K-outer loop model: B pinned in L1 ---
    hbm_bw = hw.per_core_bw("hbm", C)
    l2_bw = hw.per_core_bw("l2", C)
    l0a_bw = hw.per_core_bw("l1_to_l0a", C)
    l0b_bw = hw.per_core_bw("l1_to_l0b", C)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", C)
    mmad_tput = hw.mmad_throughput(dtype_a)

    a_source = "L2" if a_reused else "HBM"
    a_bw = l2_bw if a_reused else hbm_bw

    # B loaded ONCE per kL1 slab from HBM (or L2 if already cached)
    load_B_once_s = B_slab / hbm_bw if hbm_bw > 0 else 0

    # Per M-block within kL1:
    load_A_mb_bytes = baseM * kL1StepSize * dtype_a
    load_A_mb_s = load_A_mb_bytes / a_bw if a_bw > 0 else 0

    # Inner loop (same as standard model)
    l0a_bytes = baseM * baseK * dtype_a
    l0b_bytes = baseK * per_core_N * dtype_b
    l0a_time_s = l0a_bytes / l0a_bw if l0a_bw > 0 else 0
    l0b_time_s = l0b_bytes / l0b_bw if l0b_bw > 0 else 0
    mmad_ops = baseM * per_core_N * baseK
    mmad_time_s = mmad_ops / mmad_tput if mmad_tput > 0 else 0
    inner_iter_s = max(l0a_time_s, l0b_time_s, mmad_time_s)
    total_inner_s = stepK * inner_iter_s

    fixpipe_bytes = baseM * per_core_N * dtype_c
    fixpipe_time_s = fixpipe_bytes / fixpipe_bw if fixpipe_bw > 0 else 0

    per_m_block_s = max(load_A_mb_s, total_inner_s, fixpipe_time_s)

    # Per kL1 slab: B load (once) overlaps with M-block processing
    m_blocks_total_s = m_loops * per_m_block_s
    per_kl1_s = max(load_B_once_s, m_blocks_total_s)

    total_s = kL1_loops * per_kl1_s

    # Determine bound
    if per_m_block_s == total_inner_s:
        if inner_iter_s == mmad_time_s:
            mb_bound = "CUBE"
        elif inner_iter_s == l0b_time_s:
            mb_bound = "L0B"
        else:
            mb_bound = "L0A"
    elif per_m_block_s == load_A_mb_s:
        mb_bound = "HBM_A" if not a_reused else "L2_A"
    else:
        mb_bound = "FIXPIPE"

    if per_kl1_s == load_B_once_s and load_B_once_s > m_blocks_total_s:
        bound = "HBM_B"
        inner_bound = "MTE1_B"
    else:
        bound = mb_bound
        inner_bound = mb_bound

    return MatMulTiming(
        name=name, M=M, K=K, N=N,
        active_cores=C,
        load_A_hbm_us=(load_A_mb_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(load_A_mb_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=load_B_once_s * us,
        cube_us=mmad_time_s * stepK * kL1_loops * m_loops * us,
        fixpipe_us=fixpipe_time_s * kL1_loops * m_loops * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=max(load_A_mb_s, load_B_once_s / max(m_loops, 1)) * us,
        kL1_loops=kL1_loops,
        inner_bound=inner_bound,
    )
