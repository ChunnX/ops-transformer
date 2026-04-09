#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-axis split matmul model (prefill optimization).

Each core computes [M_per_core, K] x [K, N_full]. N is tiled into N-blocks
within each core (3-level loop): m_loops × kL1_loops × n_blocks × per_nb.

The B (weight) matrix is shared across all cores via L2 cache reuse.
The A (activation) matrix has unique rows per core, loaded from HBM.
"""

import math
from dataclasses import dataclass

from .block_spec import MatmulBlockSpec, derive_stepK
from .hardware import AscendHWSpec
from .timing_types import MatMulTiming


@dataclass
class SplitMSpec:
    """M-axis split across cores. Each core handles full N."""
    name: str
    m_groups: int         # cores splitting M
    single_m: int         # ceil(M_total / m_groups)

    @property
    def total_cores(self) -> int:
        return self.m_groups


def estimate_matmul_split_m(
    name: str,
    M_total: int, K: int, N: int,
    spec: SplitMSpec,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool = False,
) -> MatMulTiming:
    """Model matmul with M-axis split across cores (prefill mode).

    Each core computes [M_per_core, K] x [K, N_full].
    N is tiled into N-blocks within each core (3-level loop):
      m_loops x kL1_loops x n_blocks x per_nb

    B (weight) matrix is shared across all cores via L2 cache reuse.
    A (activation) matrix rows are unique per core, loaded from HBM.
    """
    us = 1e6
    m_groups = spec.m_groups
    M_per_core = spec.single_m
    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size

    baseM = block.baseM
    baseN = block.baseN
    baseK = block.baseK

    # Re-derive stepK for split-M: L1 holds one N-block's worth of B
    derived = derive_stepK(block, K, hw, n_per_core=baseN, M=M_per_core)
    stepK = derived if derived > 0 else block.stepK
    kL1StepSize = baseK * stepK
    kL1_loops = math.ceil(K / kL1StepSize) if kL1StepSize > 0 else 1

    m_loops = math.ceil(M_per_core / baseM) if baseM > 0 else 1
    n_blocks = math.ceil(N / baseN) if baseN > 0 else 1

    # Bandwidth (all m_groups cores contend on HBM)
    hbm_bw = hw.per_core_bw("hbm", m_groups)
    l0a_bw = hw.per_core_bw("l1_to_l0a", m_groups)
    l0b_bw = hw.per_core_bw("l1_to_l0b", m_groups)
    fixpipe_bw = hw.per_core_bw("l0c_to_out", m_groups)
    mmad_tput = hw.mmad_throughput(dtype_a)

    # B matrix: shared across cores -> L2 reuse
    # effective_load_bw models 1/m cold HBM + (m-1)/m warm L2
    b_bw = hw.effective_load_bw(m_groups, reuse_count=m_groups)

    # A matrix: each core loads unique rows from HBM
    a_bw = hbm_bw
    if a_reused:
        a_bw = hw.per_core_bw("l2", m_groups)

    # --- Per N-block (innermost, double-buffered) ---
    load_B_nb_bytes = kL1StepSize * baseN * dtype_b
    load_B_nb_s = load_B_nb_bytes / b_bw if b_bw > 0 else 0

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

    # --- Per kL1 slab: A loaded per N-block with L2 reuse ---
    load_A_kl1_bytes = baseM * kL1StepSize * dtype_a
    # First N-block: A from HBM (or L2 if a_reused)
    load_A_first_s = load_A_kl1_bytes / a_bw if a_bw > 0 else 0
    # Subsequent N-blocks: A from L2 (warm from first load)
    l2_bw_val = hw.per_core_bw("l2", m_groups)
    load_A_reuse_s = load_A_kl1_bytes / l2_bw_val if l2_bw_val > 0 else 0

    # MTE1: A and B serialized
    mte1_first_nb = load_A_first_s + load_B_nb_s
    per_nb_first_s = max(mte1_first_nb, total_inner_s, fixpipe_time_s)
    mte1_rest_nb = load_A_reuse_s + load_B_nb_s
    per_nb_rest_s = max(mte1_rest_nb, total_inner_s, fixpipe_time_s)

    per_kl1_s = per_nb_first_s + max(n_blocks - 1, 0) * per_nb_rest_s

    # --- Per M-block ---
    per_m_block_s = kL1_loops * per_kl1_s

    # --- Total ---
    total_s = m_loops * per_m_block_s

    # Determine bound (based on steady-state N-block)
    steady_nb_s = per_nb_rest_s if n_blocks > 1 else per_nb_first_s
    if steady_nb_s == total_inner_s:
        if inner_iter_s == mmad_time_s:
            nb_bound = "CUBE"
        elif inner_iter_s == l0b_time_s:
            nb_bound = "L0B"
        else:
            nb_bound = "L0A"
    elif steady_nb_s >= mte1_rest_nb:
        nb_bound = "L2" if b_bw > hbm_bw * 0.5 else "HBM"
    else:
        nb_bound = "FIXPIPE"

    bound = nb_bound
    inner_bound = nb_bound

    a_source = "L2" if a_reused else "HBM"

    return MatMulTiming(
        name=name, M=M_total, K=K, N=N,
        active_cores=m_groups,
        load_A_hbm_us=(load_A_kl1_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(load_A_kl1_bytes / l2_bw_val * us) if l2_bw_val > 0 else 0,
        load_B_hbm_us=load_B_nb_s * us,
        cube_us=mmad_time_s * stepK * kL1_loops * n_blocks * m_loops * us,
        fixpipe_us=fixpipe_time_s * kL1_loops * n_blocks * m_loops * us,
        total_us=total_s * us,
        bound=bound,
        a_source=a_source,
        l0a_us=l0a_time_s * us,
        l0b_us=l0b_time_s * us,
        mmad_us=mmad_time_s * us,
        mte1_outer_us=load_A_first_s + load_B_nb_s * us,
        kL1_loops=kL1_loops,
        inner_bound=inner_bound,
    )
