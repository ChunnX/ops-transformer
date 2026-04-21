#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-Split + K-outer matmul model for prefill (cube-bound target).

Combines M-Split (cut tokens across m_groups, each core gets full or partial N)
with K-outer loop ordering (B pinned in L1, reused across M-blocks via L2).

Key modeling details:
  - 2D core grid: m_groups × n_groups = total_cores
  - Each core: M_per_core tokens, N_per_core = N / n_groups columns
  - Within each core: K-outer × N-blocking × M-blocking
  - B first load from HBM (cold), subsequent m_blocks from L2 (warm, ~3.25× HBM)
  - Cube-bound when baseM ≥ 64 (BF16) as MMAD dominates L0B loading
  - Workspace: baseM × N_per_core × dtype_c per core (for MM3 QcQr intermediate)
"""

import math
from dataclasses import dataclass
from typing import Optional

from .block_spec import MatmulBlockSpec, derive_stepK
from .hardware import AscendHWSpec
from .timing_types import MatMulTiming


@dataclass
class MSplitSpec:
    """M-Split + optional N-Split configuration."""
    name: str
    m_groups: int       # cores splitting M axis
    n_groups: int       # cores splitting N axis within each m-group
    M_per_core: int     # tokens per core (= ceil(T / m_groups))
    N_per_core: int     # columns per core (= ceil(N / n_groups))

    @property
    def total_cores(self) -> int:
        return self.m_groups * self.n_groups

    @property
    def workspace_mm3_bytes(self) -> int:
        """Workspace for MM3 QcQr intermediate per m-group (baseM × full_N × dtype_c)."""
        # Actual per m-group workspace = M_per_core × N_per_core × n_groups × dtype_c
        # But n_groups cores write different N slices into a contiguous row
        return self.M_per_core * self.N_per_core * self.n_groups * 4  # worst-case fp32


def estimate_matmul_msplit(
    name: str,
    M_total: int, K: int, N: int,
    spec: MSplitSpec,
    hw: AscendHWSpec,
    block: MatmulBlockSpec,
    a_reused: bool = False,
    fixpipe_dest: str = "UB",
) -> MatMulTiming:
    """Estimate matmul with M-Split + K-outer + N-blocking.

    Loop order (K-outer): kL1_loops × n_blocks × m_loops × stepK × inner
    B loaded once per (kL1, n_block), reused across m_blocks via L2.
    A loaded per (kL1, m_block) from HBM (unique rows per core).

    Args:
        M_total: Total tokens (T).
        K: Reduction dimension.
        N: Output columns (total, before core split).
        spec: MSplitSpec with m_groups, n_groups, M_per_core, N_per_core.
        block: MatmulBlockSpec (baseM should be ≥64 for cube-bound).
        a_reused: True if A matrix hits L2 from a prior matmul.
    """
    us = 1e6
    total_cores = spec.total_cores
    M_per_core = spec.M_per_core
    N_per_core = spec.N_per_core

    dtype_a = block.dtype_a_size
    dtype_b = block.dtype_b_size
    dtype_c = block.dtype_c_size

    baseM = block.baseM
    baseN = block.baseN
    baseK = block.baseK

    # Derive stepK for N_per_core (fits in L1 alongside A)
    derived = derive_stepK(block, K, hw, n_per_core=baseN, M=baseM)
    stepK = derived if derived > 0 else block.stepK
    kL1StepSize = baseK * stepK
    kL1_loops = math.ceil(K / kL1StepSize) if kL1StepSize > 0 else 1

    m_loops = math.ceil(M_per_core / baseM) if baseM > 0 else 1
    n_blocks = math.ceil(N_per_core / baseN) if baseN > 0 else 1

    # Bandwidth: all total_cores contend on chip-total HBM
    hbm_bw = hw.per_core_bw("hbm", total_cores)
    l2_bw = hw.per_core_bw("l2", total_cores)
    l0a_bw = hw.per_core_bw("l1_to_l0a", total_cores)
    l0b_bw = hw.per_core_bw("l1_to_l0b", total_cores)
    fixpipe_bw = hw.fixpipe_bw(total_cores, dest=fixpipe_dest)
    mmad_tput = hw.mmad_throughput(dtype_a)

    a_source = "L2" if a_reused else "HBM"
    a_bw = l2_bw if a_reused else hbm_bw

    # --- Inner loop: L1 → L0 + MMAD (per stepK iteration) ---
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

    # --- Per (kL1, n_block, m_block): full 3D loop ---
    # B slab: kL1StepSize × baseN × dtype_b
    B_slab_bytes = kL1StepSize * baseN * dtype_b
    B_load_hbm_s = B_slab_bytes / hbm_bw if hbm_bw > 0 else 0  # first m_block: cold
    B_load_l2_s = B_slab_bytes / l2_bw if l2_bw > 0 else 0      # subsequent m_blocks: L2

    # A block: baseM × kL1StepSize × dtype_a
    A_block_bytes = baseM * kL1StepSize * dtype_a

    # Per (kL1, nb, mb):
    #   A: nb=0 from HBM (or L2 if a_reused), nb≥1 from L2 (intra-kL1 reuse)
    #   B: mb=0 from HBM, mb≥1 from L2 (K-outer reuse across m_blocks)
    per_kl1_s = 0.0
    for nb in range(n_blocks):
        A_load_s = A_block_bytes / l2_bw if (nb > 0 or a_reused) else A_block_bytes / hbm_bw
        if l2_bw <= 0 or hbm_bw <= 0:
            A_load_s = 0.0

        mte1_first_mb = B_load_hbm_s + A_load_s
        mte1_rest_mb = B_load_l2_s + A_load_s
        if l0c_double_buffered:
            per_mb_first_s = max(mte1_first_mb, total_inner_s, fixpipe_time_s)
            per_mb_rest_s = max(mte1_rest_mb, total_inner_s, fixpipe_time_s)
        else:
            per_mb_first_s = max(mte1_first_mb, inner_plus_fix_s)
            per_mb_rest_s = max(mte1_rest_mb, inner_plus_fix_s)

        per_kl1_nb_s = per_mb_first_s + max(m_loops - 1, 0) * per_mb_rest_s
        per_kl1_s += per_kl1_nb_s

    # Total matmul time
    total_s = kL1_loops * per_kl1_s

    # --- Determine bound (steady-state: last nb, last mb) ---
    A_steady_s = A_block_bytes / l2_bw if l2_bw > 0 else 0
    B_steady_s = B_load_l2_s
    mte1_steady = B_steady_s + A_steady_s
    if l0c_double_buffered:
        dominant_s = max(mte1_steady, total_inner_s, fixpipe_time_s)
        if dominant_s == total_inner_s:
            if inner_iter_s == mmad_time_s:
                bound = "CUBE"
            elif inner_iter_s == l0b_time_s:
                bound = "L0B"
            else:
                bound = "L0A"
        elif dominant_s >= mte1_steady:
            bound = "HBM" if not a_reused else "L2"
        else:
            bound = "FIXPIPE"
    else:
        dominant_s = max(mte1_steady, inner_plus_fix_s)
        if dominant_s == inner_plus_fix_s:
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
        name=name, M=M_total, K=K, N=N,
        active_cores=total_cores,
        load_A_hbm_us=(A_block_bytes / hbm_bw * us) if hbm_bw > 0 else 0,
        load_A_l2_us=(A_block_bytes / l2_bw * us) if l2_bw > 0 else 0,
        load_B_hbm_us=B_load_hbm_s * us,
        cube_us=mmad_time_s * stepK * kL1_loops * n_blocks * m_loops * us,
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
