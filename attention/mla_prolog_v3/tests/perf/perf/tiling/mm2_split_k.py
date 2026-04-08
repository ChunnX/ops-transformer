#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MM2 cross-core K-split search.

Compares the standard split-N approach for MM2 against several K-split
configurations to find the best per-core partition.
"""

from dataclasses import dataclass
from typing import List

from ..model.hardware import AscendHWSpec
from ..model.matmul_detailed import estimate_matmul_detailed
from ..model.matmul_split_kn import estimate_mm2_split_k_cross_core
from ..model.operator_params import OperatorParams
from ..model.orchestration import get_default_block_specs
from ..model.timing_types import MatMulTiming, VectorOpTiming
from .config import TilingConfig


@dataclass
class MM2SplitKResult:
    """Result of comparing split-N vs split-K for MM2."""
    k_cores: int
    mode: str             # "split_n" or "split_k"
    cube_timing: MatMulTiming = None
    accum_timing: VectorOpTiming = None
    cube_us: float = 0.0
    accum_us: float = 0.0
    total_us: float = 0.0


def search_mm2_split_k(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[MM2SplitKResult]:
    """Search split-K configurations for MM2 and compare with split-N baseline.

    Returns sorted list of MM2SplitKResult (best first).
    """
    if not hw.has_cycle_spec:
        return []

    T = tiling.step_batch_size
    He = params.He
    N = params.Hckv + params.Dr
    qm = params.quant_mode
    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size
    vec_cores = tiling.vector_block_num

    block_specs = get_default_block_specs(params, hw)
    mm2_block = block_specs["MM2_CkvKr"]

    results = []

    # Baseline: split-N (current approach)
    split_n_timing = estimate_matmul_detailed(
        "MM2_CkvKr", T, He, N,
        active_cores=tiling.mm2_block_num,
        hw=hw, block=mm2_block, a_reused=True,
    )
    results.append(MM2SplitKResult(
        k_cores=tiling.mm2_block_num,
        mode="split_n",
        cube_timing=split_n_timing,
        cube_us=split_n_timing.total_us,
        accum_us=0.0,
        total_us=split_n_timing.total_us,
    ))

    # Split-K candidates
    k_candidates = [9, 12, 16, 18, 24, 32]
    for k_cores in k_candidates:
        if k_cores > hw.aic_num:
            continue
        cube_t, accum_t = estimate_mm2_split_k_cross_core(
            T, He, N, k_cores, hw, mm2_block,
            a_reused=True, vec_cores=vec_cores, out_size=out_size,
        )
        results.append(MM2SplitKResult(
            k_cores=k_cores,
            mode="split_k",
            cube_timing=cube_t,
            accum_timing=accum_t,
            cube_us=cube_t.total_us,
            accum_us=accum_t.total_us,
            total_us=cube_t.total_us + accum_t.total_us,
        ))

    results.sort(key=lambda r: r.total_us)
    return results
