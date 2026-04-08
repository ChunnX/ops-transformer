#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-axis split (Split-M) search.

Searches m_groups (core count) × baseM (block height) configurations for the
prefill scenario, where each core handles a subset of M tokens with the full
N. Larger baseM enables MMAD to dominate over L2-bound B-loading, making the
matmul cube-bound.
"""

import math
from dataclasses import dataclass
from typing import List

from ..model.block_spec import MatmulBlockSpec
from ..model.hardware import AscendHWSpec
from ..model.matmul_split_m import SplitMSpec
from ..model.operator_params import OperatorParams
from ..model.orchestration import (
    estimate_all_stages_split_m, get_default_block_specs,
)
from ..model.timing_types import VectorOpTiming
from .config import TilingConfig


@dataclass
class SplitMResult:
    """Result of an M-axis split pipeline evaluation."""
    m_groups: int = 0
    single_m: int = 0
    pipeline_us: float = 0.0
    mm1_cube_us: float = 0.0
    mm2_cube_us: float = 0.0
    mm3_cube_us: float = 0.0
    mm4_cube_us: float = 0.0
    vec_total_us: float = 0.0
    aic_busy_us: float = 0.0
    aiv_busy_us: float = 0.0
    cube_utilization: float = 0.0
    label: str = ""


def _get_split_m_block_specs(params: 'OperatorParams', hw: AscendHWSpec,
                             baseM: int = 32):
    """Get block specs optimized for split-M (larger baseM for compute overlap)."""
    specs = get_default_block_specs(params, hw)
    for name, spec in specs.items():
        if spec.mode == "full_load":
            continue
        # Override baseM — key for split-M to reach cube-bound
        # Check L0A: baseM × baseK × dtype × 2 ≤ l0a_size
        new_baseM = baseM
        while new_baseM > 16:
            if new_baseM * spec.baseK * spec.dtype_a_size * 2 <= hw.cache.l0a_size:
                # Also check L0C: baseM × baseN × dtype_c × 2 ≤ l0c_size
                if new_baseM * spec.baseN * spec.dtype_c_size * 2 <= hw.cache.l0c_size:
                    break
            new_baseM //= 2
        specs[name] = MatmulBlockSpec(
            spec.name, new_baseM, spec.baseN, spec.baseK, spec.stepK,
            mode=spec.mode, dtype_a_size=spec.dtype_a_size,
            dtype_b_size=spec.dtype_b_size, dtype_c_size=spec.dtype_c_size,
        )
    return specs


def search_split_m(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[SplitMResult]:
    """Search M-axis split configs for prefill using full pipeline evaluation.

    Searches m_groups (core count) × baseM (block height) for optimal config.
    Larger baseM generates more compute per N-block, enabling overlap with B
    matrix L2 loading (transition from L2-bound to cube-bound).
    """
    try:
        from ..pipeline.dag import build_pipeline_dag
        from ..model.orchestration import estimate_all_stages
    except ImportError:
        from pipeline_model import build_pipeline_dag
        from perf_model import estimate_all_stages

    if not hw.has_cycle_spec:
        return []

    T = params.T
    if T <= 1:
        return []

    def _eval(m_groups, block_specs, label):
        single_m = math.ceil(T / m_groups)
        spec = SplitMSpec("split_m", m_groups, single_m)
        stages = estimate_all_stages_split_m(params, tiling, hw, spec,
                                             block_specs=block_specs)
        result = build_pipeline_dag(params, tiling, hw, stages)
        vec_total = sum(
            s.total_us for s in stages.values()
            if isinstance(s, VectorOpTiming)
        )
        return SplitMResult(
            m_groups=m_groups,
            single_m=single_m,
            pipeline_us=result.total_us,
            mm1_cube_us=stages["MM1_Cq"].total_us,
            mm2_cube_us=stages["MM2_CkvKr"].total_us,
            mm3_cube_us=stages["MM3_QcQr"].total_us,
            mm4_cube_us=stages["MM4_Qn"].total_us,
            vec_total_us=vec_total,
            aic_busy_us=result.aic_busy_us,
            aiv_busy_us=result.aiv_busy_us,
            cube_utilization=result.aic_busy_us / result.total_us if result.total_us > 0 else 0,
            label=label,
        )

    results = []

    # Search m_groups candidates
    m_candidates = [m for m in [2, 4, 8, 12, 16, 24, 32]
                    if m <= hw.aic_num and math.ceil(T / m) >= 16]
    if hw.aic_num not in m_candidates and math.ceil(T / hw.aic_num) >= 1:
        m_candidates.append(hw.aic_num)

    # Search baseM candidates: larger baseM enables cube-bound at cost of more m_loops
    baseM_candidates = [32, 64, 128, 192, 256]

    for m in sorted(set(m_candidates)):
        best_for_m = None
        for bm in baseM_candidates:
            single_m = math.ceil(T / m)
            if bm > single_m:
                continue  # baseM can't exceed M_per_core
            block_specs = _get_split_m_block_specs(params, hw, baseM=bm)
            label = f"M{m}_bm{bm}"
            r = _eval(m, block_specs, label)
            if best_for_m is None or r.pipeline_us < best_for_m.pipeline_us:
                best_for_m = r
        if best_for_m is not None:
            results.append(best_for_m)

    results.sort(key=lambda r: r.pipeline_us)
    return results
