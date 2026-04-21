#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L0C→UB core-internal pipeline search.

Searches m_groups configurations with UB-constrained baseM for the L0C→UB
prefill optimization. Each core handles a subset of M tokens with full N,
using core-internal data paths (L0C→UB, UB→L1) instead of HBM for
intermediate results.
"""

import math
from dataclasses import dataclass
from typing import List

from ..model.block_spec import MatmulBlockSpec
from ..model.hardware import AscendHWSpec
from ..model.matmul_split_m import SplitMSpec
from ..model.operator_params import OperatorParams
from ..model.orchestration import (
    compute_ub_constrained_baseM,
    estimate_all_stages_l0c_ub,
    get_default_block_specs,
)
from ..model.timing_types import VectorOpTiming
from .config import TilingConfig


@dataclass
class L0CUBResult:
    """Result of an L0C→UB pipeline evaluation."""
    m_groups: int = 0
    single_m: int = 0
    baseM: int = 0
    m_loops: int = 0
    pipeline_us: float = 0.0
    mm1_cube_us: float = 0.0
    mm2_cube_us: float = 0.0
    mm3_cube_us: float = 0.0
    mm4_cube_us: float = 0.0
    vec_total_us: float = 0.0
    ub_l1_total_us: float = 0.0
    aic_busy_us: float = 0.0
    aiv_busy_us: float = 0.0
    cube_utilization: float = 0.0
    mm3_bound: str = ""
    label: str = ""


def _get_l0c_ub_block_specs(params: OperatorParams, hw: AscendHWSpec,
                            baseM: int):
    """Get block specs with UB-constrained baseM for L0C→UB path."""
    specs = get_default_block_specs(params, hw)
    for name, spec in specs.items():
        if spec.mode == "full_load":
            continue
        new_baseM = baseM
        while new_baseM > 16:
            if (new_baseM * spec.baseK * spec.dtype_a_size * 2 <= hw.cache.l0a_size and
                    spec.baseK * spec.baseN * spec.dtype_b_size * 2 <= hw.cache.l0b_size and
                    new_baseM * spec.baseN * spec.dtype_c_size * 2 <= hw.cache.l0c_size):
                break
            new_baseM //= 2
        specs[name] = MatmulBlockSpec(
            spec.name, new_baseM, spec.baseN, spec.baseK, spec.stepK,
            mode=spec.mode, dtype_a_size=spec.dtype_a_size,
            dtype_b_size=spec.dtype_b_size, dtype_c_size=spec.dtype_c_size,
        )
    return specs


def search_l0c_ub(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[L0CUBResult]:
    """Search L0C→UB configs with UB-constrained baseM.

    Uses the UB capacity constraint to determine baseM, then searches
    m_groups (core count) for optimal configuration.
    """
    try:
        from ..pipeline.dag import build_pipeline_dag_l0c_ub
    except ImportError:
        return []

    if not hw.has_cycle_spec:
        return []

    T = params.T
    if T <= 1:
        return []

    baseM = compute_ub_constrained_baseM(params.Hcq, hw.cache.ub_size)

    def _eval(m_groups, label):
        single_m = math.ceil(T / m_groups)
        m_loops = math.ceil(single_m / baseM) if baseM > 0 else 1
        spec = SplitMSpec("l0c_ub", m_groups, single_m)
        block_specs = _get_l0c_ub_block_specs(params, hw, baseM)
        stages = estimate_all_stages_l0c_ub(params, tiling, hw, spec,
                                            block_specs=block_specs)
        result = build_pipeline_dag_l0c_ub(params, tiling, hw, stages)

        vec_total = sum(
            s.total_us for s in stages.values()
            if isinstance(s, VectorOpTiming) and not s.name.startswith("UB_to_L1")
        )
        ub_l1_total = sum(
            s.total_us for s in stages.values()
            if isinstance(s, VectorOpTiming) and s.name.startswith("UB_to_L1")
        )
        mm3_bound = stages["MM3_QcQr"].bound if hasattr(stages["MM3_QcQr"], 'bound') else ""

        return L0CUBResult(
            m_groups=m_groups,
            single_m=single_m,
            baseM=baseM,
            m_loops=m_loops,
            pipeline_us=result.total_us,
            mm1_cube_us=stages["MM1_Cq"].total_us,
            mm2_cube_us=stages["MM2_CkvKr"].total_us,
            mm3_cube_us=stages["MM3_QcQr"].total_us,
            mm4_cube_us=stages["MM4_Qn"].total_us,
            vec_total_us=vec_total,
            ub_l1_total_us=ub_l1_total,
            aic_busy_us=result.aic_busy_us,
            aiv_busy_us=result.aiv_busy_us,
            cube_utilization=result.aic_busy_us / result.total_us if result.total_us > 0 else 0,
            mm3_bound=mm3_bound,
            label=label,
        )

    results = []

    m_candidates = [m for m in [2, 4, 8, 12, 16, 24, 32]
                    if m <= hw.aic_num and math.ceil(T / m) >= 16]
    if hw.aic_num not in m_candidates and math.ceil(T / hw.aic_num) >= 1:
        m_candidates.append(hw.aic_num)

    for m in sorted(set(m_candidates)):
        label = f"L0C_M{m}_bm{baseM}"
        results.append(_eval(m, label))

    results.sort(key=lambda r: r.pipeline_us)
    return results
