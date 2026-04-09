#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prefill K-outer loop search.

Searches baseM candidates with the K-outer loop reorder (B pinned in L1 to
amortize loading across M-blocks). Optionally combined with Split-KN for
MM1/MM2 to drive the operator into the cube-bound regime in prefill.
"""

from dataclasses import dataclass
from typing import List

from ..model.block_spec import MatmulBlockSpec
from ..model.hardware import AscendHWSpec
from ..model.matmul_split_kn import SplitKNSpec
from ..model.operator_params import OperatorParams
from ..model.orchestration import (
    estimate_all_stages_k_outer, get_default_block_specs,
)
from ..model.timing_types import VectorOpTiming
from .config import TilingConfig
from .split_kn_search import _gen_kn_candidates


@dataclass
class PrefillResult:
    """Result of a prefill (K-outer) pipeline evaluation."""
    baseM: int = 0
    loop_order: str = ""
    pipeline_us: float = 0.0
    mm1_cube_us: float = 0.0
    mm2_cube_us: float = 0.0
    mm3_cube_us: float = 0.0
    mm4_cube_us: float = 0.0
    vec_total_us: float = 0.0
    aic_busy_us: float = 0.0
    aiv_busy_us: float = 0.0
    cube_utilization: float = 0.0
    mm3_bound: str = ""
    label: str = ""


def search_prefill_optimal(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[PrefillResult]:
    """Search K-outer configs for prefill with full pipeline evaluation.

    Searches baseM candidates with K-outer loop order.
    Also tests combining Split-KN for MM1/MM2 with K-outer for MM3.
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

    def _eval_k_outer(baseM, mm1_kn, mm2_kn, label):
        block_specs = get_default_block_specs(params, hw)
        for name, spec in block_specs.items():
            if spec.mode == "full_load":
                continue
            new_bm = baseM
            while new_bm > 16:
                if (new_bm * spec.baseK * spec.dtype_a_size * 2 <= hw.cache.l0a_size and
                    spec.baseK * spec.baseN * spec.dtype_b_size * 2 <= hw.cache.l0b_size and
                    new_bm * spec.baseN * spec.dtype_c_size * 2 <= hw.cache.l0c_size):
                    break
                new_bm //= 2
            block_specs[name] = MatmulBlockSpec(
                spec.name, new_bm, spec.baseN, spec.baseK, spec.stepK,
                mode=spec.mode, dtype_a_size=spec.dtype_a_size,
                dtype_b_size=spec.dtype_b_size, dtype_c_size=spec.dtype_c_size,
            )

        stages = estimate_all_stages_k_outer(
            params, tiling, hw, block_specs=block_specs,
            mm1_split_kn=mm1_kn, mm2_split_kn=mm2_kn,
        )
        result = build_pipeline_dag(params, tiling, hw, stages)
        vec_total = sum(
            s.total_us for s in stages.values()
            if isinstance(s, VectorOpTiming)
        )
        mm3_bound = stages["MM3_QcQr"].bound if hasattr(stages["MM3_QcQr"], 'bound') else ""
        return PrefillResult(
            baseM=baseM,
            loop_order="k_outer",
            pipeline_us=result.total_us,
            mm1_cube_us=stages["MM1_Cq"].total_us,
            mm2_cube_us=stages["MM2_CkvKr"].total_us,
            mm3_cube_us=stages["MM3_QcQr"].total_us,
            mm4_cube_us=stages["MM4_Qn"].total_us,
            vec_total_us=vec_total,
            aic_busy_us=result.aic_busy_us,
            aiv_busy_us=result.aiv_busy_us,
            cube_utilization=result.aic_busy_us / result.total_us if result.total_us > 0 else 0,
            mm3_bound=mm3_bound,
            label=label,
        )

    results = []

    baseM_candidates = [32, 64, 128, 192, 256]

    # Pure K-outer (no Split-KN)
    for bm in baseM_candidates:
        results.append(_eval_k_outer(bm, None, None, f"KO_bm{bm}"))

    # K-outer with Split-KN for MM1/MM2
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    Dr = params.Dr
    max_cores = hw.aic_num

    best_mm1_kn = None
    for sn, sk, ng, kg in _gen_kn_candidates(He, Hcq, max_cores):
        if ng * kg <= max_cores:
            best_mm1_kn = SplitKNSpec("MM1_Cq", ng, kg, sn, sk)
            break

    best_mm2_kn = None
    for sn, sk, ng, kg in _gen_kn_candidates(He, Hckv + Dr, max_cores):
        if ng * kg <= max_cores:
            best_mm2_kn = SplitKNSpec("MM2_CkvKr", ng, kg, sn, sk)
            break

    if best_mm1_kn and best_mm2_kn:
        for bm in baseM_candidates:
            results.append(_eval_k_outer(
                bm, best_mm1_kn, best_mm2_kn, f"KO_bm{bm}+KN"))

    results.sort(key=lambda r: r.pipeline_us)
    return results
