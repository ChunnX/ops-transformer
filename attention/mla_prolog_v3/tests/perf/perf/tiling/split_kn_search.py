#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D N×K split (Split-KN) search.

Generates valid (n_groups × k_groups = total_cores) factorizations for MM1
and MM2 and evaluates each via the full pipeline DAG. Returns the best
configuration sorted by end-to-end pipeline duration.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..model.hardware import AscendHWSpec
from ..model.matmul_split_kn import SplitKNSpec
from ..model.operator_params import OperatorParams
from ..model.timing_types import VectorOpTiming
from .config import TilingConfig


@dataclass
class SplitKNResult:
    """Result of a 2D N×K split pipeline evaluation."""
    mm1_spec: Optional[SplitKNSpec] = None
    mm2_spec: Optional[SplitKNSpec] = None
    pipeline_us: float = 0.0        # total kernel duration (pipeline critical path)
    mm1_cube_us: float = 0.0
    mm2_cube_us: float = 0.0
    mm3_cube_us: float = 0.0
    mm4_cube_us: float = 0.0
    mm1_accum_us: float = 0.0
    mm2_accum_us: float = 0.0
    aic_busy_us: float = 0.0
    aiv_busy_us: float = 0.0
    cube_utilization: float = 0.0    # aic_busy / pipeline_total
    label: str = ""


def _gen_kn_candidates(K: int, N: int, max_cores: int):
    """Generate valid (singleN, singleK) -> (n_groups, k_groups) pairs."""
    single_n_cands = [sn for sn in range(64, N + 1, 64) if N % sn == 0]
    if not single_n_cands:
        single_n_cands = [N]
    single_k_cands = [sk for sk in range(128, K + 1, 128) if K % sk == 0]
    if not single_k_cands:
        single_k_cands = [K]
    if K not in single_k_cands:
        single_k_cands.append(K)

    seen = set()
    for sn in single_n_cands:
        n_groups = N // sn
        for sk in single_k_cands:
            k_groups = K // sk
            total = n_groups * k_groups
            if total > max_cores or total < 1:
                continue
            key = (n_groups, k_groups)
            if key in seen:
                continue
            seen.add(key)
            yield sn, sk, n_groups, k_groups


def search_split_kn(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[SplitKNResult]:
    """Search split-KN configs for MM1 and MM2 using full pipeline evaluation.

    Strategy: search MM1 and MM2 independently (fix the other at baseline),
    then combine the best of each. Returns all evaluated configs sorted by
    pipeline duration.
    """
    try:
        from ..pipeline.dag import build_pipeline_dag
        from ..model.orchestration import estimate_all_stages
    except ImportError:
        from pipeline_model import build_pipeline_dag
        from perf_model import estimate_all_stages

    if not hw.has_cycle_spec:
        return []

    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    Dr = params.Dr
    max_cores = hw.aic_num

    def _eval(mm1_spec, mm2_spec, label):
        stages = estimate_all_stages(params, tiling, hw,
                                     mm1_split_kn=mm1_spec, mm2_split_kn=mm2_spec)
        result = build_pipeline_dag(params, tiling, hw, stages)
        return SplitKNResult(
            mm1_spec=mm1_spec,
            mm2_spec=mm2_spec,
            pipeline_us=result.total_us,
            mm1_cube_us=stages["MM1_Cq"].total_us,
            mm2_cube_us=stages["MM2_CkvKr"].total_us,
            mm3_cube_us=stages["MM3_QcQr"].total_us,
            mm4_cube_us=stages["MM4_Qn"].total_us,
            mm1_accum_us=stages.get("AccumCq", VectorOpTiming("")).total_us,
            mm2_accum_us=stages.get("AccumCkvKr", VectorOpTiming("")).total_us,
            aic_busy_us=result.aic_busy_us,
            aiv_busy_us=result.aiv_busy_us,
            cube_utilization=result.aic_busy_us / result.total_us if result.total_us > 0 else 0,
            label=label,
        )

    results = []

    # Baseline: no split-KN
    results.append(_eval(None, None, "baseline"))

    # Search MM1 configs (MM2 at baseline)
    for sn, sk, ng, kg in _gen_kn_candidates(He, Hcq, max_cores):
        spec = SplitKNSpec("MM1_Cq", ng, kg, sn, sk)
        lbl = f"MM1:N{ng}xK{kg}"
        results.append(_eval(spec, None, lbl))

    # Search MM2 configs (MM1 at baseline)
    for sn, sk, ng, kg in _gen_kn_candidates(He, Hckv + Dr, max_cores):
        spec = SplitKNSpec("MM2_CkvKr", ng, kg, sn, sk)
        lbl = f"MM2:N{ng}xK{kg}"
        results.append(_eval(None, spec, lbl))

    # Combined: best MM1 + best MM2
    mm1_results = [r for r in results if r.mm1_spec and not r.mm2_spec]
    mm2_results = [r for r in results if r.mm2_spec and not r.mm1_spec]
    if mm1_results and mm2_results:
        best_mm1 = min(mm1_results, key=lambda r: r.pipeline_us)
        best_mm2 = min(mm2_results, key=lambda r: r.pipeline_us)
        combined = _eval(best_mm1.mm1_spec, best_mm2.mm2_spec,
                         f"{best_mm1.label}+{best_mm2.label}")
        results.append(combined)

    results.sort(key=lambda r: r.pipeline_us)
    return results
