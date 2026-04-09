#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-Split prefill search: find optimal m_groups × n_groups configuration.

Searches 2D core grid configurations for prefill scenarios, combining M-Split
(eliminate full-core barriers) with K-outer loop (cube-bound target).

Reports per-config: pipeline time, sync overhead, workspace size, cube util,
and comparison vs split-N baseline.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..model.block_spec import MatmulBlockSpec, derive_stepK
from ..model.hardware import AscendHWSpec
from ..model.matmul_msplit import MSplitSpec, estimate_matmul_msplit
from ..model.operator_params import OperatorParams
from ..model.orchestration import estimate_all_stages, get_default_block_specs
from ..model.quant import is_full_quant, is_int8_quant, is_mxfp8
from ..model.timing_types import MatMulTiming, VectorOpTiming
from ..model.vector_ops import estimate_vector_op, vector_instr_count
from .config import TilingConfig


@dataclass
class MSplitResult:
    """Result of an M-Split prefill pipeline evaluation."""
    m_groups: int = 0
    n_groups: int = 0
    M_per_core: int = 0
    N_per_core_mm3: int = 0
    baseM: int = 0
    m_loops: int = 0
    pipeline_us: float = 0.0
    mm1_us: float = 0.0
    mm2_us: float = 0.0
    mm3_us: float = 0.0
    mm4_us: float = 0.0
    vec_us: float = 0.0
    sync_us: float = 0.0
    n_sync_events: int = 0
    workspace_mb: float = 0.0
    cube_util: float = 0.0
    mm3_bound: str = ""
    label: str = ""


def _estimate_all_stages_msplit(
    params: OperatorParams,
    hw: AscendHWSpec,
    m_groups: int,
    n_groups: int,
    baseM: int,
) -> Dict:
    """Estimate all matmul + vector stages under M-Split configuration."""
    T = params.T
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    qm = params.quant_mode
    out_size = params.mm_output_dtype_size

    total_cores = m_groups * n_groups
    M_per_core = math.ceil(T / m_groups)

    # Per-matmul dimensions and MSplitSpec
    mm_configs = {
        "MM1_Cq":    (He, Hcq),
        "MM2_CkvKr": (He, Hckv + Dr),
        "MM3_QcQr":  (Hcq, N * (D + Dr)),
        "MM4_Qn":    (D, Hckv),
    }

    block_specs = get_default_block_specs(params, hw)
    stages = {}

    for mm_name, (K, N_total) in mm_configs.items():
        spec_block = block_specs[mm_name]

        # Override baseM for cube-bound targeting
        actual_baseM = baseM
        # Validate L0A fit: baseM × baseK × dtype_a × 2 <= L0A
        while actual_baseM > 16:
            if (actual_baseM * spec_block.baseK * spec_block.dtype_a_size * 2
                    <= hw.cache.l0a_size
                and spec_block.baseK * spec_block.baseN * spec_block.dtype_b_size * 2
                    <= hw.cache.l0b_size
                and actual_baseM * spec_block.baseN * spec_block.dtype_c_size * 2
                    <= hw.cache.l0c_size):
                break
            actual_baseM //= 2

        new_block = MatmulBlockSpec(
            spec_block.name, actual_baseM, spec_block.baseN, spec_block.baseK,
            spec_block.stepK, mode=spec_block.mode,
            dtype_a_size=spec_block.dtype_a_size,
            dtype_b_size=spec_block.dtype_b_size,
            dtype_c_size=spec_block.dtype_c_size,
        )

        N_per_core = math.ceil(N_total / n_groups) if n_groups > 1 else N_total
        a_reused = (mm_name == "MM2_CkvKr")

        msplit_spec = MSplitSpec(
            name=mm_name,
            m_groups=m_groups,
            n_groups=n_groups,
            M_per_core=M_per_core,
            N_per_core=N_per_core,
        )

        if spec_block.mode == "full_load":
            # MM4 is small, use full-load model with M-split dimensions
            from ..model.matmul_detailed import estimate_matmul_detailed
            stages[mm_name] = estimate_matmul_detailed(
                mm_name, M_per_core, K, N_total,
                active_cores=n_groups, hw=hw, block=new_block,
                a_reused=a_reused,
            )
        else:
            stages[mm_name] = estimate_matmul_msplit(
                mm_name, T, K, N_total, msplit_spec,
                hw=hw, block=new_block, a_reused=a_reused,
            )

    # Vector ops: each m-group has n_groups × cv_ratio AIV cores
    # With CV 1:2 mode, each AIC core has 2 AIV cores
    cv_ratio = hw.aiv_num // hw.aic_num if hw.aic_num > 0 else 2
    vec_cores_per_mgroup = n_groups * cv_ratio

    rmsnorm_cq_load = M_per_core * Hcq * out_size
    rmsnorm_cq_store = M_per_core * Hcq * 2
    stages["RmsNormCq"] = estimate_vector_op(
        "RmsNormCq", rmsnorm_cq_load, M_per_core * Hcq, rmsnorm_cq_store,
        hw, active_cores=vec_cores_per_mgroup,
        n_instructions=vector_instr_count("RmsNormCq", qm),
    )

    rmsnorm_ckvkr_load = M_per_core * Hckv * out_size
    rmsnorm_ckvkr_store = M_per_core * Hckv * 2
    stages["RmsNormCkvKr"] = estimate_vector_op(
        "RmsNormCkvKr", rmsnorm_ckvkr_load, M_per_core * Hckv, rmsnorm_ckvkr_store,
        hw, active_cores=vec_cores_per_mgroup,
        n_instructions=vector_instr_count("RmsNormCkvKr", qm),
    )

    rope_kr_load = M_per_core * Dr * 4
    stages["RopeKr"] = estimate_vector_op(
        "RopeKr", rope_kr_load, M_per_core * Dr, M_per_core * Dr * 2,
        hw, active_cores=vec_cores_per_mgroup,
        n_instructions=vector_instr_count("RopeKr", qm),
    )

    scatter_load = M_per_core * (Hckv + Dr) * 2
    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", scatter_load, M_per_core * (Hckv + Dr),
        M_per_core * (Hckv + Dr) * 2,
        hw, active_cores=vec_cores_per_mgroup, is_scatter=True,
        n_instructions=vector_instr_count("ScatterCkvKr", qm),
    )

    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        dequant_load = M_per_core * N * D * out_size
        stages["DequantQc"] = estimate_vector_op(
            "DequantQc", dequant_load, M_per_core * N * D,
            M_per_core * N * D * 2,
            hw, active_cores=vec_cores_per_mgroup,
            n_instructions=vector_instr_count("DequantQc", qm),
        )

    rope_qr_load = M_per_core * N * Dr * 2 + M_per_core * Dr * 2
    stages["RopeQr"] = estimate_vector_op(
        "RopeQr", rope_qr_load, M_per_core * N * Dr,
        M_per_core * N * Dr * 2,
        hw, active_cores=vec_cores_per_mgroup,
        n_instructions=vector_instr_count("RopeQr", qm),
    )

    if params.query_quant_mode == 1:
        stages["DynamicQuantQn"] = estimate_vector_op(
            "DynamicQuantQn", M_per_core * N * Hckv * 2,
            M_per_core * N * Hckv, M_per_core * N * Hckv,
            hw, active_cores=vec_cores_per_mgroup,
            n_instructions=vector_instr_count("DynamicQuantQn", qm),
        )
        stages["MulQr"] = estimate_vector_op(
            "MulQr", M_per_core * N * Dr * 2, M_per_core * N * Dr,
            M_per_core * N * Dr * 2,
            hw, active_cores=vec_cores_per_mgroup,
            n_instructions=vector_instr_count("MulQr", qm),
        )

    return stages


def _compute_pipeline_msplit(stages: Dict, hw: AscendHWSpec,
                             n_groups: int) -> tuple:
    """Compute simplified pipeline for M-Split: AIC→AIV serial with local barriers.

    In M-Split mode, the pipeline within each m-group is:
      MM1 → [LocalBarrier if n_groups>1] → RmsNormCq →
      MM2 → [LocalBarrier if n_groups>1] → RmsNormCkvKr+RoPE+Scatter →
      MM3 → [LocalBarrier if n_groups>1] → DequantQc+RopeQr → MM4

    No full-core WaitAllCore. LocalBarrier cost = (n_groups-1) × sync_overhead.
    AIC↔AIV paired CrossCore sync cost included per stage transition.
    """
    sync_per_barrier = (n_groups - 1) * hw.sync_overhead_us if n_groups > 1 else 0
    aic_aiv_sync = hw.sync_overhead_us  # CrossCore AIC↔AIV per transition

    def get_us(name):
        t = stages.get(name)
        return t.total_us if t is not None else 0.0

    # Count barriers: 3 stage transitions that need local barrier
    # MM1→RmsNormCq, MM2→RmsNormCkvKr, MM3→RopeQr
    n_barriers = 3 if n_groups > 1 else 0
    total_barrier_us = n_barriers * sync_per_barrier

    # AIC↔AIV sync: 5 transitions (same as current kernel)
    # MM1→AIV, AIV→MM3, MM2→AIV, MM3→AIV, MM4 dependencies
    n_aic_aiv_syncs = 5
    total_aic_aiv_sync_us = n_aic_aiv_syncs * aic_aiv_sync

    total_sync_us = total_barrier_us + total_aic_aiv_sync_us

    # AIC stages (sequential within each m-group)
    aic_busy = (get_us("MM1_Cq") + get_us("MM2_CkvKr") +
                get_us("MM3_QcQr") + get_us("MM4_Qn"))

    # AIV stages (overlap with AIC where possible)
    aiv_stages = ["RmsNormCq", "RmsNormCkvKr", "RopeKr", "ScatterCkvKr",
                  "DequantQc", "RopeQr", "DynamicQuantQn", "MulQr"]
    aiv_busy = sum(get_us(s) for s in aiv_stages)

    # Pipeline: AIC dominates (cube-bound), AIV overlaps
    # Critical path ≈ max(AIC + sync, AIV total with dependencies)
    # In practice AIC >> AIV for prefill, so pipeline ≈ AIC + sync
    pipeline_us = aic_busy + total_sync_us

    return pipeline_us, aic_busy, aiv_busy, total_sync_us, n_barriers + n_aic_aiv_syncs


def _compute_workspace_mb(params: OperatorParams, m_groups: int,
                          n_groups: int, baseM: int) -> float:
    """Estimate total workspace in MB for M-Split configuration.

    Main workspace consumers:
    - MM1 output (Cq): M_per_core × Hcq × out_size per m-group
    - MM2 output (CkvKr): M_per_core × (Hckv+Dr) × out_size per m-group
    - RmsNormCq output: M_per_core × Hcq × 2 per m-group
    - MM3 output (QcQr): baseM × N×(D+Dr) × out_size per m-group (m-block pipeline reuse)
    - MM4: direct output, no workspace
    """
    M_per_core = math.ceil(params.T / m_groups)
    out_size = params.mm_output_dtype_size
    N_qcqr = params.N * (params.D + params.Dr)

    mm1_ws = M_per_core * params.Hcq * out_size
    mm2_ws = M_per_core * (params.Hckv + params.Dr) * out_size
    rmsnorm_ws = M_per_core * params.Hcq * 2
    # MM3: m-block pipeline reuse — only baseM rows at a time
    mm3_ws = baseM * N_qcqr * out_size

    per_mgroup = mm1_ws + mm2_ws + rmsnorm_ws + mm3_ws
    total_bytes = m_groups * per_mgroup
    return total_bytes / (1024 * 1024)


def search_msplit_optimal(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> List[MSplitResult]:
    """Search M-Split configurations for prefill optimization.

    Searches m_groups × n_groups = aic_num factorizations,
    with baseM candidates for cube-bound targeting.
    """
    if not hw.has_cycle_spec:
        return []

    T = params.T
    if T <= 1:
        return []

    max_cores = hw.aic_num
    N_qcqr = params.N * (params.D + params.Dr)

    results = []

    # Generate valid (m_groups, n_groups) factorizations
    factorizations = []
    for mg in range(1, max_cores + 1):
        if max_cores % mg == 0:
            ng = max_cores // mg
            M_per_core = math.ceil(T / mg)
            if M_per_core >= 16:  # minimum useful M slice
                factorizations.append((mg, ng, M_per_core))

    baseM_candidates = [32, 64, 128, 256]

    for mg, ng, M_per_core in factorizations:
        for baseM in baseM_candidates:
            if baseM > M_per_core:
                continue  # baseM can't exceed M_per_core

            m_loops = math.ceil(M_per_core / baseM)
            N_per_core_mm3 = math.ceil(N_qcqr / ng)
            label = f"M{mg}×N{ng}_bm{baseM}"

            try:
                stages = _estimate_all_stages_msplit(params, hw, mg, ng, baseM)
                pipeline_us, aic_busy, aiv_busy, sync_us, n_syncs = \
                    _compute_pipeline_msplit(stages, hw, ng)
                workspace_mb = _compute_workspace_mb(params, mg, ng, baseM)

                mm3_bound = ""
                if hasattr(stages.get("MM3_QcQr", None), 'bound'):
                    mm3_bound = stages["MM3_QcQr"].bound

                vec_us = sum(
                    s.total_us for s in stages.values()
                    if isinstance(s, VectorOpTiming)
                )

                cube_util = aic_busy / pipeline_us if pipeline_us > 0 else 0

                results.append(MSplitResult(
                    m_groups=mg,
                    n_groups=ng,
                    M_per_core=M_per_core,
                    N_per_core_mm3=N_per_core_mm3,
                    baseM=baseM,
                    m_loops=m_loops,
                    pipeline_us=pipeline_us,
                    mm1_us=stages["MM1_Cq"].total_us,
                    mm2_us=stages["MM2_CkvKr"].total_us,
                    mm3_us=stages["MM3_QcQr"].total_us,
                    mm4_us=stages["MM4_Qn"].total_us,
                    vec_us=vec_us,
                    sync_us=sync_us,
                    n_sync_events=n_syncs,
                    workspace_mb=workspace_mb,
                    cube_util=cube_util,
                    mm3_bound=mm3_bound,
                    label=label,
                ))
            except Exception:
                continue

    results.sort(key=lambda r: r.pipeline_us)
    return results
