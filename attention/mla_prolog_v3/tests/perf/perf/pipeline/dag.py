#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIC/AIV pipeline DAG and critical path analysis.

Builds the dependency graph from the kernel's CrossCoreSetFlag/WaitFlag
sequence, runs Kahn's topological sort, computes start/end times via a
forward pass, and backtracks the critical path.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..model.dequant_opt import (
    DequantOptStageTiming, estimate_dequant_opt_pipeline,
    is_dequant_opt_applicable,
)
from ..model.dequant_opt import (
    DequantOptStageTiming, estimate_dequant_opt_pipeline,
    is_dequant_opt_applicable,
)
from ..model.hardware import AscendHWSpec
from ..model.operator_params import OperatorParams
from ..model.orchestration import estimate_all_stages
from ..model.quant import is_full_quant, is_int8_quant, is_mxfp8


@dataclass
class PipelineStage:
    name: str
    pipe: str  # "AIC" or "AIV"
    duration_us: float
    depends_on: List[str] = field(default_factory=list)
    start_us: float = 0.0
    end_us: float = 0.0
    on_critical_path: bool = False


@dataclass
class PipelineResult:
    stages: Dict[str, PipelineStage]
    critical_path: List[str]
    total_us: float
    aic_busy_us: float
    aiv_busy_us: float
    sync_overhead_us: float


def build_pipeline_dag(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    stage_timings: Optional[Dict] = None,
) -> PipelineResult:
    """Build and solve the AIC/AIV pipeline DAG.

    Pipeline from kernel_mla_prolog_split_n.h lines 786-910:

    AIC: MM1 -> signal(CQ) -> MM2 -> signal(CKVKR) -> wait(RMSNORM_CQ) -> MM3 -> signal(QCQR) -> wait(DEQUANT_QC) -> MM4
    AIV: CopyInSinCos -> wait(CQ) -> RmsNormCq -> signal(RMSNORM_CQ) -> wait(CKVKR) -> RmsNormCkvKr+RopeKr+Scatter
         -> wait(QCQR) -> [DequantQc] -> signal(DEQUANT_QC) -> RopeQr -> [DynamicQuantQn+MulQr]
    """
    if stage_timings is None:
        stage_timings = estimate_all_stages(params, tiling, hw)

    qm = params.quant_mode
    has_dequant = is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm)
    has_dynamic_quant = params.query_quant_mode == 1

    sync_us = hw.sync_overhead_us

    def get_us(name: str) -> float:
        t = stage_timings.get(name)
        if t is None:
            return 0.0
        return t.total_us

    stages: Dict[str, PipelineStage] = {}

    # --- AIC pipeline ---
    stages["MM1_Cq"] = PipelineStage(
        "MM1_Cq", "AIC", get_us("MM1_Cq"))

    stages["sync_signal_CQ"] = PipelineStage(
        "sync_signal_CQ", "AIC", sync_us, depends_on=["MM1_Cq"])

    stages["MM2_CkvKr"] = PipelineStage(
        "MM2_CkvKr", "AIC", get_us("MM2_CkvKr"), depends_on=["sync_signal_CQ"])

    stages["sync_signal_CKVKR"] = PipelineStage(
        "sync_signal_CKVKR", "AIC", sync_us, depends_on=["MM2_CkvKr"])

    # AIC waits for vector RmsNormCq before MM3
    stages["aic_wait_RMSNORM_CQ"] = PipelineStage(
        "aic_wait_RMSNORM_CQ", "AIC", sync_us,
        depends_on=["sync_signal_CKVKR", "sync_signal_RMSNORM_CQ"])

    stages["MM3_QcQr"] = PipelineStage(
        "MM3_QcQr", "AIC", get_us("MM3_QcQr"),
        depends_on=["aic_wait_RMSNORM_CQ"])

    stages["sync_signal_QCQR"] = PipelineStage(
        "sync_signal_QCQR", "AIC", sync_us, depends_on=["MM3_QcQr"])

    # AIC waits for vector DequantQc before MM4
    if has_dequant:
        stages["aic_wait_DEQUANT_QC"] = PipelineStage(
            "aic_wait_DEQUANT_QC", "AIC", sync_us,
            depends_on=["sync_signal_QCQR", "sync_signal_DEQUANT_QC"])
        stages["MM4_Qn"] = PipelineStage(
            "MM4_Qn", "AIC", get_us("MM4_Qn"),
            depends_on=["aic_wait_DEQUANT_QC"])
    else:
        stages["aic_wait_QCQR_for_rope"] = PipelineStage(
            "aic_wait_QCQR_for_rope", "AIC", sync_us,
            depends_on=["sync_signal_QCQR"])
        stages["MM4_Qn"] = PipelineStage(
            "MM4_Qn", "AIC", get_us("MM4_Qn"),
            depends_on=["aic_wait_QCQR_for_rope"])

    # --- AIV pipeline ---
    stages["CopyInSinCos"] = PipelineStage(
        "CopyInSinCos", "AIV", 0.5)  # Small constant

    stages["aiv_wait_CQ"] = PipelineStage(
        "aiv_wait_CQ", "AIV", sync_us,
        depends_on=["CopyInSinCos", "sync_signal_CQ"])

    # If K-split is used for MM1, add vector accumulation before RmsNorm
    has_accum_cq = "AccumCq" in stage_timings
    if has_accum_cq:
        stages["AccumCq"] = PipelineStage(
            "AccumCq", "AIV", get_us("AccumCq"),
            depends_on=["aiv_wait_CQ"])
        rmsnorm_cq_dep = "AccumCq"
    else:
        rmsnorm_cq_dep = "aiv_wait_CQ"

    stages["RmsNormCq"] = PipelineStage(
        "RmsNormCq", "AIV", get_us("RmsNormCq"),
        depends_on=[rmsnorm_cq_dep])

    stages["sync_signal_RMSNORM_CQ"] = PipelineStage(
        "sync_signal_RMSNORM_CQ", "AIV", sync_us,
        depends_on=["RmsNormCq"])

    stages["aiv_wait_CKVKR"] = PipelineStage(
        "aiv_wait_CKVKR", "AIV", sync_us,
        depends_on=["sync_signal_RMSNORM_CQ", "sync_signal_CKVKR"])

    # If cross-core K-split is used for MM2, add vector accumulation stage
    has_accum_ckvkr = "AccumCkvKr" in stage_timings
    if has_accum_ckvkr:
        stages["AccumCkvKr"] = PipelineStage(
            "AccumCkvKr", "AIV", get_us("AccumCkvKr"),
            depends_on=["aiv_wait_CKVKR"])
        rmsnorm_ckvkr_dep = "AccumCkvKr"
    else:
        rmsnorm_ckvkr_dep = "aiv_wait_CKVKR"

    stages["RmsNormCkvKr"] = PipelineStage(
        "RmsNormCkvKr", "AIV", get_us("RmsNormCkvKr"),
        depends_on=[rmsnorm_ckvkr_dep])

    stages["RopeKr"] = PipelineStage(
        "RopeKr", "AIV", get_us("RopeKr"),
        depends_on=["RmsNormCkvKr"])

    stages["ScatterCkvKr"] = PipelineStage(
        "ScatterCkvKr", "AIV", get_us("ScatterCkvKr"),
        depends_on=["RopeKr"])

    # After MM3 completes: Dequant/Cast Qc, then RoPE Qr
    if has_dequant:
        stages["aiv_wait_QCQR"] = PipelineStage(
            "aiv_wait_QCQR", "AIV", sync_us,
            depends_on=["ScatterCkvKr", "sync_signal_QCQR"])

        stages["DequantQc"] = PipelineStage(
            "DequantQc", "AIV", get_us("DequantQc"),
            depends_on=["aiv_wait_QCQR"])

        stages["sync_signal_DEQUANT_QC"] = PipelineStage(
            "sync_signal_DEQUANT_QC", "AIV", sync_us,
            depends_on=["DequantQc"])

        stages["RopeQr"] = PipelineStage(
            "RopeQr", "AIV", get_us("RopeQr"),
            depends_on=["sync_signal_DEQUANT_QC"])
    else:
        stages["aiv_wait_QCQR_vec"] = PipelineStage(
            "aiv_wait_QCQR_vec", "AIV", sync_us,
            depends_on=["ScatterCkvKr", "sync_signal_QCQR"])

        stages["RopeQr"] = PipelineStage(
            "RopeQr", "AIV", get_us("RopeQr"),
            depends_on=["aiv_wait_QCQR_vec"])

    if has_dynamic_quant:
        stages["DynamicQuantQn"] = PipelineStage(
            "DynamicQuantQn", "AIV", get_us("DynamicQuantQn"),
            depends_on=["RopeQr", "MM4_Qn"])

        stages["MulQr"] = PipelineStage(
            "MulQr", "AIV", get_us("MulQr"),
            depends_on=["DynamicQuantQn"])

    # --- Solve DAG: forward pass longest-path ---
    # Topological sort via Kahn's algorithm
    in_degree = {name: 0 for name in stages}
    for s in stages.values():
        for dep in s.depends_on:
            if dep in stages:
                in_degree[s.name] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        # Process in insertion order for determinism
        node = queue.pop(0)
        topo_order.append(node)
        for s in stages.values():
            if node in s.depends_on:
                in_degree[s.name] -= 1
                if in_degree[s.name] == 0:
                    queue.append(s.name)

    # Forward pass: compute start/end times
    for name in topo_order:
        s = stages[name]
        if s.depends_on:
            s.start_us = max(stages[dep].end_us for dep in s.depends_on if dep in stages)
        else:
            s.start_us = 0.0
        s.end_us = s.start_us + s.duration_us

    # Find total time
    total_us = max(s.end_us for s in stages.values())

    # Backtrack critical path
    critical_path = []
    # Find the stage that ends at total_us
    current = max(stages.values(), key=lambda s: s.end_us)
    while current:
        current.on_critical_path = True
        critical_path.append(current.name)
        if not current.depends_on:
            break
        # Follow the dependency that determined start time
        pred = max(
            (stages[dep] for dep in current.depends_on if dep in stages),
            key=lambda s: s.end_us,
        )
        current = pred

    critical_path.reverse()

    # Compute busy times
    aic_busy = sum(s.duration_us for s in stages.values() if s.pipe == "AIC" and not s.name.startswith("sync_") and not s.name.startswith("aic_wait"))
    aiv_busy = sum(s.duration_us for s in stages.values() if s.pipe == "AIV" and not s.name.startswith("sync_") and not s.name.startswith("aiv_wait"))
    sync_total = sum(s.duration_us for s in stages.values() if s.name.startswith("sync_") or "wait" in s.name)

    return PipelineResult(
        stages=stages,
        critical_path=critical_path,
        total_us=total_us,
        aic_busy_us=aic_busy,
        aiv_busy_us=aiv_busy,
        sync_overhead_us=sync_total,
    )


def _solve_dag(stages: Dict[str, PipelineStage]) -> PipelineResult:
    """Topological sort + forward pass + critical-path backtrack.

    Shared by `build_pipeline_dag` and `build_pipeline_dag_dequant_opt`. The
    body is identical to the inline solver in `build_pipeline_dag`; extracted
    for reuse without changing any numerics.
    """
    in_degree = {name: 0 for name in stages}
    for s in stages.values():
        for dep in s.depends_on:
            if dep in stages:
                in_degree[s.name] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in stages.values():
            if node in s.depends_on:
                in_degree[s.name] -= 1
                if in_degree[s.name] == 0:
                    queue.append(s.name)

    for name in topo_order:
        s = stages[name]
        if s.depends_on:
            s.start_us = max(stages[dep].end_us for dep in s.depends_on if dep in stages)
        else:
            s.start_us = 0.0
        s.end_us = s.start_us + s.duration_us

    total_us = max(s.end_us for s in stages.values())

    critical_path = []
    current = max(stages.values(), key=lambda s: s.end_us)
    while current:
        current.on_critical_path = True
        critical_path.append(current.name)
        if not current.depends_on:
            break
        pred = max(
            (stages[dep] for dep in current.depends_on if dep in stages),
            key=lambda s: s.end_us,
        )
        current = pred
    critical_path.reverse()

    aic_busy = sum(s.duration_us for s in stages.values()
                   if s.pipe == "AIC" and not s.name.startswith("sync_")
                   and not s.name.startswith("aic_wait"))
    aiv_busy = sum(s.duration_us for s in stages.values()
                   if s.pipe == "AIV" and not s.name.startswith("sync_")
                   and not s.name.startswith("aiv_wait"))
    sync_total = sum(s.duration_us for s in stages.values()
                     if s.name.startswith("sync_") or "wait" in s.name)

    return PipelineResult(
        stages=stages,
        critical_path=critical_path,
        total_us=total_us,
        aic_busy_us=aic_busy,
        aiv_busy_us=aiv_busy,
        sync_overhead_us=sync_total,
    )


def build_pipeline_dag_dequant_opt(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    stage_timings: Optional[Dict] = None,
    dequant_opt_timing: Optional[DequantOptStageTiming] = None,
) -> PipelineResult:
    """Build the AIC/AIV pipeline DAG with the dequantOpt UB pipeline applied.

    Differences from `build_pipeline_dag`:
      - Removes the `sync_signal_QCQR`, `aic_wait_DEQUANT_QC`,
        `sync_signal_DEQUANT_QC`, and `aiv_wait_QCQR` cross-core sync stages
        between MM3, the dequant/RoPE vector ops, and MM4.
      - Replaces the (DequantQc + RopeQr) AIV stages with a single
        `DequantRopeUB` stage placed on the AIC pipeline (because in the
        dequantOpt path, the same AI core's vector unit runs the dequant+rope
        synchronously between its cube MM3 and MM4 — there is no work for the
        independent AIV pipeline to schedule against).
      - Routes RopeKr / ScatterCkvKr to terminate at MM3 (they still run on
        the wider AIV pool that is not participating in MM3) instead of being
        a prerequisite for the (now-fused) Qc dequant.

    The dequantOpt path requires `is_dequant_opt_applicable(params, tiling)`
    to be true; otherwise the resulting timing equals the baseline (no win).
    """
    if stage_timings is None:
        stage_timings = estimate_all_stages(params, tiling, hw)

    if dequant_opt_timing is None:
        dequant_opt_timing = estimate_dequant_opt_pipeline(params, tiling, hw)

    qm = params.quant_mode
    has_dynamic_quant = params.query_quant_mode == 1
    sync_us = hw.sync_overhead_us

    def get_us(name: str) -> float:
        t = stage_timings.get(name)
        if t is None:
            return 0.0
        return t.total_us

    stages: Dict[str, PipelineStage] = {}

    # --- AIC pipeline (unchanged through MM3) ---
    stages["MM1_Cq"] = PipelineStage("MM1_Cq", "AIC", get_us("MM1_Cq"))
    stages["sync_signal_CQ"] = PipelineStage(
        "sync_signal_CQ", "AIC", sync_us, depends_on=["MM1_Cq"])
    stages["MM2_CkvKr"] = PipelineStage(
        "MM2_CkvKr", "AIC", get_us("MM2_CkvKr"), depends_on=["sync_signal_CQ"])
    stages["sync_signal_CKVKR"] = PipelineStage(
        "sync_signal_CKVKR", "AIC", sync_us, depends_on=["MM2_CkvKr"])
    stages["aic_wait_RMSNORM_CQ"] = PipelineStage(
        "aic_wait_RMSNORM_CQ", "AIC", sync_us,
        depends_on=["sync_signal_CKVKR", "sync_signal_RMSNORM_CQ"])
    stages["MM3_QcQr"] = PipelineStage(
        "MM3_QcQr", "AIC", get_us("MM3_QcQr"),
        depends_on=["aic_wait_RMSNORM_CQ"])

    # --- DequantOpt fused stage on AIC (intra-core, no cross-core sync) ---
    # Placed on AIC pipeline because the same AI core's cube finishes MM3,
    # then its vector immediately processes UB, then its cube starts MM4.
    # AIV pipeline is free to run RmsNormCkvKr / RopeKr / Scatter independently
    # on the non-MM3 cores in parallel during MM3, but does not gate the fused
    # stage.
    stages["DequantRopeUB"] = PipelineStage(
        "DequantRopeUB", "AIC", dequant_opt_timing.total_us,
        depends_on=["MM3_QcQr"])

    stages["MM4_Qn"] = PipelineStage(
        "MM4_Qn", "AIC", get_us("MM4_Qn"),
        depends_on=["DequantRopeUB"])

    # --- AIV pipeline (unchanged up through ScatterCkvKr) ---
    stages["CopyInSinCos"] = PipelineStage("CopyInSinCos", "AIV", 0.5)
    stages["aiv_wait_CQ"] = PipelineStage(
        "aiv_wait_CQ", "AIV", sync_us,
        depends_on=["CopyInSinCos", "sync_signal_CQ"])

    has_accum_cq = "AccumCq" in stage_timings
    if has_accum_cq:
        stages["AccumCq"] = PipelineStage(
            "AccumCq", "AIV", get_us("AccumCq"), depends_on=["aiv_wait_CQ"])
        rmsnorm_cq_dep = "AccumCq"
    else:
        rmsnorm_cq_dep = "aiv_wait_CQ"

    stages["RmsNormCq"] = PipelineStage(
        "RmsNormCq", "AIV", get_us("RmsNormCq"), depends_on=[rmsnorm_cq_dep])
    stages["sync_signal_RMSNORM_CQ"] = PipelineStage(
        "sync_signal_RMSNORM_CQ", "AIV", sync_us, depends_on=["RmsNormCq"])
    stages["aiv_wait_CKVKR"] = PipelineStage(
        "aiv_wait_CKVKR", "AIV", sync_us,
        depends_on=["sync_signal_RMSNORM_CQ", "sync_signal_CKVKR"])

    has_accum_ckvkr = "AccumCkvKr" in stage_timings
    if has_accum_ckvkr:
        stages["AccumCkvKr"] = PipelineStage(
            "AccumCkvKr", "AIV", get_us("AccumCkvKr"),
            depends_on=["aiv_wait_CKVKR"])
        rmsnorm_ckvkr_dep = "AccumCkvKr"
    else:
        rmsnorm_ckvkr_dep = "aiv_wait_CKVKR"

    stages["RmsNormCkvKr"] = PipelineStage(
        "RmsNormCkvKr", "AIV", get_us("RmsNormCkvKr"),
        depends_on=[rmsnorm_ckvkr_dep])
    stages["RopeKr"] = PipelineStage(
        "RopeKr", "AIV", get_us("RopeKr"), depends_on=["RmsNormCkvKr"])
    stages["ScatterCkvKr"] = PipelineStage(
        "ScatterCkvKr", "AIV", get_us("ScatterCkvKr"), depends_on=["RopeKr"])

    # --- DynamicQuantQn / MulQr (if enabled) — still post-MM4 on AIV ---
    if has_dynamic_quant:
        stages["DynamicQuantQn"] = PipelineStage(
            "DynamicQuantQn", "AIV", get_us("DynamicQuantQn"),
            depends_on=["ScatterCkvKr", "MM4_Qn"])
        stages["MulQr"] = PipelineStage(
            "MulQr", "AIV", get_us("MulQr"),
            depends_on=["DynamicQuantQn"])

    return _solve_dag(stages)


def build_pipeline_dag_l0c_ub(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    stage_timings: Optional[Dict] = None,
) -> PipelineResult:
    """Build pipeline DAG with L0C→UB core-internal sync (HardEvent ≈ 0µs).

    Differences from `build_pipeline_dag`:
      - No CrossCore sync stages (HardEvent replaces CrossCoreSetFlag)
      - UB→L1 transfer stages between vector ops and next matmul
      - Direct dependency edges with 0µs sync overhead
      - Scatter still goes to HBM (MTE3), but other vector ops are UB-internal

    Pipeline structure:
      AIC: MM1 → MM2 → [wait UB→L1_Cq] → MM3 → [wait UB→L1_Qc] → MM4
      AIV: [wait MM1] → RmsNormCq → UB→L1_Cq → [signal]
           [wait MM2] → RmsNormCkvKr → RopeKr → Scatter(HBM)
           [wait MM3] → [DequantQc] → RopeQr → UB→L1_Qc → [signal]
    """
    if stage_timings is None:
        from ..model.orchestration import estimate_all_stages_l0c_ub
        stage_timings = estimate_all_stages_l0c_ub(params, tiling, hw)

    qm = params.quant_mode
    has_dequant = is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm)
    has_dynamic_quant = params.query_quant_mode == 1

    def get_us(name: str) -> float:
        t = stage_timings.get(name)
        return t.total_us if t is not None else 0.0

    stages: Dict[str, PipelineStage] = {}

    # --- AIC pipeline (no CrossCore sync, direct edges) ---
    stages["MM1_Cq"] = PipelineStage("MM1_Cq", "AIC", get_us("MM1_Cq"))

    stages["MM2_CkvKr"] = PipelineStage(
        "MM2_CkvKr", "AIC", get_us("MM2_CkvKr"),
        depends_on=["MM1_Cq"])

    # AIC waits for UB→L1_Cq (RmsNormCq result) before MM3
    stages["MM3_QcQr"] = PipelineStage(
        "MM3_QcQr", "AIC", get_us("MM3_QcQr"),
        depends_on=["MM2_CkvKr", "UB_to_L1_Cq"])

    # AIC waits for UB→L1_Qc (DequantQc/RopeQr result) before MM4
    if has_dequant:
        stages["MM4_Qn"] = PipelineStage(
            "MM4_Qn", "AIC", get_us("MM4_Qn"),
            depends_on=["MM3_QcQr", "UB_to_L1_Qc"])
    else:
        stages["MM4_Qn"] = PipelineStage(
            "MM4_Qn", "AIC", get_us("MM4_Qn"),
            depends_on=["MM3_QcQr", "RopeQr"])

    # --- AIV pipeline (HardEvent sync ≈ 0µs) ---
    stages["RmsNormCq"] = PipelineStage(
        "RmsNormCq", "AIV", get_us("RmsNormCq"),
        depends_on=["MM1_Cq"])

    stages["UB_to_L1_Cq"] = PipelineStage(
        "UB_to_L1_Cq", "AIV", get_us("UB_to_L1_Cq"),
        depends_on=["RmsNormCq"])

    # RmsNormCkvKr + RopeKr + Scatter run in parallel with MM3 path
    stages["RmsNormCkvKr"] = PipelineStage(
        "RmsNormCkvKr", "AIV", get_us("RmsNormCkvKr"),
        depends_on=["MM2_CkvKr"])

    stages["RopeKr"] = PipelineStage(
        "RopeKr", "AIV", get_us("RopeKr"),
        depends_on=["RmsNormCkvKr"])

    stages["ScatterCkvKr"] = PipelineStage(
        "ScatterCkvKr", "AIV", get_us("ScatterCkvKr"),
        depends_on=["RopeKr"])

    # After MM3: DequantQc (if quantized) + RopeQr
    if has_dequant:
        stages["DequantQc"] = PipelineStage(
            "DequantQc", "AIV", get_us("DequantQc"),
            depends_on=["ScatterCkvKr", "MM3_QcQr"])

        stages["RopeQr"] = PipelineStage(
            "RopeQr", "AIV", get_us("RopeQr"),
            depends_on=["DequantQc"])

        stages["UB_to_L1_Qc"] = PipelineStage(
            "UB_to_L1_Qc", "AIV", get_us("UB_to_L1_Qc"),
            depends_on=["RopeQr"])
    else:
        stages["RopeQr"] = PipelineStage(
            "RopeQr", "AIV", get_us("RopeQr"),
            depends_on=["ScatterCkvKr", "MM3_QcQr"])

    # DynamicQuantQn / MulQr (post-MM4)
    if has_dynamic_quant:
        stages["DynamicQuantQn"] = PipelineStage(
            "DynamicQuantQn", "AIV", get_us("DynamicQuantQn"),
            depends_on=["ScatterCkvKr", "MM4_Qn"])
        stages["MulQr"] = PipelineStage(
            "MulQr", "AIV", get_us("MulQr"),
            depends_on=["DynamicQuantQn"])

    return _solve_dag(stages)
