#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`advice` mode: automated bottleneck detection and improvement suggestions."""

from ...constants import (
    AIV_AIC_BALANCE_HIGH, AIV_AIC_BALANCE_LOW, SYNC_OVERHEAD_THRESHOLD,
)
from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import estimate_all_stages
from ...pipeline.dag import build_pipeline_dag
from ...tiling.config import TilingConfig
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 4: Improvement suggestions."""
    print_header("PERFORMANCE ADVICE")

    stages = estimate_all_stages(params, tiling, hw)
    result = build_pipeline_dag(params, tiling, hw, stages)
    advice = []

    # Check each matmul for bottleneck
    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        t = stages[name]
        if t.bound in ("HBM", "L2"):
            if params.weight_quant_mode == 0:
                advice.append(
                    f"[{name}] MTE1-bound ({t.bound}): consider INT8/FP8 quantization to halve weight load "
                    f"({t.mte1_us:.2f} us MTE1 vs {t.cube_us:.2f} us Cube)")
            else:
                advice.append(
                    f"[{name}] MTE1-bound ({t.bound}): already quantized. Consider restructuring tiling to "
                    f"improve L2 reuse.")

    # Check core utilization
    for name, block_num_attr in [("MM1", "mm1_block_num"), ("MM2", "mm2_block_num"),
                                  ("MM3", "mm3_block_num"), ("MM4", "mm4_block_num")]:
        cores = getattr(tiling, block_num_attr)
        if cores < hw.aic_num * 0.5:
            advice.append(
                f"[{name}] Low core utilization: {cores}/{hw.aic_num} AIC cores active. "
                f"Consider rebalancing N-split to use more cores.")

    # Check AIC/AIV balance
    if result.aiv_busy_us > 0 and result.aic_busy_us > 0:
        ratio = result.aiv_busy_us / result.aic_busy_us
        if ratio < AIV_AIC_BALANCE_LOW:
            advice.append(
                f"AIV largely idle ({result.aiv_busy_us:.2f} vs AIC {result.aic_busy_us:.2f} us). "
                f"Vector pipeline is underutilized — consider fusing more post-MM vector ops.")
        elif ratio > AIV_AIC_BALANCE_HIGH:
            advice.append(
                f"AIC largely idle ({result.aic_busy_us:.2f} vs AIV {result.aiv_busy_us:.2f} us). "
                f"Cube pipeline is underutilized — potential for more matmul overlap.")

    # Check sync overhead
    total_us = result.total_us
    if total_us > 0 and result.sync_overhead_us / total_us > SYNC_OVERHEAD_THRESHOLD:
        advice.append(
            f"Sync overhead is {result.sync_overhead_us:.2f} us ({result.sync_overhead_us/total_us*100:.1f}% of total). "
            f"Consider reducing cross-core synchronization events.")

    # Check fixpipe bound
    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        t = stages[name]
        if t.bound == "FIXPIPE":
            advice.append(
                f"[{name}] FixPipe-bound: store bandwidth is the bottleneck. "
                f"Consider fusing with downstream vector ops to avoid GM round-trip.")

    # Check scatter efficiency
    scatter = stages.get("ScatterCkvKr")
    if scatter and scatter.bound == "MTE3":
        advice.append(
            f"[ScatterCkvKr] Scatter store is the bottleneck ({scatter.mte3_store_us:.2f} us). "
            f"Indexed writes have {hw.scatter_efficiency:.0%} efficiency. "
            f"Consider contiguous cache layout if possible.")

    if not advice:
        advice.append("No obvious bottlenecks detected. Performance appears well-balanced.")

    for i, a in enumerate(advice, 1):
        print(f"  {i}. {a}")
        print()
