#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`bound` mode: per-stage resource breakdown table."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...model.orchestration import estimate_all_stages
from ...tiling.config import TilingConfig
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 1: Per-stage resource analysis table."""
    print_header("BOUND ANALYSIS — Per-Stage Resource Breakdown")

    stages = estimate_all_stages(params, tiling, hw)

    print(f"Parameters: T={params.T}, He={params.He}, N={params.N}, Hcq={params.Hcq}, Hckv={params.Hckv}, D={params.D}, Dr={params.Dr}")
    print(f"Quant: weight={params.weight_quant_mode}, kv_cache={params.kv_cache_quant_mode}, query={params.query_quant_mode}")
    print(f"Tiling: stepBatch={tiling.step_batch_size}")
    print()

    # MatMul stages - extended columns for detailed mode
    if hw.has_cycle_spec:
        mm_header = (
            f"{'Stage':<14} {'M':>4} {'K':>6} {'N':>7} {'C':>3} | "
            f"{'MTE1':>7} {'L0A':>7} {'L0B':>7} {'MMAD':>7} {'FxPipe':>7} {'Total':>7} | "
            f"{'kL1':>3} {'Bound':<8} {'A src':<4}"
        )
    else:
        mm_header = f"{'Stage':<16} {'M':>5} {'K':>6} {'N':>7} {'Cores':>5} | {'MTE1':>8} {'Cube':>8} {'FixPipe':>8} {'Total':>8} | {'Bound':<8} {'A src':<5}"
    print(mm_header)
    print("-" * len(mm_header))

    for name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        t = stages[name]
        if hw.has_cycle_spec:
            print(f"{t.name:<14} {t.M:>4} {t.K:>6} {t.N:>7} {t.active_cores:>3} | "
                  f"{t.mte1_outer_us:>6.2f} {t.l0a_us:>6.2f} {t.l0b_us:>6.2f} {t.mmad_us:>6.2f} {t.fixpipe_us:>6.2f} {t.total_us:>6.2f} | "
                  f"{t.kL1_loops:>3} {t.bound:<8} {t.a_source:<4}")
        else:
            print(f"{t.name:<16} {t.M:>5} {t.K:>6} {t.N:>7} {t.active_cores:>5} | "
                  f"{t.mte1_us:>7.2f} {t.cube_us:>7.2f} {t.fixpipe_us:>7.2f} {t.total_us:>7.2f} | "
                  f"{t.bound:<8} {t.a_source:<5}")

    print()

    # Vector stages
    vec_header = f"{'Stage':<16} | {'MTE2':>8} {'Vector':>8} {'MTE3':>8} {'Total':>8} | {'Bound':<8}"
    print(vec_header)
    print("-" * len(vec_header))

    vec_names = ["RmsNormCq", "RmsNormCkvKr", "RopeKr", "ScatterCkvKr",
                 "DequantQc", "RopeQr", "DynamicQuantQn", "MulQr"]
    for name in vec_names:
        if name not in stages:
            continue
        t = stages[name]
        print(f"{t.name:<16} | "
              f"{t.mte2_load_us:>7.2f} {t.vec_compute_us:>7.2f} {t.mte3_store_us:>7.2f} {t.total_us:>7.2f} | "
              f"{t.bound:<8}")
