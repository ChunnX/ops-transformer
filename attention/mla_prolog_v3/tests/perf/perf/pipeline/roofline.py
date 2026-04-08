#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roofline model analysis for the four MLA Prolog matmuls.

For each matmul, computes arithmetic intensity, peak vs achieved GFLOPS,
and ridge point relative to per-core HBM bandwidth.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ..model.block_spec import MatmulBlockSpec
from ..model.hardware import AscendHWSpec
from ..model.matmul_detailed import estimate_matmul_detailed
from ..model.operator_params import OperatorParams
from ..model.orchestration import get_default_block_specs


@dataclass
class RooflineResult:
    """Roofline analysis result for a single matmul."""
    name: str
    block: MatmulBlockSpec
    M: int
    K: int
    N: int
    active_cores: int
    # Time components (us)
    cube_us: float = 0.0
    mte2_l0_us: float = 0.0      # max(L0A, L0B) load time
    fixpipe_us: float = 0.0
    hbm_l1_us: float = 0.0       # HBM/L2 -> L1 time
    total_us: float = 0.0
    # Derived metrics
    arithmetic_intensity: float = 0.0  # FLOPs / bytes_loaded
    peak_gflops: float = 0.0
    achieved_gflops: float = 0.0
    efficiency: float = 0.0       # achieved / peak
    bound: str = ""


def roofline_analysis(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    block_specs: Optional[Dict[str, MatmulBlockSpec]] = None,
) -> Dict[str, RooflineResult]:
    """Compute roofline analysis for each matmul.

    Requires per-cycle HW spec (hw.has_cycle_spec == True).
    """
    if not hw.has_cycle_spec:
        return {}

    if block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    T = tiling.step_batch_size
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N

    mm_configs = {
        "MM1_Cq": (T, He, Hcq, tiling.mm1_block_num, False),
        "MM2_CkvKr": (T, He, Hckv + Dr, tiling.mm2_block_num, True),
        "MM3_QcQr": (T, Hcq, N * (D + Dr), tiling.mm3_block_num, False),
        "MM4_Qn": (T, D, Hckv, tiling.mm4_block_num, False),
    }

    results = {}
    for mm_name, (M, K, N_total, cores, a_reused) in mm_configs.items():
        block = block_specs[mm_name]
        C = max(cores, 1)
        per_core_N = math.ceil(N_total / C)

        # Get detailed timing
        timing = estimate_matmul_detailed(
            mm_name, M, K, N_total, cores, hw, block, a_reused)

        # Arithmetic intensity: FLOPs / bytes loaded from HBM
        total_flops = 2.0 * M * per_core_N * K
        # Bytes loaded per core from HBM/L2 to L1
        dtype_a = block.dtype_a_size
        dtype_b = block.dtype_b_size
        dtype_c = block.dtype_c_size
        bytes_A = M * K * dtype_a
        bytes_B = K * per_core_N * dtype_b
        bytes_C = M * per_core_N * dtype_c
        total_bytes = bytes_A + bytes_B + bytes_C
        ai = total_flops / total_bytes if total_bytes > 0 else 0

        # Peak compute (per core)
        if dtype_a == 1:
            peak_ops_per_sec = hw.mmad_ops_per_cycle_fp8 * hw.freq_hz * 2.0
        else:
            peak_ops_per_sec = hw.mmad_ops_per_cycle_fp16 * hw.freq_hz * 2.0
        peak_gflops = peak_ops_per_sec / 1e9

        # Achieved
        actual_time_s = timing.total_us / 1e6
        achieved_gflops = (total_flops / actual_time_s / 1e9) if actual_time_s > 0 else 0
        efficiency = achieved_gflops / peak_gflops if peak_gflops > 0 else 0

        results[mm_name] = RooflineResult(
            name=mm_name,
            block=block,
            M=M, K=K, N=N_total,
            active_cores=C,
            cube_us=timing.cube_us,
            mte2_l0_us=max(timing.l0a_us, timing.l0b_us) * block.stepK * timing.kL1_loops
                if block.mode == "split_k" else max(timing.l0a_us, timing.l0b_us) * timing.kL1_loops,
            fixpipe_us=timing.fixpipe_us,
            hbm_l1_us=timing.mte1_outer_us * timing.kL1_loops
                if block.mode == "split_k" else timing.mte1_outer_us,
            total_us=timing.total_us,
            arithmetic_intensity=ai,
            peak_gflops=peak_gflops,
            achieved_gflops=achieved_gflops,
            efficiency=efficiency,
            bound=timing.bound,
        )

    return results


def format_roofline_report(
    results: Dict[str, RooflineResult],
    hw: AscendHWSpec,
) -> str:
    """Format roofline analysis as a table."""
    lines = []
    lines.append(f"Roofline Analysis ({hw.name}, freq={hw.freq_ghz}GHz)")
    lines.append("")

    header = (
        f"{'Matmul':<12} {'Block(M*N*K)':<16} {'stepK':>5} {'Cores':>5} | "
        f"{'Cube':>8} {'MTE2_L0':>8} {'FixPipe':>8} {'HBM>L1':>8} {'Total':>8} | "
        f"{'AI':>6} {'Peak':>8} {'Achv':>8} {'Eff':>6} {'Bound':<8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for mm_name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        if mm_name not in results:
            continue
        r = results[mm_name]
        b = r.block
        block_str = f"{b.baseM}x{b.baseN}x{b.baseK}"
        lines.append(
            f"{r.name:<12} {block_str:<16} {b.stepK:>5} {r.active_cores:>5} | "
            f"{r.cube_us:>7.2f} {r.mte2_l0_us:>7.2f} {r.fixpipe_us:>7.2f} {r.hbm_l1_us:>7.2f} {r.total_us:>7.2f} | "
            f"{r.arithmetic_intensity:>5.1f} {r.peak_gflops:>7.1f} {r.achieved_gflops:>7.1f} {r.efficiency:>5.1%} {r.bound:<8}"
        )

    lines.append("")

    # Summary: ridge point
    if hw.has_cycle_spec:
        hbm_bw_per_core = hw.per_core_bw("hbm", hw.aic_num)
        fp16_peak = hw.mmad_ops_per_cycle_fp16 * hw.freq_hz * 2.0
        fp8_peak = hw.mmad_ops_per_cycle_fp8 * hw.freq_hz * 2.0
        ridge_fp16 = fp16_peak / hbm_bw_per_core if hbm_bw_per_core > 0 else 0
        ridge_fp8 = fp8_peak / hbm_bw_per_core if hbm_bw_per_core > 0 else 0
        lines.append(f"Ridge point (HBM, {hw.aic_num} cores): "
                     f"BF16={ridge_fp16:.1f} FLOPs/B, FP8={ridge_fp8:.1f} FLOPs/B")
        lines.append(f"Per-core HBM BW: {hbm_bw_per_core/1e9:.1f} GB/s, "
                     f"L2 BW: {hw.per_core_bw('l2', hw.aic_num)/1e9:.1f} GB/s")

    return "\n".join(lines)
