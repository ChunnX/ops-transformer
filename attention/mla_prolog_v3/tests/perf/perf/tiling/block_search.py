#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-matmul block size search.

Exhaustive search over (baseM, baseN, baseK) candidates with stepK derived
from the L1 capacity constraint. Used by the `block-search` analysis mode.
"""

import math
from typing import Dict, List, Tuple

from ..model.block_spec import MatmulBlockSpec, check_cache_fit, derive_stepK
from ..model.hardware import AscendHWSpec
from ..model.matmul_detailed import estimate_matmul_detailed
from ..model.operator_params import OperatorParams
from ..model.quant import is_full_quant, is_int8_quant, is_mxfp8
from ..model.timing_types import MatMulTiming
from .config import TilingConfig

SEARCH_BASE_M = [16, 32, 64, 128, 192, 256, 512]
SEARCH_BASE_N = [64, 128, 192, 256, 512]
SEARCH_BASE_K = [64, 128, 256, 512]


def search_matmul_block_sizes(
    name: str,
    M: int, K: int, N_total: int,
    active_cores: int,
    hw: AscendHWSpec,
    dtype_a_size: int = 2,
    dtype_b_size: int = 2,
    dtype_c_size: int = 2,
    a_reused: bool = False,
    mode: str = "split_k",
) -> List[Tuple[MatmulBlockSpec, MatMulTiming]]:
    """Exhaustive search over (baseM, baseN, baseK) with stepK derived from L1.

    Returns list of (block_spec, timing) sorted by total_us ascending.
    """
    if not hw.has_cycle_spec:
        return []

    C = max(active_cores, 1)
    per_core_N = math.ceil(N_total / C)
    results = []

    if mode == "full_load":
        # For full_load mode, only search baseN (nSplitSize)
        for baseN in SEARCH_BASE_N:
            if baseN > per_core_N:
                baseN = per_core_N
            block = MatmulBlockSpec(
                name, baseM=M, baseN=baseN, baseK=K, stepK=1,
                mode="full_load",
                dtype_a_size=dtype_a_size, dtype_b_size=dtype_b_size,
                dtype_c_size=dtype_c_size,
            )
            if not check_cache_fit(block, hw, per_core_N, M=M, K=K):
                continue
            timing = estimate_matmul_detailed(
                name, M, K, N_total, active_cores, hw, block, a_reused)
            results.append((block, timing))
        results.sort(key=lambda x: x[1].total_us)
        return results

    # Build baseM candidates: always include actual M (aligned to 16 for cube)
    baseM_candidates = sorted(set(SEARCH_BASE_M + [M]))
    # Build baseN candidates: always include per_core_N if smaller than max
    baseN_candidates = sorted(set(SEARCH_BASE_N + [per_core_N]))

    for baseM in baseM_candidates:
        if baseM > M:
            continue
        for baseN in baseN_candidates:
            if baseN > per_core_N:
                continue
            for baseK in SEARCH_BASE_K:
                if baseK > K:
                    continue
                block = MatmulBlockSpec(
                    name, baseM=baseM, baseN=baseN, baseK=baseK, stepK=1,
                    mode="split_k",
                    dtype_a_size=dtype_a_size, dtype_b_size=dtype_b_size,
                    dtype_c_size=dtype_c_size,
                )
                # Check cache constraints (with actual M, K for buffer factor)
                if not check_cache_fit(block, hw, per_core_N, M=M, K=K):
                    continue
                # Derive stepK from L1
                stepK = derive_stepK(block, K, hw, per_core_N, M=M)
                if stepK < 1:
                    continue
                block.stepK = stepK

                timing = estimate_matmul_detailed(
                    name, M, K, N_total, active_cores, hw, block, a_reused)
                results.append((block, timing))

    results.sort(key=lambda x: x[1].total_us)
    return results


def search_all_matmul_blocks(
    params: OperatorParams,
    tiling: TilingConfig,
    hw: AscendHWSpec,
) -> Dict[str, List[Tuple[MatmulBlockSpec, MatMulTiming]]]:
    """Search optimal block sizes for all four matmuls.

    Returns dict mapping MM name -> sorted list of (block_spec, timing).
    """
    qm = params.quant_mode
    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    is_quant = is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm)
    mm3_wt_size = 1 if is_quant else 2
    T = tiling.step_batch_size

    results = {}

    # MM1: tokenX[T, He] x weightDq[He, Hcq]
    results["MM1_Cq"] = search_matmul_block_sizes(
        "MM1_Cq", T, params.He, params.Hcq,
        active_cores=tiling.mm1_block_num, hw=hw,
        dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
        a_reused=False,
    )

    # MM2: tokenX[T, He] x weightDkvKr[He, Hckv+Dr] (A reused from MM1)
    results["MM2_CkvKr"] = search_matmul_block_sizes(
        "MM2_CkvKr", T, params.He, params.Hckv + params.Dr,
        active_cores=tiling.mm2_block_num, hw=hw,
        dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
        a_reused=True,
    )

    # MM3: Cq[T, Hcq] x weightUqQr[Hcq, N*(D+Dr)]
    results["MM3_QcQr"] = search_matmul_block_sizes(
        "MM3_QcQr", T, params.Hcq, params.N * (params.D + params.Dr),
        active_cores=tiling.mm3_block_num, hw=hw,
        dtype_a_size=2, dtype_b_size=mm3_wt_size, dtype_c_size=out_size,
        a_reused=False,
    )

    # MM4: Qc[T, D] x Uk[D, Hckv] (full_load mode)
    results["MM4_Qn"] = search_matmul_block_sizes(
        "MM4_Qn", T, params.D, params.Hckv,
        active_cores=tiling.mm4_block_num, hw=hw,
        dtype_a_size=2, dtype_b_size=wt_size, dtype_c_size=2,
        a_reused=False, mode="full_load",
    )

    return results
