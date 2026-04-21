#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestration: per-stage timing for MM1-4 plus all vector operations.

Provides three top-level entry points:
- estimate_all_stages: standard split-N path used by most analysis modes
- estimate_all_stages_split_m: M-axis split (prefill optimization)
- estimate_all_stages_k_outer: K-outer loop (cube-bound prefill)

All three return a dict mapping stage name -> MatMulTiming or VectorOpTiming.
"""

import math

from .block_spec import MatmulBlockSpec, derive_stepK
from .hardware import AscendHWSpec
from .matmul_detailed import estimate_matmul_detailed
from .matmul_k_outer import estimate_matmul_k_outer
from .matmul_legacy import estimate_matmul
from .matmul_split_kn import (
    SplitKNSpec, estimate_matmul_split_kn, estimate_mm2_split_k_cross_core,
)
from .matmul_split_m import SplitMSpec, estimate_matmul_split_m, estimate_matmul_split_m_l0c_ub
from .operator_params import OperatorParams
from .quant import is_full_quant, is_int8_quant, is_mxfp8
from .timing_types import VectorOpTiming
from .vector_ops import estimate_vector_op, estimate_vector_op_l0c_ub, vector_instr_count


def get_default_block_specs(params: 'OperatorParams', hw: AscendHWSpec):
    """Return default MatmulBlockSpec for each matmul based on quant mode."""
    qm = params.quant_mode
    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    is_fp8 = is_mxfp8(qm)
    is_quant = is_full_quant(qm) or is_int8_quant(qm) or is_fp8

    # MM3 weight dtype
    mm3_wt_size = 1 if is_quant else 2
    mm3_act_size = 2  # Cq is always BF16 after RMSNorm

    if is_fp8 or is_quant:
        mm1_baseK = 256
        mm2_baseK = 256
        mm3_baseK = 128
    else:
        mm1_baseK = 128
        mm2_baseK = 128
        mm3_baseK = 64

    mm1_baseN = 64 if is_fp8 else 128
    mm2_baseN = 128
    mm3_baseN = 128 if (is_fp8 or is_quant) else 256

    specs = {}
    specs["MM1_Cq"] = MatmulBlockSpec(
        "MM1_Cq", baseM=32, baseN=mm1_baseN, baseK=mm1_baseK, stepK=4,
        mode="split_k", dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
    )
    specs["MM2_CkvKr"] = MatmulBlockSpec(
        "MM2_CkvKr", baseM=32, baseN=mm2_baseN, baseK=mm2_baseK, stepK=4,
        mode="split_k", dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
    )
    specs["MM3_QcQr"] = MatmulBlockSpec(
        "MM3_QcQr", baseM=32, baseN=mm3_baseN, baseK=mm3_baseK, stepK=4,
        mode="split_k", dtype_a_size=mm3_act_size, dtype_b_size=mm3_wt_size, dtype_c_size=out_size,
    )
    specs["MM4_Qn"] = MatmulBlockSpec(
        "MM4_Qn", baseM=32, baseN=128, baseK=128, stepK=1,
        mode="full_load", dtype_a_size=2, dtype_b_size=wt_size, dtype_c_size=2,
    )

    # Derive stepK from L1 constraint for each spec
    mm_dims = {
        "MM1_Cq": (params.He, params.Hcq),
        "MM2_CkvKr": (params.He, params.Hckv + params.Dr),
        "MM3_QcQr": (params.Hcq, params.N * (params.D + params.Dr)),
        "MM4_Qn": (params.D, params.Hckv),
    }
    T = params.T
    for mm_name, (K, N_total) in mm_dims.items():
        spec = specs[mm_name]
        if spec.mode == "full_load":
            continue
        n_per_core = math.ceil(N_total / hw.aic_num)
        derived = derive_stepK(spec, K, hw, n_per_core, M=T)
        if derived > 0:
            spec.stepK = derived

    return specs


def estimate_all_stages(params: OperatorParams, tiling, hw: AscendHWSpec,
                        block_specs=None, mm2_split_k_cores: int = 0,
                        mm1_split_kn: 'SplitKNSpec' = None,
                        mm2_split_kn: 'SplitKNSpec' = None):
    """Estimate timing for all matmul and vector stages.

    When hw.has_cycle_spec is True, uses the detailed inner-loop model.
    Otherwise falls back to the chip-total estimate_matmul().

    Args:
        mm2_split_k_cores: If > 0, use pure K-split for MM2 (legacy).
        mm1_split_kn: If set, use 2D N×K split for MM1.
        mm2_split_kn: If set, use 2D N×K split for MM2 (overrides mm2_split_k_cores).
    """
    T = tiling.step_batch_size
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    qm = params.quant_mode

    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    use_detailed = hw.has_cycle_spec
    if use_detailed and block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    stages = {}

    if use_detailed:
        # MM1: tokenX[T, He] x weightDq[He, Hcq] -> Cq[T, Hcq]
        if mm1_split_kn and hw.has_cycle_spec:
            cube_t, accum_t = estimate_matmul_split_kn(
                "MM1_Cq", T, He, Hcq, mm1_split_kn,
                hw=hw, block=block_specs["MM1_Cq"], a_reused=False,
                vec_cores=tiling.vector_block_num, out_size=out_size,
            )
            stages["MM1_Cq"] = cube_t
            if accum_t.total_us > 0:
                stages["AccumCq"] = accum_t
        else:
            stages["MM1_Cq"] = estimate_matmul_detailed(
                "MM1_Cq", T, He, Hcq,
                active_cores=tiling.mm1_block_num,
                hw=hw, block=block_specs["MM1_Cq"], a_reused=False,
                a_shared_cores=tiling.mm1_block_num,
            )
        # MM2: tokenX reused from MM1 (L2 warm)
        if mm2_split_kn and hw.has_cycle_spec:
            cube_t, accum_t = estimate_matmul_split_kn(
                "MM2_CkvKr", T, He, Hckv + Dr, mm2_split_kn,
                hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
                vec_cores=tiling.vector_block_num, out_size=out_size,
            )
            stages["MM2_CkvKr"] = cube_t
            if accum_t.total_us > 0:
                stages["AccumCkvKr"] = accum_t
        elif mm2_split_k_cores > 0 and hw.has_cycle_spec:
            cube_t, accum_t = estimate_mm2_split_k_cross_core(
                T, He, Hckv + Dr, mm2_split_k_cores,
                hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
                vec_cores=tiling.vector_block_num, out_size=out_size,
            )
            stages["MM2_CkvKr"] = cube_t
            stages["AccumCkvKr"] = accum_t
        else:
            stages["MM2_CkvKr"] = estimate_matmul_detailed(
                "MM2_CkvKr", T, He, Hckv + Dr,
                active_cores=tiling.mm2_block_num,
                hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
            )
        # MM3: Cq[T, Hcq] x weightUqQr[Hcq, N*(D+Dr)] -> QcQr
        stages["MM3_QcQr"] = estimate_matmul_detailed(
            "MM3_QcQr", T, Hcq, N * (D + Dr),
            active_cores=tiling.mm3_block_num,
            hw=hw, block=block_specs["MM3_QcQr"], a_reused=False,
            a_shared_cores=tiling.mm3_block_num,
        )
        # MM4: Qc[T, D] x Uk[D, Hckv] -> Qn
        stages["MM4_Qn"] = estimate_matmul_detailed(
            "MM4_Qn", T, D, Hckv,
            active_cores=tiling.mm4_block_num,
            hw=hw, block=block_specs["MM4_Qn"], a_reused=False,
            a_shared_cores=tiling.mm4_block_num,
        )
    else:
        # Legacy chip-total path
        stages["MM1_Cq"] = estimate_matmul(
            "MM1_Cq", T, He, Hcq,
            active_cores=tiling.mm1_block_num, hw=hw,
            dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
            a_reused=False,
        )
        stages["MM2_CkvKr"] = estimate_matmul(
            "MM2_CkvKr", T, He, Hckv + Dr,
            active_cores=tiling.mm2_block_num, hw=hw,
            dtype_a_size=act_size, dtype_b_size=wt_size, dtype_c_size=out_size,
            a_reused=True,
        )
        mm3_wt_size = 1 if (is_full_quant(qm) or is_mxfp8(qm) or is_int8_quant(qm)) else 2
        stages["MM3_QcQr"] = estimate_matmul(
            "MM3_QcQr", T, Hcq, N * (D + Dr),
            active_cores=tiling.mm3_block_num, hw=hw,
            dtype_a_size=2, dtype_b_size=mm3_wt_size, dtype_c_size=out_size,
            a_reused=False,
        )
        stages["MM4_Qn"] = estimate_matmul(
            "MM4_Qn", T, D, Hckv,
            active_cores=tiling.mm4_block_num, hw=hw,
            dtype_a_size=2, dtype_b_size=wt_size, dtype_c_size=2,
            a_reused=False,
        )

    vec_cores = tiling.vector_block_num

    # RMSNorm Cq
    rmsnorm_cq_load = T * Hcq * out_size
    rmsnorm_cq_store = T * Hcq * 2
    stages["RmsNormCq"] = estimate_vector_op(
        "RmsNormCq", rmsnorm_cq_load, T * Hcq, rmsnorm_cq_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCq", qm),
    )

    # RMSNorm CkvKr — kernel only loads the Hckv portion (Dr is read separately by RopeKr)
    rmsnorm_ckvkr_load = T * Hckv * out_size
    rmsnorm_ckvkr_store = T * Hckv * 2
    stages["RmsNormCkvKr"] = estimate_vector_op(
        "RmsNormCkvKr", rmsnorm_ckvkr_load, T * Hckv, rmsnorm_ckvkr_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCkvKr", qm),
    )

    # RoPE Kr
    rope_kr_load = T * Dr * 2 + T * Dr * 2
    rope_kr_store = T * Dr * 2
    stages["RopeKr"] = estimate_vector_op(
        "RopeKr", rope_kr_load, T * Dr, rope_kr_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeKr", qm),
    )

    # Scatter CkvKr
    scatter_store = T * (Hckv + Dr) * 2
    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", T * (Hckv + Dr) * 2, T * (Hckv + Dr), scatter_store,
        hw, active_cores=vec_cores, is_scatter=True,
        n_instructions=vector_instr_count("ScatterCkvKr", qm),
    )

    # DequantQc
    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        dequant_load = T * N * D * out_size
        dequant_store = T * N * D * 2
        stages["DequantQc"] = estimate_vector_op(
            "DequantQc", dequant_load, T * N * D, dequant_store,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DequantQc", qm),
        )

    # RoPE Qr
    rope_qr_load = T * N * Dr * 2 + T * Dr * 2
    rope_qr_store = T * N * Dr * 2
    stages["RopeQr"] = estimate_vector_op(
        "RopeQr", rope_qr_load, T * N * Dr, rope_qr_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeQr", qm),
    )

    # DynamicQuant Qn
    if params.query_quant_mode == 1:
        dyn_quant_load = T * N * Hckv * 2
        dyn_quant_store = T * N * Hckv * 1
        stages["DynamicQuantQn"] = estimate_vector_op(
            "DynamicQuantQn", dyn_quant_load, T * N * Hckv, dyn_quant_store,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DynamicQuantQn", qm),
        )

    # MulQr
    if params.query_quant_mode == 1:
        mul_qr_load = T * N * Dr * 2
        mul_qr_store = T * N * Dr * 2
        stages["MulQr"] = estimate_vector_op(
            "MulQr", mul_qr_load, T * N * Dr, mul_qr_store,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("MulQr", qm),
        )

    return stages


def estimate_all_stages_k_outer(
    params,
    tiling,
    hw: AscendHWSpec,
    block_specs=None,
    mm1_split_kn: 'SplitKNSpec' = None,
    mm2_split_kn: 'SplitKNSpec' = None,
):
    """Estimate all stages with K-outer loop for prefill.

    Uses estimate_matmul_k_outer for each MM (or split-KN for MM1/MM2).
    Vector ops use full T tokens (single launch processes all tokens).
    """
    T = params.T  # All tokens in single launch
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    qm = params.quant_mode

    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    if block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    stages = {}

    # MM1: tokenX[T, He] x weightDq[He, Hcq] -> Cq
    if mm1_split_kn:
        cube_t, accum_t = estimate_matmul_split_kn(
            "MM1_Cq", T, He, Hcq, mm1_split_kn,
            hw=hw, block=block_specs["MM1_Cq"], a_reused=False,
            vec_cores=hw.aiv_num, out_size=out_size,
        )
        stages["MM1_Cq"] = cube_t
        if accum_t.total_us > 0:
            stages["AccumCq"] = accum_t
    else:
        stages["MM1_Cq"] = estimate_matmul_k_outer(
            "MM1_Cq", T, He, Hcq,
            active_cores=hw.aic_num, hw=hw,
            block=block_specs["MM1_Cq"], a_reused=False,
        )

    # MM2: tokenX[T, He] x weightCkvKr[He, Hckv+Dr] (A reused from MM1)
    if mm2_split_kn:
        cube_t, accum_t = estimate_matmul_split_kn(
            "MM2_CkvKr", T, He, Hckv + Dr, mm2_split_kn,
            hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
            vec_cores=hw.aiv_num, out_size=out_size,
        )
        stages["MM2_CkvKr"] = cube_t
        if accum_t.total_us > 0:
            stages["AccumCkvKr"] = accum_t
    else:
        stages["MM2_CkvKr"] = estimate_matmul_k_outer(
            "MM2_CkvKr", T, He, Hckv + Dr,
            active_cores=hw.aic_num, hw=hw,
            block=block_specs["MM2_CkvKr"], a_reused=True,
        )

    # MM3: Cq[T, Hcq] x weightUqQr[Hcq, N*(D+Dr)] -> QcQr
    stages["MM3_QcQr"] = estimate_matmul_k_outer(
        "MM3_QcQr", T, Hcq, N * (D + Dr),
        active_cores=hw.aic_num, hw=hw,
        block=block_specs["MM3_QcQr"], a_reused=False,
    )

    # MM4: Qc[T, D] x Uk[D, Hckv] -> Qn
    stages["MM4_Qn"] = estimate_matmul_k_outer(
        "MM4_Qn", T, D, Hckv,
        active_cores=hw.aic_num, hw=hw,
        block=block_specs["MM4_Qn"], a_reused=False,
    )

    vec_cores = hw.aiv_num

    # Vector ops use full T (single launch)
    stages["RmsNormCq"] = estimate_vector_op(
        "RmsNormCq", T * Hcq * out_size, T * Hcq, T * Hcq * 2,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCq", qm))

    ckv_kr_size = Hckv + Dr
    stages["RmsNormCkvKr"] = estimate_vector_op(
        "RmsNormCkvKr", T * Hckv * out_size, T * Hckv, T * Hckv * 2,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCkvKr", qm))

    stages["RopeKr"] = estimate_vector_op(
        "RopeKr", T * Dr * 4, T * Dr, T * Dr * 2,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeKr", qm))

    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", T * ckv_kr_size * 2, T * ckv_kr_size, T * ckv_kr_size * 2,
        hw, active_cores=vec_cores, is_scatter=True,
        n_instructions=vector_instr_count("ScatterCkvKr", qm))

    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        stages["DequantQc"] = estimate_vector_op(
            "DequantQc", T * N * D * out_size, T * N * D, T * N * D * 2,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DequantQc", qm))

    stages["RopeQr"] = estimate_vector_op(
        "RopeQr", T * N * Dr * 2 + T * Dr * 2, T * N * Dr, T * N * Dr * 2,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeQr", qm))

    if params.query_quant_mode == 1:
        stages["DynamicQuantQn"] = estimate_vector_op(
            "DynamicQuantQn", T * N * Hckv * 2, T * N * Hckv, T * N * Hckv,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DynamicQuantQn", qm))
        stages["MulQr"] = estimate_vector_op(
            "MulQr", T * N * Dr * 2, T * N * Dr, T * N * Dr * 2,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("MulQr", qm))

    return stages


def estimate_all_stages_split_m(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    spec: SplitMSpec,
    block_specs=None,
):
    """Estimate timing for all stages under M-axis split (prefill mode).

    Each core handles M_per_core tokens with full N.
    Vector ops use per-AIC vector cores (aiv_num // aic_num) instead of full pool.
    """
    M_per_core = spec.single_m
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    qm = params.quant_mode

    act_size = params.activation_dtype_size
    wt_size = params.weight_dtype_size
    out_size = params.mm_output_dtype_size

    if block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    stages = {}

    # MM1: tokenX[M_per_core, He] x weightDq[He, Hcq] -> Cq
    stages["MM1_Cq"] = estimate_matmul_split_m(
        "MM1_Cq", params.T, He, Hcq, spec,
        hw=hw, block=block_specs["MM1_Cq"], a_reused=False,
    )

    # MM2: tokenX[M_per_core, He] x weightCkvKr[He, Hckv+Dr] -> CkvKr
    # tokenX reused from MM1 (L2 warm)
    stages["MM2_CkvKr"] = estimate_matmul_split_m(
        "MM2_CkvKr", params.T, He, Hckv + Dr, spec,
        hw=hw, block=block_specs["MM2_CkvKr"], a_reused=True,
    )

    # MM3: Cq[M_per_core, Hcq] x weightUqQr[Hcq, N*(D+Dr)] -> QcQr
    stages["MM3_QcQr"] = estimate_matmul_split_m(
        "MM3_QcQr", params.T, Hcq, N * (D + Dr), spec,
        hw=hw, block=block_specs["MM3_QcQr"], a_reused=False,
    )

    # MM4: Qc[M_per_core, D] x Uk[D, Hckv] -> Qn
    stages["MM4_Qn"] = estimate_matmul_split_m(
        "MM4_Qn", params.T, D, Hckv, spec,
        hw=hw, block=block_specs["MM4_Qn"], a_reused=False,
    )

    # Vector ops: each AIC core gets aiv_num // aic_num vector cores
    vec_cores = max(hw.aiv_num // hw.aic_num, 1)

    # RMSNorm Cq (per core: M_per_core tokens)
    rmsnorm_cq_load = M_per_core * Hcq * out_size
    rmsnorm_cq_store = M_per_core * Hcq * 2
    stages["RmsNormCq"] = estimate_vector_op(
        "RmsNormCq", rmsnorm_cq_load, M_per_core * Hcq, rmsnorm_cq_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCq", qm),
    )

    # RMSNorm CkvKr — kernel only loads Hckv portion (Dr read separately by RopeKr)
    rmsnorm_ckvkr_load = M_per_core * Hckv * out_size
    rmsnorm_ckvkr_store = M_per_core * Hckv * 2
    stages["RmsNormCkvKr"] = estimate_vector_op(
        "RmsNormCkvKr", rmsnorm_ckvkr_load, M_per_core * Hckv, rmsnorm_ckvkr_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCkvKr", qm),
    )

    # RoPE Kr
    rope_kr_load = M_per_core * Dr * 2 + M_per_core * Dr * 2
    rope_kr_store = M_per_core * Dr * 2
    stages["RopeKr"] = estimate_vector_op(
        "RopeKr", rope_kr_load, M_per_core * Dr, rope_kr_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeKr", qm),
    )

    # Scatter CkvKr
    scatter_load = M_per_core * (Hckv + Dr) * 2
    scatter_store = M_per_core * (Hckv + Dr) * 2
    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", scatter_load, M_per_core * (Hckv + Dr), scatter_store,
        hw, active_cores=vec_cores, is_scatter=True,
        n_instructions=vector_instr_count("ScatterCkvKr", qm),
    )

    # DequantQc
    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        dequant_load = M_per_core * N * D * out_size
        dequant_store = M_per_core * N * D * 2
        stages["DequantQc"] = estimate_vector_op(
            "DequantQc", dequant_load, M_per_core * N * D, dequant_store,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DequantQc", qm),
        )

    # RoPE Qr
    rope_qr_load = M_per_core * N * Dr * 2 + M_per_core * Dr * 2
    rope_qr_store = M_per_core * N * Dr * 2
    stages["RopeQr"] = estimate_vector_op(
        "RopeQr", rope_qr_load, M_per_core * N * Dr, rope_qr_store,
        hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeQr", qm),
    )

    # DynamicQuant Qn
    if params.query_quant_mode == 1:
        dyn_quant_load = M_per_core * N * Hckv * 2
        dyn_quant_store = M_per_core * N * Hckv * 1
        stages["DynamicQuantQn"] = estimate_vector_op(
            "DynamicQuantQn", dyn_quant_load, M_per_core * N * Hckv, dyn_quant_store,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DynamicQuantQn", qm),
        )

    # MulQr
    if params.query_quant_mode == 1:
        mul_qr_load = M_per_core * N * Dr * 2
        mul_qr_store = M_per_core * N * Dr * 2
        stages["MulQr"] = estimate_vector_op(
            "MulQr", mul_qr_load, M_per_core * N * Dr, mul_qr_store,
            hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("MulQr", qm),
        )

    return stages


def compute_ub_constrained_baseM(hcq: int, ub_size: int) -> int:
    """Compute max baseM from UB capacity constraint for L0C→UB path.

    UB must hold matmul output (baseM * Hcq * 2B) plus vector intermediate
    buffers (~17KB for RmsNorm f32 working area, gamma constant, alignment).
    """
    overhead = 17 * 1024  # RmsNorm temp + gamma + alignment
    available = ub_size - overhead
    baseM = (available // (hcq * 2) // 16) * 16
    return max(baseM, 16)


def estimate_all_stages_l0c_ub(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
    spec: SplitMSpec,
    block_specs=None,
):
    """Estimate timing for all stages with L0C→UB core-internal pipeline.

    Key differences from estimate_all_stages_split_m:
    1. baseM constrained by UB capacity (must hold matmul output + vector intermediates)
    2. MM3/MM4: a_from_ub=True (A via UB→L1, not MTE2 from HBM)
    3. Vector ops: compute-only (data in UB, no MTE2/MTE3)
    4. UB→L1 transfer stages between vector ops and next matmul
    5. Scatter still uses MTE3→HBM (writes KVCache)
    """
    M_per_core = spec.single_m
    He = params.He
    Hcq = params.Hcq
    Hckv = params.Hckv
    D = params.D
    Dr = params.Dr
    N = params.N
    qm = params.quant_mode

    if block_specs is None:
        block_specs = get_default_block_specs(params, hw)

    stages = {}

    # MM1: tokenX[M_per_core, He] x weightDq[He, Hcq] -> Cq
    stages["MM1_Cq"] = estimate_matmul_split_m_l0c_ub(
        "MM1_Cq", params.T, He, Hcq, spec,
        hw=hw, block=block_specs["MM1_Cq"],
        a_from_ub=False, a_reused=False,
    )

    # MM2: tokenX reused from MM1 (L2 warm)
    stages["MM2_CkvKr"] = estimate_matmul_split_m_l0c_ub(
        "MM2_CkvKr", params.T, He, Hckv + Dr, spec,
        hw=hw, block=block_specs["MM2_CkvKr"],
        a_from_ub=False, a_reused=True,
    )

    # MM3: A from UB→L1 (RmsNormCq output, core-internal)
    stages["MM3_QcQr"] = estimate_matmul_split_m_l0c_ub(
        "MM3_QcQr", params.T, Hcq, N * (D + Dr), spec,
        hw=hw, block=block_specs["MM3_QcQr"],
        a_from_ub=True,
    )

    # MM4: A from UB→L1 (DequantQc output, core-internal)
    stages["MM4_Qn"] = estimate_matmul_split_m_l0c_ub(
        "MM4_Qn", params.T, D, Hckv, spec,
        hw=hw, block=block_specs["MM4_Qn"],
        a_from_ub=True,
    )

    # UB→L1 transfer stages (core-internal, ~404 GB/s)
    ub_l1_bw = hw.per_core_bw("ub_to_l1", 1)
    if ub_l1_bw > 0:
        ub_l1_cq_us = M_per_core * Hcq * 2 / ub_l1_bw * 1e6
        ub_l1_qc_us = M_per_core * D * 2 / ub_l1_bw * 1e6
    else:
        ub_l1_cq_us = 0.0
        ub_l1_qc_us = 0.0
    stages["UB_to_L1_Cq"] = VectorOpTiming(
        "UB_to_L1_Cq", 0.0, 0.0, 0.0, ub_l1_cq_us, "UB_L1")
    stages["UB_to_L1_Qc"] = VectorOpTiming(
        "UB_to_L1_Qc", 0.0, 0.0, 0.0, ub_l1_qc_us, "UB_L1")

    # Vector ops: compute-only (data in UB from L0C→UB)
    vec_cores = max(hw.aiv_num // hw.aic_num, 1)

    stages["RmsNormCq"] = estimate_vector_op_l0c_ub(
        "RmsNormCq", M_per_core * Hcq, hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCq", qm))

    stages["RmsNormCkvKr"] = estimate_vector_op_l0c_ub(
        "RmsNormCkvKr", M_per_core * Hckv, hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RmsNormCkvKr", qm))

    stages["RopeKr"] = estimate_vector_op_l0c_ub(
        "RopeKr", M_per_core * Dr, hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeKr", qm))

    # Scatter: still MTE3→HBM (writes KVCache)
    scatter_load = M_per_core * (Hckv + Dr) * 2
    scatter_store = M_per_core * (Hckv + Dr) * 2
    stages["ScatterCkvKr"] = estimate_vector_op(
        "ScatterCkvKr", scatter_load, M_per_core * (Hckv + Dr), scatter_store,
        hw, active_cores=vec_cores, is_scatter=True,
        n_instructions=vector_instr_count("ScatterCkvKr", qm))

    if is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm):
        stages["DequantQc"] = estimate_vector_op_l0c_ub(
            "DequantQc", M_per_core * N * D, hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DequantQc", qm))

    stages["RopeQr"] = estimate_vector_op_l0c_ub(
        "RopeQr", M_per_core * N * Dr, hw, active_cores=vec_cores,
        n_instructions=vector_instr_count("RopeQr", qm))

    if params.query_quant_mode == 1:
        stages["DynamicQuantQn"] = estimate_vector_op_l0c_ub(
            "DynamicQuantQn", M_per_core * N * Hckv, hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("DynamicQuantQn", qm))
        stages["MulQr"] = estimate_vector_op_l0c_ub(
            "MulQr", M_per_core * N * Dr, hw, active_cores=vec_cores,
            n_instructions=vector_instr_count("MulQr", qm))

    return stages
