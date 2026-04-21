#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector operation timing model.

Models RmsNorm/RoPE/Scatter/Dequant/DynamicQuant as load -> compute -> store
through MTE2 -> vector unit -> MTE3, with double-buffered overlap.

Compute time uses an instruction-level model on Ascend 950:
  - Each vector instruction processes data at vec_instr_bytes_per_cycle (256 B/cycle)
  - IPC = 1.3 models pipeline overlap between instructions in a basic block
  - Different quant modes have different instruction counts per stage

On Ascend 910B (legacy), falls back to chip-total vector_ops_per_sec_bf16.
"""

from .hardware import AscendHWSpec
from .quant import QuantMode, is_full_quant, is_int8_quant, is_mxfp8, has_kv_quant
from .timing_types import VectorOpTiming


# Per-stage instruction counts derived from kernel analysis.
# All instructions operate on float32 (4 bytes/element) at 256 B/cycle.
#
# Instruction rationale:
#   RmsNormCq BF16:  cast(bf16->f32) + mul(x^2) + reducesum + mul(rsqrt) + mul(gamma) + cast(f32->bf16) = 6
#   RmsNormCq INT8:  + cast(i32->f32) + mul(deqScale) at input = 7
#   RmsNormCkvKr + KV quant: replace final cast with abs + reducemax + mul(scale) + cast(f32->int8) = +3 net
#   RoPE:            mul(even*cos) + mul(odd*sin) + mul(odd*cos) + mul(even*sin) + add + sub = 6
#   DequantQc:       cast(i32->f32) + mul(rowScale) + mul(colScale) = 3
#   DynamicQuantQn:  abs + reducemax + mul(scale) + cast(f32->bf16) + cast(bf16->int8) = 5
#   MulQr:           mul(scale) + cast = 2
#   ScatterCkvKr:    data movement dominated, minimal compute = 1
_INSTR_TABLE_BF16 = {
    "RmsNormCq": 6,
    "RmsNormCkvKr": 6,
    "RopeKr": 6,
    "ScatterCkvKr": 1,
    "RopeQr": 6,
    "DynamicQuantQn": 5,
    "MulQr": 2,
}

_INSTR_TABLE_QUANT = {
    "RmsNormCq": 7,
    "RmsNormCkvKr": 7,        # no KV quant
    "RmsNormCkvKr_kv": 10,    # with KV quant
    "RopeKr": 6,
    "ScatterCkvKr": 1,
    "DequantQc": 3,
    "RopeQr": 6,
    "DynamicQuantQn": 5,
    "MulQr": 2,
}


def vector_instr_count(stage: str, quant_mode: QuantMode) -> int:
    """Return the number of vector instructions per element for a given stage.

    Derived from kernel analysis of each vector stage's instruction sequence.
    All instructions operate on float32 data (4 bytes per element).
    """
    is_quant = is_full_quant(quant_mode) or is_int8_quant(quant_mode) or is_mxfp8(quant_mode)
    if is_quant:
        if stage == "RmsNormCkvKr" and has_kv_quant(quant_mode):
            return _INSTR_TABLE_QUANT["RmsNormCkvKr_kv"]
        return _INSTR_TABLE_QUANT.get(stage, 1)
    return _INSTR_TABLE_BF16.get(stage, 1)


def vector_compute_time_s(per_core_elements: int, n_instructions: int,
                          hw: AscendHWSpec) -> float:
    """Vector basic block time in seconds for one AIV core.

    Formula:
        per_instr_cycles = per_core_elements * 4 / vec_instr_bytes_per_cycle
        total_cycles = n_instructions * per_instr_cycles / vec_ipc
        time_s = total_cycles / freq_hz

    All instructions operate on float32 (4 bytes/element) at 256 B/cycle.
    IPC=1.3 models pipeline overlap between independent instructions.
    """
    data_bytes = per_core_elements * 4  # float32
    per_instr_cycles = data_bytes / hw.vec_instr_bytes_per_cycle
    total_cycles = n_instructions * per_instr_cycles / hw.vec_ipc
    return total_cycles / hw.freq_hz


def estimate_vector_op(
    name: str,
    load_bytes: int,
    compute_elements: int,
    store_bytes: int,
    hw: AscendHWSpec,
    active_cores: int = 48,
    is_scatter: bool = False,
    n_instructions: int = 0,
) -> VectorOpTiming:
    """Estimate vector operation timing (RMSNorm, RoPE, Scatter, Dequant, DynamicQuant).

    Args:
        compute_elements: Number of data elements (not "ops"). E.g., T * Hcq.
        n_instructions: Number of vector instructions per element for this stage.
            When > 0 and hw.has_vec_instr_spec, uses instruction-level model.
            When 0, falls back to legacy chip-total model.
    """
    C = max(active_cores, 1)
    per_core_load = load_bytes / C
    per_core_store = store_bytes / C
    per_core_compute = compute_elements / C

    mte2_time_s = per_core_load / (hw.mte2_bw_bytes_per_sec / C)

    if hw.has_vec_instr_spec and n_instructions > 0:
        vec_time_s = vector_compute_time_s(per_core_compute, n_instructions, hw)
    else:
        # Legacy chip-total model (910B or callers that haven't migrated).
        # When n_instructions > 0, compute_elements is raw data elements;
        # scale by n_instructions to approximate ops count for the legacy formula.
        effective_compute = per_core_compute
        if n_instructions > 0:
            effective_compute *= n_instructions
        vec_time_s = effective_compute / (hw.vector_ops_per_sec_bf16 / hw.aiv_num)

    effective_mte3_bw = hw.mte3_bw_bytes_per_sec
    if is_scatter:
        effective_mte3_bw *= hw.scatter_efficiency
    mte3_time_s = per_core_store / (effective_mte3_bw / C)

    total_s = max(mte2_time_s, vec_time_s, mte3_time_s)

    if total_s == mte2_time_s:
        bound = "MTE2"
    elif total_s == vec_time_s:
        bound = "VECTOR"
    else:
        bound = "MTE3"

    us = 1e6
    return VectorOpTiming(
        name=name,
        mte2_load_us=mte2_time_s * us,
        vec_compute_us=vec_time_s * us,
        mte3_store_us=mte3_time_s * us,
        total_us=total_s * us,
        bound=bound,
    )


def estimate_vector_op_l0c_ub(
    name: str,
    compute_elements: int,
    hw: AscendHWSpec,
    active_cores: int = 2,
    n_instructions: int = 0,
) -> VectorOpTiming:
    """Estimate vector op timing with L0C→UB path (data already in UB).

    With L0C→UB, matmul output is delivered directly to UB. Vector ops
    read from UB (no MTE2 load from HBM) and write results back to UB
    (no MTE3 store to HBM). The UB→L1 transfer for the next matmul's
    A matrix is modeled as a separate pipeline stage.

    Total = vector compute time only (no MTE2/MTE3 overhead).
    """
    C = max(active_cores, 1)
    per_core_compute = compute_elements / C

    if hw.has_vec_instr_spec and n_instructions > 0:
        vec_time_s = vector_compute_time_s(per_core_compute, n_instructions, hw)
    else:
        vec_time_s = per_core_compute / (hw.vector_ops_per_sec_bf16 / hw.aiv_num)

    us = 1e6
    return VectorOpTiming(
        name=name,
        mte2_load_us=0.0,
        vec_compute_us=vec_time_s * us,
        mte3_store_us=0.0,
        total_us=vec_time_s * us,
        bound="VECTOR",
    )
