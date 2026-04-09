#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dequantOpt UB-pipelined model: L0C → UB → L1 path between MM3 and MM4.

Models the optimization where mmQcQr's L0C output is moved directly into UB
(via the l0c_to_ub iterator with VDEQ16 inline dequant), processed by RoPE in
UB, then moved into L1 (via the ub_to_l1 iterator) to feed mmQn — all within
one AI core, with intra-core HardEvent sync instead of cross-core flags.

Reference implementations in production code:
- attention/mla_preprocess/op_kernel/l0c_to_ub_iterator.h  (L0C → UB w/ VDEQ16)
- attention/mla_preprocess/op_kernel/l1_to_ub_iterator.h   (ub_to_l1 / l1_to_ub)

This module is purely additive — it does not modify any existing perf model
function. The bit-identical regression invariant is preserved.
"""

import math
from dataclasses import dataclass

from .hardware import AscendHWSpec
from .operator_params import OperatorParams
from .quant import is_full_quant, is_int8_quant, is_mxfp8


@dataclass
class DequantOptStageTiming:
    """Timing for the fused L0C→UB→L1 dequant+rope stage.

    Fields are scoped to one AI core's worth of work for one stepBatchSize.
    intra_core_sync_us is assumed 0 per CLAUDE.md hypothesis H6 (HardEvent
    flags inside one AI core are below the model's resolution).
    """
    name: str = "DequantRopeUB"
    l0c_to_ub_us: float = 0.0
    ub_compute_us: float = 0.0
    ub_to_l1_us: float = 0.0
    intra_core_sync_us: float = 0.0
    total_us: float = 0.0
    bound: str = ""

    # Resource accounting (for budget reporting)
    ub_bytes: int = 0
    l1_bytes: int = 0
    per_core_n: int = 0
    active_cores: int = 0


@dataclass
class DequantOptBudget:
    """UB / L1 budget summary for one configuration."""
    ub_used_bytes: int
    ub_total_bytes: int
    l1_used_bytes: int
    l1_total_bytes: int

    @property
    def ub_fraction(self) -> float:
        return self.ub_used_bytes / self.ub_total_bytes if self.ub_total_bytes else 0.0

    @property
    def l1_fraction(self) -> float:
        return self.l1_used_bytes / self.l1_total_bytes if self.l1_total_bytes else 0.0

    @property
    def fits(self) -> bool:
        return self.ub_fraction <= 1.0 and self.l1_fraction <= 1.0


def estimate_dequant_opt_pipeline(
    params: OperatorParams,
    tiling,
    hw: AscendHWSpec,
) -> DequantOptStageTiming:
    """Estimate the L0C → UB(dequant+rope) → L1 fused stage.

    Per-core breakdown for one stepBatchSize:
      1. L0C → UB direct copy (via l0c_to_ub iterator, optional VDEQ16 inline
         dequant). Uses fixpipe bandwidth (`l0c_to_out`) — local, no contention.
      2. UB vector compute: with VDEQ16 the Qc region is already dequanted, so
         only the RoPE on the Qr region runs in vector. We charge the full
         dequant+cast+rope element count to be conservative for non-VDEQ16
         quant modes.
      3. UB → L1 direct copy (via ub_to_l1 iterator). Uses `ub_to_l1`
         bandwidth (245 B/cycle × 1.65 GHz ≈ 404 GB/s on 950).

    The three steps are conservatively summed (not max'd): a single decode
    token has only one in-flight work unit per AI core, so there's no internal
    double-buffering. The intra-core sync cost is 0 per H6.
    """
    if not hw.has_cycle_spec:
        # The model is meaningful only on the per-cycle (950) path; on the
        # legacy chip-total 910B path the L0C→UB and UB→L1 buses are not
        # exposed by the hardware spec.
        return DequantOptStageTiming(name="DequantRopeUB", total_us=0.0,
                                     bound="N/A")

    T = tiling.step_batch_size
    N = params.N
    D = params.D
    Dr = params.Dr

    out_size = params.mm_output_dtype_size  # MM3 output dtype (INT32 for quant, BF16 otherwise)
    bf16_size = 2

    # Per-core slice of MM3 output columns. Mirrors the host tiling computation
    # in tiling/config.py: when dequant_opt is on, mm3 splits N along (D+Dr)
    # boundaries; otherwise the standard split-N alignment is used.
    active_cores = max(tiling.mm3_block_num, 1)
    full_n = N * (D + Dr)
    per_core_n = math.ceil(full_n / active_cores)

    # --- 1. L0C → UB direct transfer (with optional VDEQ16 inline dequant) ---
    l0c_to_ub_bytes = T * per_core_n * out_size
    fixpipe_bw = hw.per_core_bw("l0c_to_out", active_cores)
    l0c_to_ub_s = l0c_to_ub_bytes / fixpipe_bw if fixpipe_bw > 0 else 0.0

    # --- 2. UB compute: dequant + cast + rope ---
    # Element counts per core. With VDEQ16 the dequant on Qc is folded into the
    # MTE2 transfer, but we still charge the elements here so that non-VDEQ16
    # paths (per-token quant) are not under-counted.
    qc_elements = (per_core_n * D) // (D + Dr) * T  # Qc portion of per_core_n
    qr_elements = (per_core_n * Dr) // (D + Dr) * T
    # Dequant Qc: cast(i32→f32) + mul(rowScale) + mul(colScale) = 3 instructions
    # RoPE Qr: 6 instructions (same as RopeQr in orchestration)
    if hw.has_vec_instr_spec:
        from .vector_ops import vector_compute_time_s
        ub_compute_s = (vector_compute_time_s(qc_elements, 3, hw)
                        + vector_compute_time_s(qr_elements, 6, hw))
    else:
        compute_ops = qc_elements * 3 + qr_elements * 4
        vec_throughput = hw.vector_ops_per_sec_bf16 / hw.aiv_num  # per AIV core
        ub_compute_s = compute_ops / vec_throughput if vec_throughput > 0 else 0.0

    # --- 3. UB → L1 direct transfer ---
    ub_to_l1_bytes = T * per_core_n * bf16_size
    ub_to_l1_bw = hw.per_core_bw("ub_to_l1", active_cores)
    ub_to_l1_s = ub_to_l1_bytes / ub_to_l1_bw if ub_to_l1_bw > 0 else 0.0

    # Conservative sum (no internal pingpong assumed for one stepBatchSize)
    total_s = l0c_to_ub_s + ub_compute_s + ub_to_l1_s

    # Bound classification for the report
    candidates = {
        "L0C2UB": l0c_to_ub_s,
        "UBCOMPUTE": ub_compute_s,
        "UB2L1": ub_to_l1_s,
    }
    bound = max(candidates, key=candidates.get)

    us = 1e6
    return DequantOptStageTiming(
        name="DequantRopeUB",
        l0c_to_ub_us=l0c_to_ub_s * us,
        ub_compute_us=ub_compute_s * us,
        ub_to_l1_us=ub_to_l1_s * us,
        intra_core_sync_us=0.0,
        total_us=total_s * us,
        bound=bound,
        ub_bytes=_estimate_ub_bytes(per_core_n, T, out_size, bf16_size, Dr),
        l1_bytes=_estimate_l1_bytes(per_core_n, T, bf16_size),
        per_core_n=per_core_n,
        active_cores=active_cores,
    )


def _estimate_ub_bytes(per_core_n: int, T: int, in_size: int, bf16_size: int,
                       Dr: int) -> int:
    """UB byte budget per AI core for the fused stage.

    Layout (with double-buffering on the qcqr input and the L1 output):
      - qcqr input  (×2 ping-pong)         : 2 * T * per_core_n * in_size
      - dequant scale (per-N)              : per_core_n * 4
      - FP32 working                       : T * per_core_n * 4
      - BF16 cast / rope working           : T * per_core_n * bf16_size
      - sin/cos                            : 2 * Dr * bf16_size
      - rope temp                          : Dr * bf16_size
      - UB → L1 output  (×2 ping-pong)     : 2 * T * per_core_n * bf16_size
    """
    qcqr_in = 2 * T * per_core_n * in_size
    dequant_scale = per_core_n * 4
    fp32_work = T * per_core_n * 4
    bf16_work = T * per_core_n * bf16_size
    sincos = 2 * Dr * bf16_size
    rope_temp = Dr * bf16_size
    l1_out = 2 * T * per_core_n * bf16_size
    return qcqr_in + dequant_scale + fp32_work + bf16_work + sincos + rope_temp + l1_out


def _estimate_l1_bytes(per_core_n: int, T: int, bf16_size: int) -> int:
    """L1 byte budget per AI core for holding the dequanted Qc/Qr buffer.

    Double-buffered so MM4 can read iteration N while the vector unit is
    landing iteration N+1.
    """
    return 2 * T * per_core_n * bf16_size


def is_dequant_opt_applicable(params: OperatorParams, tiling) -> bool:
    """Whether the dequantOpt UB pipeline can be applied for this config.

    Mirrors the host-side gate in `perf/tiling/config.py:148-149`: the
    optimization is meaningful when the MM3 weight is byte-quant (FP8/INT8)
    or MXFP8, *and* N >= 8 so that the per-core (D+Dr) alignment yields a
    non-trivial split.
    """
    qm = params.quant_mode
    has_byte_weight = is_full_quant(qm) or is_int8_quant(qm) or is_mxfp8(qm)
    return has_byte_weight and params.N >= 8


def compute_budget(timing: DequantOptStageTiming, hw: AscendHWSpec) -> DequantOptBudget:
    """Build the UB / L1 budget summary against the hardware caps."""
    return DequantOptBudget(
        ub_used_bytes=timing.ub_bytes,
        ub_total_bytes=hw.cache.ub_size,
        l1_used_bytes=timing.l1_bytes,
        l1_total_bytes=hw.cache.l1_size,
    )
