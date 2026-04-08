#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector operation timing model.

Models RmsNorm/RoPE/Scatter/Dequant/DynamicQuant as load → compute → store
through MTE2 → vector unit → MTE3, with double-buffered overlap.
"""

from .hardware import AscendHWSpec
from .timing_types import VectorOpTiming


def estimate_vector_op(
    name: str,
    load_bytes: int,
    compute_elements: int,
    store_bytes: int,
    hw: AscendHWSpec,
    active_cores: int = 48,
    is_scatter: bool = False,
) -> VectorOpTiming:
    """Estimate vector operation timing (RMSNorm, RoPE, Scatter, Dequant, DynamicQuant)."""
    C = max(active_cores, 1)
    per_core_load = load_bytes / C
    per_core_store = store_bytes / C
    per_core_compute = compute_elements / C

    mte2_time_s = per_core_load / (hw.mte2_bw_bytes_per_sec / C)
    vec_time_s = per_core_compute / (hw.vector_ops_per_sec_bf16 / hw.aiv_num)

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
