#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matmul block specification and on-chip cache fit / stepK derivation.

A MatmulBlockSpec captures the block sizes (baseM/baseN/baseK/stepK) and dtypes
for a single matmul. The two helpers validate that the chosen block fits the
on-chip buffer hierarchy and derive the optimal stepK from L1 capacity.
"""

import math
from dataclasses import dataclass
from typing import Optional

from .hardware import AscendHWSpec


@dataclass
class MatmulBlockSpec:
    """Block sizes for a single matmul's inner loop."""
    name: str
    baseM: int
    baseN: int
    baseK: int
    stepK: int = 4
    mode: str = "split_k"   # "split_k" or "full_load"
    dtype_a_size: int = 2
    dtype_b_size: int = 2
    dtype_c_size: int = 2


def check_cache_fit(block: MatmulBlockSpec, hw: AscendHWSpec,
                    n_per_core: Optional[int] = None,
                    M: Optional[int] = None,
                    K: Optional[int] = None) -> bool:
    """Verify that block sizes fit in on-chip buffers.

    Default: double buffering (2x) for all buffers.
    Full-load optimization: if the entire matrix fits in the buffer AND
    using single buffer (1x) allows fitting, use single buffer.
    """
    n = n_per_core if n_per_core is not None else block.baseN
    actual_M = M if M is not None else block.baseM
    actual_K = K if K is not None else block.baseK

    tile_a = block.baseM * block.baseK * block.dtype_a_size
    tile_b = block.baseK * block.baseN * block.dtype_b_size
    tile_c = block.baseM * block.baseN * block.dtype_c_size

    # L0A: default double buffer; try full-load single buffer as fallback
    full_a = actual_M * actual_K * block.dtype_a_size
    if tile_a * 2 > hw.cache.l0a_size:
        # Double buffer doesn't fit — try full-load single buffer
        if full_a <= hw.cache.l0a_size and tile_a <= hw.cache.l0a_size:
            pass  # full-load OK
        else:
            return False

    # L0B: default double buffer; try full-load single buffer as fallback
    full_b = actual_K * n * block.dtype_b_size
    if tile_b * 2 > hw.cache.l0b_size:
        if full_b <= hw.cache.l0b_size and tile_b <= hw.cache.l0b_size:
            pass
        else:
            return False

    # L0C: default double buffer; try full-load single buffer as fallback
    full_c = actual_M * n * block.dtype_c_size
    if tile_c * 2 > hw.cache.l0c_size:
        if full_c <= hw.cache.l0c_size and tile_c <= hw.cache.l0c_size:
            pass
        else:
            return False

    # L1: default double buffer for per-step tile
    l1_per_step_a = block.baseM * block.baseK * block.dtype_a_size
    l1_per_step_b = block.baseK * n * block.dtype_b_size
    l1_per_step = l1_per_step_a + l1_per_step_b
    if l1_per_step * 2 > hw.cache.l1_size:
        # Double buffer doesn't fit — try full-load single buffer
        full_ab = full_a + full_b
        if full_ab <= hw.cache.l1_size and l1_per_step <= hw.cache.l1_size:
            pass
        else:
            return False
    return True


def derive_stepK(block: MatmulBlockSpec, K: int, hw: AscendHWSpec,
                 n_per_core: Optional[int] = None,
                 M: Optional[int] = None) -> int:
    """Derive optimal stepK from L1 size constraint.

    Default: double buffer (L1/2). If all K fits in a single L1 load
    with full-load single buffer AND it reduces kL1 loops, use it.
    """
    n = n_per_core if n_per_core is not None else block.baseN
    actual_M = M if M is not None else block.baseM

    l1_per_step_a = block.baseM * block.baseK * block.dtype_a_size
    l1_per_step_b = block.baseK * n * block.dtype_b_size
    per_step = l1_per_step_a + l1_per_step_b
    if per_step == 0:
        return 1

    k_iters = math.ceil(K / block.baseK) if block.baseK > 0 else 1

    # Default: double buffer (L1/2)
    double_buf_stepK = hw.cache.l1_size // 2 // per_step

    # Optimization: try full-load single buffer if it covers all K iterations
    # and produces a better result (fewer kL1 loops)
    single_buf_stepK = hw.cache.l1_size // per_step
    if single_buf_stepK >= k_iters:
        # Full load fits with single buffer → kL1Loops=1
        # Only use if double buffer wouldn't achieve the same (or doesn't fit)
        if double_buf_stepK < k_iters:
            return k_iters  # full-load optimization wins

    if double_buf_stepK < 1:
        # Double buffer doesn't fit — last resort: try single buffer
        if single_buf_stepK >= 1:
            return min(single_buf_stepK, k_iters)
        return 0  # invalid
    return min(double_buf_stepK, k_iters)
