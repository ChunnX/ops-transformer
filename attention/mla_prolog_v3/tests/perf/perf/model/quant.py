#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantization mode enumeration and helpers.

Resolves user-facing (weight_quant_mode, kv_cache_quant_mode) attribute pairs
into the internal QuantMode enum used throughout the perf model.
"""

from enum import IntEnum


class QuantMode(IntEnum):
    NO_QUANT = 0
    PARTIAL_QUANT_KV_NO_QUANT = 1
    PARTIAL_QUANT_KV_QUANT_PER_CHANNEL = 2
    FULL_QUANT_KV_NO_QUANT = 3
    FULL_QUANT_KV_QUANT_PER_TENSOR = 4
    PARTIAL_QUANT_KV_QUANT_PERTILE = 5
    FULL_QUANT_KV_QUANT_PERTILE = 6
    MXFP8_FULL_QUANT_KV_NO_QUANT = 7
    MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR = 8
    MXFP8_FULL_QUANT_KV_QUANT_PER_TILE = 9


def resolve_quant_mode(weight_quant_mode: int, kv_cache_quant_mode: int) -> QuantMode:
    """Derive the internal QUANT_MODE enum from user-facing weight/kv quant params."""
    if weight_quant_mode == 0:
        if kv_cache_quant_mode == 0:
            return QuantMode.NO_QUANT
        elif kv_cache_quant_mode == 3:
            return QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE
        return QuantMode.NO_QUANT
    elif weight_quant_mode in (1, 2):
        if weight_quant_mode == 1:
            if kv_cache_quant_mode == 0:
                return QuantMode.PARTIAL_QUANT_KV_NO_QUANT
            elif kv_cache_quant_mode == 1:
                return QuantMode.PARTIAL_QUANT_KV_QUANT_PER_CHANNEL
            elif kv_cache_quant_mode == 2:
                return QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE
            return QuantMode.PARTIAL_QUANT_KV_NO_QUANT
        else:
            if kv_cache_quant_mode == 0:
                return QuantMode.FULL_QUANT_KV_NO_QUANT
            elif kv_cache_quant_mode == 1:
                return QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR
            elif kv_cache_quant_mode == 2:
                return QuantMode.FULL_QUANT_KV_QUANT_PERTILE
            return QuantMode.FULL_QUANT_KV_NO_QUANT
    elif weight_quant_mode == 3:
        if kv_cache_quant_mode == 0:
            return QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT
        elif kv_cache_quant_mode == 1:
            return QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR
        elif kv_cache_quant_mode == 3:
            return QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE
        return QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT
    return QuantMode.NO_QUANT


def is_full_quant(qm: QuantMode) -> bool:
    return qm in (
        QuantMode.FULL_QUANT_KV_NO_QUANT,
        QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.FULL_QUANT_KV_QUANT_PERTILE,
        QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,
    )


def is_mxfp8(qm: QuantMode) -> bool:
    return qm in (
        QuantMode.MXFP8_FULL_QUANT_KV_NO_QUANT,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,
    )


def has_kv_quant(qm: QuantMode) -> bool:
    """True when the quant mode quantizes the KV cache (not just weights)."""
    return qm in (
        QuantMode.PARTIAL_QUANT_KV_QUANT_PER_CHANNEL,
        QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE,
        QuantMode.FULL_QUANT_KV_QUANT_PERTILE,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.MXFP8_FULL_QUANT_KV_QUANT_PER_TILE,
    )


def is_int8_quant(qm: QuantMode) -> bool:
    return qm in (
        QuantMode.PARTIAL_QUANT_KV_NO_QUANT,
        QuantMode.PARTIAL_QUANT_KV_QUANT_PER_CHANNEL,
        QuantMode.PARTIAL_QUANT_KV_QUANT_PERTILE,
        QuantMode.FULL_QUANT_KV_NO_QUANT,
        QuantMode.FULL_QUANT_KV_QUANT_PER_TENSOR,
        QuantMode.FULL_QUANT_KV_QUANT_PERTILE,
    )


def dtype_size(quant_mode: QuantMode) -> int:
    """Return weight element size in bytes based on quant mode."""
    if is_full_quant(quant_mode) or is_int8_quant(quant_mode) or is_mxfp8(quant_mode):
        return 1
    return 2  # BF16
