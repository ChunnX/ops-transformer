#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""NumPy reference for QkvRmsNormRopeCacheWithKScale.

The implementation follows the public operator contract, not the AIC/AIV
schedule.  It intentionally keeps the low-precision casts that are observable
at the interface: RoPE output is rounded to BF16 before the dense rotation,
FP8 casts saturate at 448, and M-RoPE INT8 quantization contains the device
path's FP16 intermediate rounding.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

try:
    import ml_dtypes
except ImportError as exc:  # pragma: no cover - environment error is explicit
    raise ImportError(
        "QkvRmsNormRopeCacheWithKScale golden requires ml_dtypes"
    ) from exc


FP8_MAX = np.float32(448.0)
INT8_MAX = np.float32(127.0)
BF16_DTYPE = np.dtype(ml_dtypes.bfloat16)
FP8_DTYPE = np.dtype(ml_dtypes.float8_e4m3fn)


class GoldenResult(NamedTuple):
    q_out: np.ndarray
    q_scale: Optional[np.ndarray]
    k_cache: np.ndarray
    v_cache: np.ndarray
    k_scale_cache: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def to_bf16(value) -> np.ndarray:
    """Round to BF16 (RNE) and retain the BF16 dtype."""

    return np.asarray(value, dtype=np.float32).astype(BF16_DTYPE)


def saturating_fp8_cast(value) -> np.ndarray:
    """Cast to E4M3FN with the operator's finite-value saturation policy."""

    array = np.asarray(value)
    if array.dtype == FP8_DTYPE:
        return array.copy()
    clipped = np.clip(array.astype(np.float32, copy=False), -FP8_MAX, FP8_MAX)
    return clipped.astype(FP8_DTYPE)


def fp8_storage(value) -> np.ndarray:
    """Return E4M3FN storage bytes for exact-rate comparisons."""

    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.copy()
    if array.dtype != FP8_DTYPE:
        array = saturating_fp8_cast(array)
    return np.ascontiguousarray(array).view(np.uint8)


def _dynamic_quant(value: np.ndarray, target: str):
    value = np.asarray(value, dtype=np.float32)
    quant_max = FP8_MAX if target == "fp8" else INT8_MAX
    scale = (np.max(np.abs(value), axis=-1) / quant_max).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = value / scale[..., None]
    if target == "fp8":
        quantized = saturating_fp8_cast(normalized)
    else:
        rounded = np.rint(normalized.astype(np.float16).astype(np.float32))
        quantized = np.clip(rounded, -INT8_MAX, INT8_MAX).astype(np.int8)
    return quantized, scale


def _normal_rope_positions(token_num: int, query_start_loc, seq_lens):
    starts = np.asarray(query_start_loc, dtype=np.int64)
    lengths = np.asarray(seq_lens, dtype=np.int64)
    _require(starts.ndim == 1 and starts.size >= 2,
             "query_start_loc must have shape [Batch+1]")
    _require(lengths.ndim == 1 and lengths.size + 1 == starts.size,
             "seq_lens must have shape [Batch]")
    _require(starts[0] == 0 and starts[-1] == token_num,
             "query_start_loc must start at 0 and end at T")
    _require(bool(np.all(starts[1:] >= starts[:-1])),
             "query_start_loc must be nondecreasing")

    positions = np.empty(token_num, dtype=np.int64)
    for batch in range(lengths.size):
        begin, end = int(starts[batch]), int(starts[batch + 1])
        current = end - begin
        first = int(lengths[batch]) - current
        _require(first >= 0,
                 "seq_lens values must be at least the number of current tokens")
        positions[begin:end] = first + np.arange(current, dtype=np.int64)
    return positions


def _select_cos_sin(
    cos_sin,
    token_num: int,
    query_start_loc,
    seq_lens,
    mrope_position,
    mrope_section: Optional[Sequence[int]],
):
    table = np.asarray(cos_sin, dtype=np.float32)
    head_dim = int(table.shape[1])
    half_dim = head_dim // 2
    has_position = mrope_position is not None
    has_section = mrope_section is not None and len(mrope_section) != 0
    _require(has_position == has_section,
             "mrope_position and non-empty mrope_section must be provided together")

    if not has_position:
        _require(query_start_loc is not None and seq_lens is not None,
                 "RoPE requires query_start_loc and seq_lens")
        positions = _normal_rope_positions(token_num, query_start_loc, seq_lens)
        _require(bool(np.all(positions < table.shape[0])),
                 "RoPE position exceeds cos_sin rows")
        selected = table[positions]
        return selected[:, :half_dim], selected[:, half_dim:], False

    _require(query_start_loc is None and seq_lens is None,
             "M-RoPE requires query_start_loc=None and seq_lens=None")
    section_t, section_h, section_w = (int(item) for item in mrope_section)
    _require(min(section_t, section_h, section_w) >= 0,
             "mrope_section values must be nonnegative")
    _require(section_t + section_h + section_w <= half_dim,
             "sum(mrope_section) must not exceed D/2")
    _require(section_h <= 21 and section_w <= 21,
             "for D=128, M-RoPE H/W sections must not exceed 21")

    position = np.asarray(mrope_position, dtype=np.int64)
    _require(position.shape == (token_num, 3),
             "mrope_position must have shape [T,3]")
    _require(bool(np.all(position >= 0) and np.all(position < table.shape[0])),
             "M-RoPE position exceeds cos_sin rows")

    raw = table[position]  # [T, 3, D]
    lane = np.arange(half_dim, dtype=np.int64)
    group = lane // 3
    axis = np.zeros(half_dim, dtype=np.int64)
    axis[(lane % 3 == 1) & (group < section_h)] = 1
    axis[(lane % 3 == 2) & (group < section_w)] = 2
    token = np.arange(token_num, dtype=np.int64)[:, None]
    cos = raw[token, axis[None, :], lane[None, :]]
    sin = raw[token, axis[None, :], (half_dim + lane)[None, :]]
    return cos, sin, True


def _apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray):
    half_dim = value.shape[-1] // 2
    low, high = value[..., :half_dim], value[..., half_dim:]
    return np.concatenate(
        (low * cos[:, None, :] - high * sin[:, None, :],
         high * cos[:, None, :] + low * sin[:, None, :]),
        axis=-1,
    )


def _normalize_dtype_name(value) -> str:
    if value is None:
        return "float8_e4m3fn"
    if isinstance(value, (int, np.integer)):
        return {27: "bfloat16", 36: "float8_e4m3fn"}.get(int(value), str(value))
    return str(value).lower().replace("torch.", "").replace("dt_", "")


def qkv_rms_norm_rope_cache_with_k_scale_numpy(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    rotation,
    v_scale,
    mrope_position,
    *,
    head_nums,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=1e-6,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    q_out_dtype=None,
) -> GoldenResult:
    """Compute all logical outputs without modifying the input caches."""

    _require(head_nums is not None and len(head_nums) == 3,
             "head_nums must contain [Nq,Nk,Nv]")
    q_heads, k_heads, v_heads = (int(item) for item in head_nums)
    _require(0 < q_heads <= 64, "Nq must satisfy 0 < Nq <= 64")
    _require(q_heads == 8 * k_heads, "Nq must equal 8 * Nk")
    _require(k_heads == v_heads, "Nk must equal Nv")

    in_layout = "TND" if layout_qkv in (None, "") else str(layout_qkv).upper()
    out_layout = "NTD" if layout_q_out in (None, "") else str(layout_q_out).upper()
    _require(in_layout in ("TND", "NTD"), "layout_qkv must be TND or NTD")
    _require(out_layout in ("TND", "NTD"), "layout_q_out must be TND or NTD")
    _require(not (in_layout == "NTD" and out_layout == "TND"),
             "NTD input with TND output is unsupported")

    qkv_array = np.asarray(qkv)
    _require(qkv_array.ndim == 3, "qkv must be rank 3")
    token_num = int(qkv_array.shape[0] if in_layout == "TND" else qkv_array.shape[1])
    total_heads = int(qkv_array.shape[1] if in_layout == "TND" else qkv_array.shape[0])
    head_dim = int(qkv_array.shape[2])
    _require(head_dim == 128, "D must be 128")
    _require(total_heads == q_heads + k_heads + v_heads,
             "qkv head dimension does not match head_nums")
    _require(np.asarray(q_gamma).shape == (head_dim,), "q_gamma must have shape [D]")
    _require(np.asarray(k_gamma).shape == (head_dim,), "k_gamma must have shape [D]")
    _require(np.asarray(cos_sin).ndim == 2 and np.asarray(cos_sin).shape[1] == head_dim,
             "cos_sin must have shape [MaxSeqLen,D]")
    _require(np.asarray(slot_mapping).shape == (token_num,),
             "slot_mapping must have shape [T]")
    _require(rotation is not None and np.asarray(rotation).shape == (head_dim, head_dim),
             "rotation must have shape [D,D]")

    k_cache_array = np.asarray(k_cache)
    v_cache_array = np.asarray(v_cache)
    k_scale_array = np.asarray(k_scale_cache)
    _require(k_cache_array.ndim == 4, "k_cache must be rank 4")
    block_num, cache_k_heads, block_size, cache_dim = k_cache_array.shape
    _require((cache_k_heads, cache_dim) == (k_heads, head_dim),
             "k_cache must have shape [BlockNum,Nk,BlockSize,D]")
    _require(v_cache_array.shape == (block_num, v_heads, block_size, head_dim),
             "v_cache must have shape [BlockNum,Nv,BlockSize,D]")
    _require(k_scale_array.shape == (block_num, k_heads, block_size, 1),
             "k_scale_cache must have shape [BlockNum,Nk,BlockSize,1]")

    def rms_norm(value, gamma):
        mean_square = np.sum(value * value, axis=-1, keepdims=True) / np.float32(head_dim)
        return value / np.sqrt(mean_square + np.float32(epsilon)) * np.asarray(gamma, dtype=np.float32)

    cos, sin, is_mrope = _select_cos_sin(
        cos_sin, token_num, query_start_loc, seq_lens, mrope_position, mrope_section
    )
    rotation_fp32 = to_bf16(rotation).astype(np.float32)
    dtype_name = _normalize_dtype_name(q_out_dtype)
    if is_mrope:
        _require(str(q_quant_mode) == "NoQuant", "M-RoPE requires q_quant_mode=NoQuant")
        _require("bfloat16" in dtype_name or dtype_name in ("bf16", "27"),
                 "M-RoPE requires BF16 q_out")
        expected_v_scale = (v_heads, head_dim)
        _require(np.asarray(v_scale).shape == expected_v_scale,
                 "M-RoPE v_scale must have shape [Nv,D]")
        v_multiplier = np.asarray(v_scale, dtype=np.float32)[None, :, :]
        q_dtype = BF16_DTYPE
    else:
        _require(str(q_quant_mode) == "PerTokenPerHead",
                 "RoPE requires q_quant_mode=PerTokenPerHead")
        _require("float8_e4m3fn" in dtype_name or dtype_name == "36",
                 "RoPE requires FP8 E4M3FN q_out")
        _require(np.asarray(v_scale).shape == (v_heads,),
                 "RoPE v_scale must have shape [Nv]")
        v_multiplier = np.asarray(v_scale, dtype=np.float32)[None, :, None]
        q_dtype = FP8_DTYPE

    q_shape = (
        (q_heads, token_num, head_dim)
        if out_layout == "NTD"
        else (token_num, q_heads, head_dim)
    )
    q_out = np.empty(q_shape, dtype=q_dtype)
    q_scale = None if is_mrope else np.empty(q_shape[:-1], dtype=np.float32)
    k_cache_out = k_cache_array.copy()
    v_cache_out = v_cache_array.copy()
    k_scale_cache_out = k_scale_array.astype(np.float32, copy=True)
    slots = np.asarray(slot_mapping, dtype=np.int64)
    _require(bool(np.all(slots >= 0) and np.all(slots < block_num * block_size)),
             "slot_mapping exceeds cache capacity")

    # Avoid materializing a full FP32 QKV plus all normalization/RoPE/Cube
    # intermediates.  At T=262143 and Nq=64 those temporaries exceed tens of
    # GiB.  The operator is token-independent before cache scatter, so chunking
    # preserves the exact element-wise formula and output order.
    chunk_tokens = 4096
    for begin in range(0, token_num, chunk_tokens):
        end = min(token_num, begin + chunk_tokens)
        if in_layout == "NTD":
            qkv_chunk = np.transpose(
                qkv_array[:, begin:end, :].astype(np.float32), (1, 0, 2)
            )
        else:
            qkv_chunk = qkv_array[begin:end].astype(np.float32)
        q = qkv_chunk[:, :q_heads, :]
        k = qkv_chunk[:, q_heads:q_heads + k_heads, :]
        v = qkv_chunk[:, q_heads + k_heads:, :]

        q_rope = _apply_rope(rms_norm(q, q_gamma), cos[begin:end], sin[begin:end])
        k_rope = _apply_rope(rms_norm(k, k_gamma), cos[begin:end], sin[begin:end])
        q_rot = to_bf16(q_rope).astype(np.float32) @ rotation_fp32
        k_rot = to_bf16(k_rope).astype(np.float32) @ rotation_fp32

        if is_mrope:
            q_chunk = to_bf16(q_rot)
            k_quant, k_scale = _dynamic_quant(k_rot, "int8")
        else:
            q_chunk, q_scale_chunk = _dynamic_quant(q_rot, "fp8")
            k_quant, k_scale = _dynamic_quant(k_rot, "fp8")
        v_quant = saturating_fp8_cast(v * v_multiplier)

        if out_layout == "NTD":
            q_out[:, begin:end, :] = np.transpose(q_chunk, (1, 0, 2))
            if q_scale is not None:
                q_scale[:, begin:end] = np.transpose(q_scale_chunk, (1, 0))
        else:
            q_out[begin:end] = q_chunk
            if q_scale is not None:
                q_scale[begin:end] = q_scale_chunk

        chunk_slots = slots[begin:end]
        block_ids = chunk_slots // block_size
        block_offsets = chunk_slots % block_size
        k_cache_out[block_ids, :, block_offsets, :] = k_quant
        v_cache_out[block_ids, :, block_offsets, :] = v_quant
        k_scale_cache_out[block_ids, :, block_offsets, 0] = k_scale

    return GoldenResult(q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out)
