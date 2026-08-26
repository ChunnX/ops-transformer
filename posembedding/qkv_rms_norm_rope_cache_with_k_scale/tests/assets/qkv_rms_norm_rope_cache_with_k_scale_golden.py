#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK assets for QkvRmsNormRopeCacheWithKScale.

The numerical source of truth is deliberately independent of ops-test-kit.  All
five frontend contracts below adapt names/return conventions to ``_golden_core``.
"""

import ast
import re

import numpy as np
from en_dtypes import float8_e8m0
from ml_dtypes import bfloat16, float8_e4m3fn


__spec__ = {
    "qkv_rms_norm_rope_cache_with_k_scale": "QkvRmsNormRopeCacheWithKScaleKernelSpec",
    "aclnnQkvRmsNormRopeCacheWithKScale": "QkvRmsNormRopeCacheWithKScaleAclnnSpec",
    "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale":
        "QkvRmsNormRopeCacheWithKScaleTorchSpec",
    "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_":
        "QkvRmsNormRopeCacheWithKScaleTorchInplaceSpec",
    "torch.ops.cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale":
        "QkvRmsNormRopeCacheWithKScaleTorchSpec",
    "torch.ops.cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_":
        "QkvRmsNormRopeCacheWithKScaleTorchInplaceSpec",
}


_D = 128
_HALF_D = 64
_MX_BLOCK = 32
_E4_MAX = np.float32(448.0)
_CHUNK_T = 256

_FULL_CASE_META = re.compile(
    r"__meta_b(?P<batch>\d+)_h(?P<history>[01])_s(?P<seed>\d+)_"
    r"p(?P<slot>linear|reverse|random|affine)_rk(?P<k_pad>\d+)_"
    r"rv(?P<v_pad>\d+)_rs(?P<scale_pad>\d+)$"
)


def _numpy(value):
    """Move a numpy/torch tensor to CPU numpy without losing custom FP8 bits."""
    if value is None or isinstance(value, np.ndarray):
        return value
    try:
        import torch
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            if cpu.dtype == torch.bfloat16:
                return cpu.view(torch.uint16).numpy().view(bfloat16)
            if cpu.dtype == torch.float8_e4m3fn:
                return cpu.view(torch.uint8).numpy().view(float8_e4m3fn)
            if hasattr(torch, "float8_e8m0fnu") and cpu.dtype == torch.float8_e8m0fnu:
                return cpu.view(torch.uint8).numpy().view(float8_e8m0)
            return cpu.numpy()
    except (ImportError, TypeError):
        pass
    return np.asarray(value)


def _copy_into(dst, src):
    """Write deterministic data back into the framework-created input."""
    if dst is None:
        return None
    src = np.asarray(src)
    if isinstance(dst, np.ndarray):
        dst[...] = src.astype(dst.dtype, copy=False)
        return dst
    import torch
    if dst.dtype in (torch.float8_e4m3fn, getattr(torch, "float8_e8m0fnu", torch.uint8)):
        raw = np.ascontiguousarray(src).view(np.uint8)
        dst.view(torch.uint8).copy_(torch.from_numpy(raw).to(dst.device))
    else:
        dst.copy_(torch.from_numpy(src.astype(np.float32, copy=False)).to(device=dst.device, dtype=dst.dtype))
    return dst


def _case_name(kwargs):
    return str(kwargs.get("testcase_name", ""))


def _full_case_meta(testcase_name):
    """Parse the shared ACLNN/PTA full-matrix testcase-name contract."""
    match = _FULL_CASE_META.search(str(testcase_name or ""))
    if match is None:
        return None
    values = match.groupdict()
    return {
        "batch": int(values["batch"]),
        "history": bool(int(values["history"])),
        "seed": int(values["seed"]),
        "slot": values["slot"],
    }


def _write_full_qkv(target, token_num, layout, rng):
    """Fill a full-matrix QKV tensor without allocating a full FP32 copy."""
    for begin in range(0, token_num, _CHUNK_T):
        end = min(begin + _CHUNK_T, token_num)
        view = target[:, begin:end, :] if layout == "NTD" else target[begin:end]
        values = rng.standard_normal(tuple(view.shape), dtype=np.float32)
        _copy_into(view, (values * np.float32(0.45)).astype(bfloat16))


def _customize_common(qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
                      k_cache, v_cache, k_scale_cache, query_start_loc,
                      seq_lens, rotation, v_scale, mrope_position, **kwargs):
    name = _case_name(kwargs)
    full_meta = _full_case_meta(name)
    rng = np.random.default_rng(full_meta["seed"]) if full_meta is not None else None
    qkv_np = _numpy(qkv)
    layout = kwargs.get("layout_qkv", kwargs.get("layoutQkv", "TND")) or "TND"
    t = qkv_np.shape[1] if layout == "NTD" else qkv_np.shape[0]
    total_heads = qkv_np.shape[0] if layout == "NTD" else qkv_np.shape[1]

    d = np.arange(_D, dtype=np.int32)
    if full_meta is not None:
        _write_full_qkv(qkv, t, layout, rng)
    else:
        # Four D32 regions intentionally have separated magnitudes.  This makes
        # a flat D32 scale-order bug observable instead of merely checking shape.
        mag = np.asarray([2.0 ** -5, 2.0 ** -1, 2.0 ** 3, 2.0 ** 7], np.float32)[d // _MX_BLOCK]
        row = (((d * 13 + 7) % 29).astype(np.float32) / 29.0 + 0.5) * mag
        row *= np.where(d & 1, -1.0, 1.0)
        head = (1.0 + (np.arange(total_heads, dtype=np.float32) % 5) / 64.0)[None, :, None]
        for begin in range(0, t, _CHUNK_T):
            end = min(begin + _CHUNK_T, t)
            token = (1.0 + (np.arange(begin, end, dtype=np.float32) % 7) / 32.0)[:, None, None]
            values = (token * head * row[None, None, :]).astype(bfloat16)
            if layout == "NTD":
                _copy_into(qkv[:, begin:end, :], np.swapaxes(values, 0, 1))
            else:
                _copy_into(qkv[begin:end], values)

    # White-box numerical profiles are encoded in the stable testcase name so
    # the same data contract works for Kernel, ACLNN and Torch runners.  Keep
    # these mutations before gamma generation: qkv profiles exercise RMS input
    # propagation, while tiny/saturation below exercise post-RMS MX scaling.
    if "nan_qk" in name or "inf_qk" in name:
        heads = kwargs.get("head_nums", kwargs.get("headNums"))
        qk_heads = int(heads[0]) + int(heads[1])
        special = np.float32(np.nan if "nan_qk" in name else np.inf)
        qkv_special = _numpy(qkv).copy()
        if layout == "NTD":
            qkv_special[:qk_heads, 0, 0] = special
        else:
            qkv_special[0, :qk_heads, 0] = special
        _copy_into(qkv, qkv_special)

    # Lock the PerTokenPerHead zero-row policy without also zeroing V:
    # Q/K produce scale 0 and FP8 zero, while V remains a normal regression path.
    if "zero_qk" in name:
        heads = kwargs.get("head_nums", kwargs.get("headNums"))
        qk_heads = int(heads[0]) + int(heads[1])
        if layout == "NTD":
            _copy_into(qkv[:qk_heads, :, :], np.zeros((qk_heads, t, _D), dtype=bfloat16))
        else:
            _copy_into(qkv[:, :qk_heads, :], np.zeros((t, qk_heads, _D), dtype=bfloat16))

    # These two profiles make the exact E8M0 bit algorithm observable.  All-one
    # Q/K rows with epsilon=0 have RMS exactly one, identity RoPE keeps gamma
    # unchanged, and each D32 block therefore reaches the intended raw FP32
    # boundary without depending on random input rounding.
    if "scale_edge_qk" in name or "max_finite_qk" in name:
        heads = kwargs.get("head_nums", kwargs.get("headNums"))
        qk_heads = int(heads[0]) + int(heads[1])
        if layout == "NTD":
            _copy_into(qkv[:qk_heads, :, :], np.ones((qk_heads, t, _D), dtype=bfloat16))
        else:
            _copy_into(qkv[:, :qk_heads, :], np.ones((t, qk_heads, _D), dtype=bfloat16))

    gamma = np.ones((_D,), np.float32)
    if full_meta is not None:
        gamma = rng.uniform(0.75, 1.25, size=(_D,)).astype(np.float32)
    elif "scale_order" in name:
        gamma *= np.asarray([2.0 ** -6, 2.0 ** -2, 2.0 ** 2, 2.0 ** 6], np.float32)[d // _MX_BLOCK]
    elif "scale_edge_qk" in name:
        edge = np.asarray([0x04600000, 0x04600001, 0x04600002, 0x3F800000], np.uint32).view(np.float32)
        gamma = edge[d // _MX_BLOCK]
    elif "max_finite_qk" in name:
        gamma.fill(np.asarray([0x7F7FFFFF], np.uint32).view(np.float32)[0])
    elif "tiny_qk" in name:
        gamma.fill(np.float32(2.0 ** -140))
    elif "saturation_qk" in name:
        gamma.fill(np.float32(2.0 ** 120))
    else:
        gamma *= 0.75 + (d % 17).astype(np.float32) / 32.0
    _copy_into(q_gamma, gamma)
    k_gamma_data = (
        rng.uniform(0.75, 1.25, size=(_D,)).astype(np.float32)
        if full_meta is not None else gamma[::-1].copy()
    )
    _copy_into(k_gamma, k_gamma_data)

    table = _numpy(cos_sin)
    position_count = table.shape[0]
    pos = np.arange(position_count, dtype=np.float32)[:, None]
    lane = np.arange(_HALF_D, dtype=np.float32)[None, :]
    angle = pos * (lane + 1.0) / 65536.0
    cos_table = np.concatenate((np.cos(angle), np.sin(angle)), axis=1).astype(np.float32)
    if "scale_order" in name or "scale_edge_qk" in name or "max_finite_qk" in name:
        cos_table[:, :_HALF_D] = 1.0
    cos_table[:, _HALF_D:] = 0.0
    _copy_into(cos_sin, cos_table)

    capacity = int(np.prod(_numpy(k_cache).shape[:1] + _numpy(k_cache).shape[2:3]))
    token_ids = np.arange(t, dtype=np.int64)
    if full_meta is None:
        # A cyclic shift followed by reversal is a permutation for every
        # capacity, including non-power-of-two stride/tail cases.
        slots = ((token_ids + 3) % capacity)[::-1].astype(np.int32)
    elif full_meta["slot"] == "linear":
        slots = token_ids.astype(np.int32)
    elif full_meta["slot"] == "reverse":
        slots = np.arange(capacity - 1, capacity - t - 1, -1, dtype=np.int64).astype(np.int32)
    elif full_meta["slot"] == "random":
        slots = rng.permutation(capacity)[:t].astype(np.int32)
    else:
        slots = ((token_ids * (capacity - 1) + 17) % capacity).astype(np.int32)
    block_size = _numpy(k_cache).shape[2]
    def _move_slots_to_front(values):
        nonlocal slots
        values = [int(value) for value in values if int(value) < capacity]
        tail = [int(value) for value in slots if int(value) not in values]
        slots = np.asarray(values + tail, np.int32)[:t]

    if "slot_block_begin" in name:
        _move_slots_to_front([0])
    elif "slot_block_end" in name:
        _move_slots_to_front([block_size - 1])
    elif "slot_cross_block" in name and t >= 2:
        _move_slots_to_front([block_size - 1, block_size])
    _copy_into(slot_mapping, slots)

    if query_start_loc is not None:
        if full_meta is not None:
            batch = int(_numpy(seq_lens).shape[0])
            base, remainder = divmod(t, batch)
            current = np.asarray(
                [base + (index < remainder) for index in range(batch)], np.int32
            )
            qsl = np.concatenate(([0], np.cumsum(current))).astype(np.int32)
        elif "batch1" in name:
            qsl = np.asarray([0, t], np.int32)
        elif "batch_many" in name:
            qsl = np.asarray([0, 1, 2, 4, t], np.int32)
        else:
            qsl = np.asarray([0, t // 2, t], np.int32)
        _copy_into(query_start_loc, qsl)
    if seq_lens is not None:
        current = np.diff(qsl).astype(np.int32)
        if full_meta is not None and full_meta["history"]:
            batch = current.size
            current += 257 + 113 * np.arange(batch, dtype=np.int32) + full_meta["seed"] % 97
        _copy_into(seq_lens, current)
    if rotation is not None:
        rotation_data = np.eye(_D, dtype=np.float32)
        if full_meta is not None:
            rotation_data += rng.normal(0.0, 0.01, size=rotation_data.shape).astype(np.float32)
        _copy_into(rotation, rotation_data.astype(bfloat16))
    if v_scale is not None:
        vs = _numpy(v_scale)
        vals = np.ones(vs.shape, np.float32)
        if full_meta is not None:
            vals = rng.uniform(0.5, 1.5, size=vs.shape).astype(np.float32)
        elif vals.ndim == 2:
            vals *= 0.5 + np.arange(vals.shape[-1], dtype=np.float32)[None, :] / 256.0
        else:
            vals *= 0.75
        _copy_into(v_scale, vals)
    if mrope_position is not None:
        positions = np.empty((t, 3), np.int32)
        positions[:, 0] = np.arange(t, dtype=np.int64) % position_count
        positions[:, 1] = (np.arange(t, dtype=np.int64) * 3 + 1) % position_count
        positions[:, 2] = (np.arange(t, dtype=np.int64) * 7 + 2) % position_count
        if "position_1024" in name:
            boundary_positions = np.asarray(
                [[1023, 1024, 1025],
                 [1024, 1025, 1023],
                 [1025, 1023, 1024]],
                np.int32,
            )
            positions[:] = boundary_positions[np.arange(t) % len(boundary_positions)]
        _copy_into(mrope_position, positions)

    # Full-matrix cases retain TTK's cache initialization.  Re-materializing a
    # largest-case cache here would require multi-GiB host temporaries.
    if full_meta is None:
        # Non-zero, position-dependent sentinels prove untouched cache bytes are
        # preserved by both in-place and functional routes.
        for index, cache in enumerate((k_cache, v_cache, k_scale_cache)):
            arr = _numpy(cache)
            if arr.dtype.itemsize == 1:
                raw = ((np.arange(arr.size, dtype=np.uint64) * (17 + index * 2) + 23) & 0xFF).astype(np.uint8)
                if arr.dtype == float8_e8m0:
                    raw = (raw % 253).astype(np.uint8)
                values = raw.reshape(arr.shape).view(arr.dtype)
            else:
                values = (((np.arange(arr.size, dtype=np.int64) * (7 + index)) % 31) - 15)
                values = (values.astype(np.float32) / 8.0).reshape(arr.shape).astype(arr.dtype)
            _copy_into(cache, values)

    return (qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
            k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
            mrope_position)


def _mrope_axes(section):
    section = list(section or (0, 0, 0))
    h = int(section[1]) if len(section) > 1 else 0
    w = int(section[2]) if len(section) > 2 else 0
    lane = np.arange(_HALF_D, dtype=np.int32)
    repeat = lane // 3
    axis_in_group = lane % 3
    return np.where((axis_in_group == 1) & (repeat < h), 1,
                    np.where((axis_in_group == 2) & (repeat < w), 2, 0))


def _rms_rope(x, gamma, cos_sin, positions, section, epsilon):
    x = x.astype(np.float32)
    gamma = gamma.astype(np.float32)
    norm = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(epsilon))
    norm *= gamma
    axes = _mrope_axes(section)
    selected_pos = positions[:, axes]
    lanes = np.arange(_HALF_D, dtype=np.int64)[None, :]
    cos = cos_sin[selected_pos, lanes]
    sin = cos_sin[selected_pos, lanes + _HALF_D]
    low, high = norm[..., :_HALF_D], norm[..., _HALF_D:]
    # cos/sin are token x lane; broadcast across heads.
    return np.concatenate((low * cos[:, None, :] - high * sin[:, None, :],
                           high * cos[:, None, :] + low * sin[:, None, :]), axis=-1)


def _mx_quant_cublas(x):
    """cuBLAS MXFP8 semantics: D32 blocks with E4M3FN data and E8M0 power-of-two scales."""
    shape = x.shape
    blocks = x.astype(np.float32).reshape(*shape[:-1], shape[-1] // _MX_BLOCK, _MX_BLOCK)
    amax = np.max(np.abs(blocks), axis=-1)
    nonfinite_amax = ~np.isfinite(amax)
    s = (amax / _E4_MAX).astype(np.float32)
    bits = s.view(np.uint32)
    biased = ((bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int16)
    mantissa = bits & np.uint32(0x007FFFFF)
    round_up = ((biased > 0) & (biased < 254) & (mantissa > 0)) | ((biased == 0) & (mantissa > 2 ** 22))
    biased = biased + round_up.astype(np.int16)
    exponent = biased - 127
    exponent = np.where(amax == 0, -127, exponent)
    exponent = np.maximum(exponent, -127)
    exponent = np.where(exponent > 127, 128, exponent).astype(np.int16)
    scale = np.exp2(np.minimum(exponent, 127).astype(np.float32))[..., None]
    # The cuBLAS MXFP8 scale rule maps a non-finite D32 amax to E8M0 raw
    # 0xFF and uses a NaN reciprocal.  Consequently every E4M3 value in that
    # D32 block is NaN, including lanes whose original input was finite.
    scale = np.where(nonfinite_amax[..., None], np.float32(np.nan), scale)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quant_blocks = (blocks / scale).astype(float8_e4m3fn)
    # DAV_3510 F32->E4M3FN with SatMode::SAT and CTRL[50]=0 writes raw 0x00
    # for the NaN produced by a non-finite D32 reciprocal.  ml_dtypes instead
    # writes an E4M3 NaN code, so normalize the complete affected D32 block to
    # the device cast result while retaining the E8M0 invalid scale below.
    quant_raw = quant_blocks.view(np.uint8)
    quant_raw = np.where(nonfinite_amax[..., None], np.uint8(0), quant_raw)
    quant = quant_raw.reshape(shape).view(float8_e4m3fn)
    scale_raw = np.where(nonfinite_amax | (exponent > 127), 255, exponent + 127).astype(np.uint8)
    scale_e8 = scale_raw.view(float8_e8m0).reshape(*shape[:-1], shape[-1] // 32)
    return quant, scale_e8


def _per_token_per_head_dynamic(x, dst_dtype):
    max_value = np.float32(127.0 if dst_dtype == np.int8 else 448.0)
    amax = np.max(np.abs(x), axis=-1)
    scale = (amax / max_value).astype(np.float32)
    # Match DynamicQuantD128 exactly: there is no zero-row guard.  Therefore
    # amax==0 keeps scale==0 and normalizes every zero element as 0/0 (NaN).
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = x / scale[..., None]
        if dst_dtype == np.int8:
            q = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
        else:
            q = normalized.astype(dst_dtype)
            # The kernel's SAT F32->E4M3FN cast writes 0x00 for this NaN when
            # DAV_3510 CTRL[50] is 0, unlike ml_dtypes, which writes 0xFF.
            # Model that device cast while retaining scale 0 and the 0/0.
            q = np.where((amax == 0)[..., None], np.zeros((), dtype=dst_dtype), q)
    return q, scale


def _golden_core(qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
                 k_cache, v_cache, k_scale_cache, query_start_loc=None,
                 seq_lens=None, rotation=None, v_scale=None,
                 mrope_position=None, *, head_nums, layout_qkv="TND",
                 layout_q_out="TND", epsilon=1e-6, mrope_section=None,
                 q_quant_mode="PerTokenPerHead",
                 k_quant_mode="PerTokenPerHead", q_out_dtype=None, **kwargs):
    qkv = _numpy(qkv)
    q_gamma, k_gamma, cos_sin = map(_numpy, (q_gamma, k_gamma, cos_sin))
    slots = _numpy(slot_mapping).astype(np.int64)
    k_out, v_out, ks_out = (_numpy(k_cache).copy(), _numpy(v_cache).copy(), _numpy(k_scale_cache).copy())
    layout_qkv = layout_qkv or "TND"
    layout_q_out = layout_q_out or "TND"
    qkv_tnd = np.swapaxes(qkv, 0, 1) if layout_qkv == "NTD" else qkv
    t, _, d = qkv_tnd.shape
    nq, nk, nv = [int(x) for x in head_nums]
    assert d == _D and nq + nk + nv == qkv_tnd.shape[1]
    new_mx = q_quant_mode == "MxQuant" and k_quant_mode == "MxQuant"
    is_mrope = mrope_position is not None
    if is_mrope:
        positions = _numpy(mrope_position).astype(np.int64)
    else:
        # RoPE position mapping; cases intentionally use two balanced sequences.
        positions = np.zeros((t, 3), np.int64)
        qsl, lens = _numpy(query_start_loc), _numpy(seq_lens)
        for batch in range(len(lens)):
            begin, end = int(qsl[batch]), int(qsl[batch + 1])
            first = int(lens[batch]) - (end - begin)
            positions[begin:end, :] = np.arange(first, int(lens[batch]))[:, None]

    q_shape = (t, nq, d)
    if new_mx:
        q_out = np.empty(q_shape, dtype=float8_e4m3fn)
        q_scale = np.empty((t, nq, d // 32), dtype=float8_e8m0)
    elif is_mrope:
        q_out, q_scale = np.empty(q_shape, dtype=bfloat16), None
    else:
        q_out = np.empty(q_shape, dtype=float8_e4m3fn)
        q_scale = np.empty((t, nq), dtype=np.float32)

    block_size = k_out.shape[2]
    for begin in range(0, t, _CHUNK_T):
        end = min(begin + _CHUNK_T, t)
        tile = qkv_tnd[begin:end]
        pos = positions[begin:end]
        q = _rms_rope(tile[:, :nq], q_gamma, cos_sin, pos, mrope_section, epsilon)
        k = _rms_rope(tile[:, nq:nq + nk], k_gamma, cos_sin, pos, mrope_section, epsilon)
        if not new_mx:
            q = q.astype(bfloat16).astype(np.float32)
            k = k.astype(bfloat16).astype(np.float32)
            if rotation is not None:
                rot = _numpy(rotation).astype(np.float32)
                q = np.matmul(q, rot)
                k = np.matmul(k, rot)
        if new_mx:
            q_quant, q_sc = _mx_quant_cublas(q)
            k_quant, k_sc = _mx_quant_cublas(k)
            q_out[begin:end], q_scale[begin:end] = q_quant, q_sc
        elif is_mrope:
            q_out[begin:end] = q.astype(bfloat16)
            k_quant, k_sc = _per_token_per_head_dynamic(k, np.int8)
        else:
            q_quant, q_sc = _per_token_per_head_dynamic(q, float8_e4m3fn)
            k_quant, k_sc = _per_token_per_head_dynamic(k, float8_e4m3fn)
            q_out[begin:end], q_scale[begin:end] = q_quant, q_sc

        v = tile[:, nq + nk:nq + nk + nv].astype(np.float32)
        vs = _numpy(v_scale).astype(np.float32)
        v = v * (vs[None, :, :] if vs.ndim == 2 else vs[None, :, None])
        v_quant = v.astype(float8_e4m3fn)
        tile_slots = slots[begin:end]
        blocks, offsets = tile_slots // block_size, tile_slots % block_size
        k_out[blocks, :, offsets, :] = k_quant
        v_out[blocks, :, offsets, :] = v_quant
        if new_mx:
            ks_out[blocks, :, offsets, :] = k_sc
        else:
            target = ks_out[blocks, :, offsets]
            ks_out[blocks, :, offsets] = k_sc.reshape(target.shape)

    if layout_q_out == "NTD":
        q_out = np.swapaxes(q_out, 0, 1)
        if q_scale is not None:
            q_scale = np.swapaxes(q_scale, 0, 1)
    return [q_out, q_scale, k_out, v_out, ks_out]


def _raw(arr):
    arr = _numpy(arr)
    if arr is None:
        return None
    return np.ascontiguousarray(arr).view(np.uint8)


def _compare_one(actual, expected, exact=False):
    if actual is None or expected is None:
        ok = actual is None and expected is None
        return {"pass": ok, "precision": "EXACT" if ok else "NONE_MISMATCH",
                "error_info": None if ok else "only one side is None"}
    a, e = _numpy(actual), _numpy(expected)
    if a.shape != e.shape:
        return {"pass": False, "precision": "SHAPE_MISMATCH",
                "error_info": f"actual {a.shape}, expected {e.shape}"}
    if exact or a.dtype == float8_e8m0 or e.dtype == float8_e8m0:
        mismatch = np.count_nonzero(_raw(a) != _raw(e))
        ok = mismatch == 0
        return {"pass": ok, "precision": 100.0 * (1.0 - mismatch / max(1, _raw(e).size)),
                "error_info": None if ok else f"{mismatch} raw-byte mismatches"}
    if a.dtype == float8_e4m3fn or e.dtype == float8_e4m3fn:
        # One adjacent E4M3 code is allowed for implementation rounding order.
        ar, er = _raw(a), _raw(e)
        same_sign = (ar & 0x80) == (er & 0x80)
        both_zero = ((ar & 0x7F) == 0) & ((er & 0x7F) == 0)
        diff = np.abs((ar & 0x7F).astype(np.int16) - (er & 0x7F).astype(np.int16))
        within = (same_sign & (diff <= 1)) | both_zero
        ok = bool(np.all(within))
        return {"pass": ok, "precision": 100.0 * np.count_nonzero(within) / max(1, diff.size),
                "error_info": None if ok else f"max raw-code delta {int(diff.max())}"}
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(e.dtype, np.integer):
        diff = np.abs(a.astype(np.int64) - e.astype(np.int64))
        ok = bool(np.all(diff <= 1))
        return {"pass": ok, "precision": 100.0 * np.count_nonzero(diff <= 1) / max(1, diff.size),
                "error_info": None if ok else f"max integer delta {int(diff.max())}"}
    af, ef = a.astype(np.float32), e.astype(np.float32)
    tol = 1e-2 if a.dtype == bfloat16 or e.dtype == bfloat16 else 1e-3
    ok = bool(np.allclose(af, ef, rtol=tol, atol=tol, equal_nan=True))
    return {"pass": ok, "precision": "ISCLOSE",
            "error_info": None if ok else f"max abs error {float(np.max(np.abs(af - ef)))}"}


def _compare(*values, compare_context):
    half = len(values) // 2
    outputs, goldens = values[:half], values[half:]
    api_name = str(compare_context.api_name or "")
    results = []
    input_tensors = compare_context.input_tensors or ()
    attrs = compare_context.attributes or {}
    mrope_qscale_placeholder = (
        api_name == "qkv_rms_norm_rope_cache_with_k_scale" and
        attrs.get("q_quant_mode") == "NoQuant" and
        bool(attrs.get("mrope_section"))
    )
    cache_indexes = (2, 3, 4)
    slots = _numpy(input_tensors[4]).astype(np.int64) if len(input_tensors) > 4 else None
    for index, (actual, expected) in enumerate(zip(outputs, goldens)):
        if mrope_qscale_placeholder and index == 1:
            # M-RoPE keeps a fixed kernel ABI qScale argument, but the
            # device branch neither binds nor writes it.  It is not a public
            # result and must not be compared against a fabricated value.
            results.append({"pass": True, "precision": "NOT_APPLICABLE_PLACEHOLDER"})
            continue
        result = _compare_one(actual, expected, exact=index >= 5)
        # Cache scatter is a two-part contract: touched entries follow the
        # dtype-specific rule above; every untouched byte is exactly preserved.
        if index in cache_indexes and slots is not None and actual is not None and expected is not None:
            a, e = _numpy(actual), _numpy(expected)
            touched = np.zeros(a.shape[:3], dtype=bool)
            block_size = a.shape[2]
            touched[slots // block_size, :, slots % block_size] = True
            ar, er = _raw(a), _raw(e)
            touched_raw = touched[(...,) + (None,) * (ar.ndim - 3)]
            untouched_mismatch = np.count_nonzero(np.where(touched_raw, 0, ar) != np.where(touched_raw, 0, er))
            if untouched_mismatch:
                result["pass"] = False
                result["error_info"] = f"{untouched_mismatch} untouched-cache byte mismatches"
        results.append(result)
    return results


def _kernel_golden(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache,
                   v_cache, k_scale_cache, query_start_loc=None, seq_lens=None,
                   rotation=None, v_scale=None, mrope_position=None, **attrs):
    result = _golden_core(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache,
                          v_cache, k_scale_cache, query_start_loc, seq_lens,
                          rotation, v_scale, mrope_position, **attrs)
    # Kernel ABI keeps the q_scale slot materialized for M-RoPE.
    if result[1] is None:
        result[1] = np.zeros(result[0].shape[:2], np.float32)
    return result


def _aclnn_golden(qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef,
                  vCacheRef, kScaleCacheRef, queryStartLocOptional,
                  seqLensOptional, rotationOptional, vScaleOptional,
                  mropePositionOptional, headNums, layoutQkv, layoutQOut,
                  epsilon, mropeSectionOptional, qQuantMode, kQuantMode,
                  qOut, qScaleOptional, **kwargs):
    # ACLNN custom Golden receives the complete GetWorkspaceSize positional
    # contract.  The final two tensors are pure outputs and are not inputs to
    # the numerical reference.
    del qOut, qScaleOptional
    result = _golden_core(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
        kScaleCacheRef, queryStartLocOptional, seqLensOptional, rotationOptional,
        vScaleOptional, mropePositionOptional, head_nums=headNums,
        layout_qkv=layoutQkv, layout_q_out=layoutQOut, epsilon=epsilon,
        mrope_section=mropeSectionOptional, q_quant_mode=qQuantMode,
        k_quant_mode=kQuantMode, **kwargs)
    return [x for x in result if x is not None]


def _torch_golden(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache,
                  v_cache, k_scale_cache, query_start_loc=None, seq_lens=None,
                  head_nums=None, *, rotation=None, v_scale=None, layout_qkv="TND",
                  layout_q_out="TND", epsilon=1e-6, mrope_position=None,
                  mrope_section=None, q_quant_mode="PerTokenPerHead",
                  k_quant_mode="PerTokenPerHead", **kwargs):
    # ``head_nums`` is positional in the public Torch schema.  Keep the
    # Golden adapter aligned with that schema so TTK's parameter plan can call
    # the same function for both functional and in-place APIs.
    result = _golden_core(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, head_nums=head_nums, layout_qkv=layout_qkv,
        layout_q_out=layout_q_out, epsilon=epsilon, mrope_section=mrope_section,
        q_quant_mode=q_quant_mode, k_quant_mode=k_quant_mode, **kwargs)
    # The functional API returns cloned K/V/KScale caches as outputs and does
    # not mutate caller-visible cache inputs.  Do not ask TTK to append or
    # restore those inputs through ``inplace_input_indexes``.
    return result


def _torch_inplace_golden(*args, **kwargs):
    result = _torch_golden(*args, **kwargs)
    # In-place API returns only q/q_scale; TTK appends mutated cache inputs.
    return result


def _kernel_customize(qkv, q_gamma, k_gamma, cos_sin, slot_mapping,
                      k_cache, v_cache, k_scale_cache, query_start_loc,
                      seq_lens, rotation, v_scale, mrope_position, **attrs):
    """Adapt the 13-input Kernel TestSpec contract to the shared generator."""
    return _customize_common(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, **attrs)


def _aclnn_customize(qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef,
                     vCacheRef, kScaleCacheRef, queryStartLocOptional,
                     seqLensOptional, rotationOptional, vScaleOptional,
                     mropePositionOptional, headNums, layoutQkv, layoutQOut,
                     epsilon, mropeSectionOptional, qQuantMode, kQuantMode,
                     qOut, qScaleOptional, **kwargs):
    """Adapt the 22-position ACLNN GetWorkspaceSize parameter contract."""
    del qOut, qScaleOptional
    return _customize_common(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
        kScaleCacheRef, queryStartLocOptional, seqLensOptional,
        rotationOptional, vScaleOptional, mropePositionOptional,
        head_nums=headNums, layout_qkv=layoutQkv, layout_q_out=layoutQOut,
        epsilon=epsilon, mrope_section=mropeSectionOptional,
        q_quant_mode=qQuantMode, k_quant_mode=kQuantMode, **kwargs)


def _torch_customize(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache,
                     v_cache, k_scale_cache, query_start_loc, seq_lens,
                     head_nums, *, rotation=None, v_scale=None,
                     layout_qkv="TND", layout_q_out="TND", epsilon=1e-6,
                     mrope_position=None, mrope_section=None,
                     q_quant_mode="PerTokenPerHead",
                     k_quant_mode="PerTokenPerHead", q_out_dtype=None,
                     **kwargs):
    """Adapt the Torch schema's positional and keyword-only parameters."""
    del q_out_dtype
    return _customize_common(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, head_nums=head_nums, layout_qkv=layout_qkv,
        layout_q_out=layout_q_out, epsilon=epsilon,
        mrope_section=mrope_section, q_quant_mode=q_quant_mode,
        k_quant_mode=k_quant_mode, **kwargs)


class QkvRmsNormRopeCacheWithKScaleKernelSpec:
    golden = _kernel_golden
    customize_inputs = _kernel_customize
    compare = _compare


class QkvRmsNormRopeCacheWithKScaleAclnnSpec:
    golden = _aclnn_golden
    customize_inputs = _aclnn_customize
    compare = _compare


class QkvRmsNormRopeCacheWithKScaleTorchSpec:
    golden = _torch_golden
    customize_inputs = _torch_customize
    compare = _compare


class QkvRmsNormRopeCacheWithKScaleTorchInplaceSpec:
    golden = _torch_inplace_golden
    customize_inputs = _torch_customize
    compare = _compare
