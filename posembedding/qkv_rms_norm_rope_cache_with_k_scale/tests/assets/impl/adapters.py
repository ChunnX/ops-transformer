#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""Golden adapters for Kernel, ACLNN and PTA E2E paths."""

from __future__ import annotations

import numpy as np

from .numpy_golden import BF16_DTYPE, FP8_DTYPE, qkv_rms_norm_rope_cache_with_k_scale_numpy


def to_numpy(value):
    """Convert a CPU torch tensor to NumPy without losing BF16/FP8 storage."""

    if value is None or isinstance(value, np.ndarray):
        return value
    if not hasattr(value, "detach"):
        return np.asarray(value)
    tensor = value.detach().cpu().contiguous()
    dtype_name = str(tensor.dtype)
    if "bfloat16" in dtype_name:
        return tensor.view(-1).view(__import__("torch").uint16).numpy().view(BF16_DTYPE).reshape(tensor.shape)
    if "float8_e4m3fn" in dtype_name:
        return tensor.view(-1).view(__import__("torch").uint8).numpy().view(FP8_DTYPE).reshape(tensor.shape)
    return tensor.numpy()


def _numpy_result(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
    mrope_position, *, head_nums, layout_qkv, layout_q_out, epsilon,
    mrope_section, q_quant_mode, q_out_dtype,
):
    return qkv_rms_norm_rope_cache_with_k_scale_numpy(
        to_numpy(qkv), to_numpy(q_gamma), to_numpy(k_gamma), to_numpy(cos_sin),
        to_numpy(slot_mapping), to_numpy(k_cache), to_numpy(v_cache),
        to_numpy(k_scale_cache), to_numpy(query_start_loc), to_numpy(seq_lens),
        to_numpy(rotation), to_numpy(v_scale), to_numpy(mrope_position),
        head_nums=head_nums, layout_qkv=layout_qkv, layout_q_out=layout_q_out,
        epsilon=epsilon, mrope_section=mrope_section,
        q_quant_mode=q_quant_mode, q_out_dtype=q_out_dtype,
    )


def _meaningful_outputs(result):
    values = [result.q_out]
    if result.q_scale is not None:
        values.append(result.q_scale)
    values.extend((result.k_cache, result.v_cache, result.k_scale_cache))
    return values


def kernel_golden(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc=None, seq_lens=None, rotation=None,
    v_scale=None, mrope_position=None, *, head_nums, layout_qkv="TND",
    layout_q_out="NTD", epsilon=1e-6, mrope_section=None,
    q_quant_mode="PerTokenPerHead", q_out_dtype=36, **kwargs,
):
    """Kernel adapter; argument order follows the operator def.cpp."""

    result = _numpy_result(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, head_nums=head_nums, layout_qkv=layout_qkv,
        layout_q_out=layout_q_out, epsilon=epsilon,
        mrope_section=mrope_section, q_quant_mode=q_quant_mode,
        q_out_dtype=q_out_dtype,
    )
    q_scale = result.q_scale
    if q_scale is None:
        # The raw Kernel ABI always has this output, but M-RoPE does not write
        # it.  A shaped placeholder keeps TTK hooks enabled; compare.py ignores
        # this ABI-only slot explicitly.
        q_scale = np.zeros(result.q_out.shape[:-1], dtype=np.float32)
    return [result.q_out, q_scale, result.k_cache, result.v_cache, result.k_scale_cache]


def aclnn_golden(
    qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
    kScaleCacheRef, queryStartLocOptional=None, seqLensOptional=None,
    rotationOptional=None, vScaleOptional=None, mropePositionOptional=None,
    headNums=None, layoutQkv="TND", layoutQOut="NTD", epsilon=1e-6,
    mropeSectionOptional=None, qQuantMode="PerTokenPerHead", qOut=None,
    qScaleOptional=None, **kwargs,
):
    """ACLNN adapter; argument order follows GetWorkspaceSize."""

    q_out_dtype = 27 if str(qQuantMode) == "NoQuant" else 36
    result = _numpy_result(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
        kScaleCacheRef, queryStartLocOptional, seqLensOptional,
        rotationOptional, vScaleOptional, mropePositionOptional,
        head_nums=headNums, layout_qkv=layoutQkv, layout_q_out=layoutQOut,
        epsilon=epsilon, mrope_section=mropeSectionOptional,
        q_quant_mode=qQuantMode, q_out_dtype=q_out_dtype,
    )
    return _meaningful_outputs(result)

def pta_functional_golden(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc, seq_lens, head_nums, *, rotation=None,
    v_scale=None, layout_qkv="TND", layout_q_out="NTD", epsilon=1e-6,
    mrope_position=None, mrope_section=None,
    q_quant_mode="PerTokenPerHead", q_out_dtype=None, **kwargs,
):
    """Golden for the PTA functional wrapper, including no-mutation checks."""

    dtype = 27 if str(q_quant_mode) == "NoQuant" else (36 if q_out_dtype is None else q_out_dtype)
    result = _numpy_result(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, head_nums=head_nums, layout_qkv=layout_qkv,
        layout_q_out=layout_q_out, epsilon=epsilon,
        mrope_section=mrope_section, q_quant_mode=q_quant_mode,
        q_out_dtype=dtype,
    )
    # TTK appends tensors named by inplace_input_indexes after the function's
    # returned outputs.  For the functional API they must remain unchanged.
    return _meaningful_outputs(result) + [
        to_numpy(k_cache).copy(), to_numpy(v_cache).copy(),
        to_numpy(k_scale_cache).copy(),
    ]


def pta_inplace_golden(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc, seq_lens, head_nums, *, rotation=None,
    v_scale=None, layout_qkv="TND", layout_q_out="NTD", epsilon=1e-6,
    mrope_position=None, mrope_section=None,
    q_quant_mode="PerTokenPerHead", q_out_dtype=None, **kwargs,
):
    """Golden for the PTA in-place wrapper and its three appended caches."""

    dtype = 27 if str(q_quant_mode) == "NoQuant" else (36 if q_out_dtype is None else q_out_dtype)
    result = _numpy_result(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, head_nums=head_nums, layout_qkv=layout_qkv,
        layout_q_out=layout_q_out, epsilon=epsilon,
        mrope_section=mrope_section, q_quant_mode=q_quant_mode,
        q_out_dtype=dtype,
    )
    return _meaningful_outputs(result)
