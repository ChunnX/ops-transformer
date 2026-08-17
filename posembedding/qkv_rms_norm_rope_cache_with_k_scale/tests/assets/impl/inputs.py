#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""Deterministic, contract-valid input generation for all TTK paths."""

from __future__ import annotations

import hashlib

import numpy as np

from .adapters import to_numpy
from .numpy_golden import BF16_DTYPE, FP8_DTYPE, saturating_fp8_cast, to_bf16


def _seed(testcase_name) -> int:
    digest = hashlib.sha256(str(testcase_name or "qkv").encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _write_back(target, value):
    if target is None:
        return None
    value = np.asarray(value)
    if isinstance(target, np.ndarray):
        target[...] = value.astype(target.dtype, copy=False)
        return target

    import torch

    dtype_name = str(target.dtype)
    if "bfloat16" in dtype_name:
        source = torch.from_numpy(
            np.ascontiguousarray(value.astype(BF16_DTYPE)).view(np.uint16)
        ).view(torch.bfloat16)
    elif "float8_e4m3fn" in dtype_name:
        source = torch.from_numpy(
            np.ascontiguousarray(value.astype(FP8_DTYPE)).view(np.uint8)
        ).view(torch.float8_e4m3fn)
    else:
        source = torch.from_numpy(np.ascontiguousarray(value)).to(dtype=target.dtype)
    target.copy_(source.to(target.device).reshape(target.shape))
    return target


def _prepare(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
    mrope_position, *, testcase_name=None,
):
    rng = np.random.default_rng(_seed(testcase_name))
    qkv_shape = tuple(qkv.shape)
    token_num = int(slot_mapping.shape[0])
    head_dim = int(qkv_shape[-1])
    block_num, _, block_size, _ = tuple(k_cache.shape)

    qkv_data = to_bf16(rng.normal(0.0, 0.45, size=qkv_shape).astype(np.float32))
    q_gamma_data = rng.uniform(0.75, 1.25, size=q_gamma.shape).astype(np.float32)
    k_gamma_data = rng.uniform(0.75, 1.25, size=k_gamma.shape).astype(np.float32)

    rows = int(cos_sin.shape[0])
    half = head_dim // 2
    positions = np.arange(rows, dtype=np.float32)[:, None]
    frequencies = np.exp(-np.arange(half, dtype=np.float32)[None, :] / np.float32(half))
    angles = positions * frequencies * np.float32(0.03125)
    cos_sin_data = np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)

    capacity = block_num * block_size
    slots = ((np.arange(token_num, dtype=np.int64) * 7 + 3) % capacity).astype(np.int32)
    if np.unique(slots).size != token_num:
        slots = rng.choice(capacity, size=token_num, replace=False).astype(np.int32)

    if np.dtype(to_numpy(k_cache).dtype) == np.dtype(np.int8):
        k_cache_data = rng.integers(-31, 32, size=k_cache.shape, dtype=np.int8)
    else:
        k_cache_data = saturating_fp8_cast(
            rng.uniform(-4.0, 4.0, size=k_cache.shape).astype(np.float32)
        )
    v_cache_data = saturating_fp8_cast(
        rng.uniform(-4.0, 4.0, size=v_cache.shape).astype(np.float32)
    )
    k_scale_data = rng.uniform(0.01, 0.5, size=k_scale_cache.shape).astype(np.float32)

    if query_start_loc is not None:
        batch = int(seq_lens.shape[0])
        if batch == 1:
            current = [token_num]
        elif batch == 2:
            first = max(1, token_num // 3)
            current = [first, token_num - first]
        else:
            base, remainder = divmod(token_num, batch)
            current = [base + (index < remainder) for index in range(batch)]
        starts = np.concatenate(([0], np.cumsum(current))).astype(np.int32)
        histories = np.arange(1, batch + 1, dtype=np.int32) * 3
        seq_data = np.asarray(current, dtype=np.int32) + histories
        _write_back(query_start_loc, starts)
        _write_back(seq_lens, seq_data)

    rotation_data = to_bf16(
        np.eye(head_dim, dtype=np.float32)
        + rng.normal(0.0, 0.01, size=rotation.shape).astype(np.float32)
    )
    v_scale_data = rng.uniform(0.5, 1.5, size=v_scale.shape).astype(np.float32)
    if mrope_position is not None:
        token = np.arange(token_num, dtype=np.int64)
        position_data = np.stack(
            ((token * 3) % rows, (token * 5 + 1) % rows, (token * 7 + 2) % rows),
            axis=1,
        ).astype(np.int32)
        _write_back(mrope_position, position_data)

    for target, value in (
        (qkv, qkv_data), (q_gamma, q_gamma_data), (k_gamma, k_gamma_data),
        (cos_sin, cos_sin_data), (slot_mapping, slots),
        (k_cache, k_cache_data), (v_cache, v_cache_data),
        (k_scale_cache, k_scale_data), (rotation, rotation_data),
        (v_scale, v_scale_data),
    ):
        _write_back(target, value)


def kernel_customize_inputs(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc=None, seq_lens=None, rotation=None,
    v_scale=None, mrope_position=None, **kwargs,
):
    _prepare(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, testcase_name=kwargs.get("testcase_name"),
    )
    return (
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position,
    )

def aclnn_customize_inputs(
    qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
    kScaleCacheRef, queryStartLocOptional=None, seqLensOptional=None,
    rotationOptional=None, vScaleOptional=None, mropePositionOptional=None,
    headNums=None, layoutQkv="TND", layoutQOut="NTD", epsilon=1e-6,
    mropeSectionOptional=None, qQuantMode="PerTokenPerHead", qOut=None,
    qScaleOptional=None, **kwargs,
):
    _prepare(
        qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
        kScaleCacheRef, queryStartLocOptional, seqLensOptional,
        rotationOptional, vScaleOptional, mropePositionOptional,
        testcase_name=kwargs.get("testcase_name"),
    )


def pta_customize_inputs(
    qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
    k_scale_cache, query_start_loc, seq_lens, head_nums, *, rotation=None,
    v_scale=None, layout_qkv="TND", layout_q_out="NTD", epsilon=1e-6,
    mrope_position=None, mrope_section=None,
    q_quant_mode="PerTokenPerHead", q_out_dtype=None, **kwargs,
):
    _prepare(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, rotation, v_scale,
        mrope_position, testcase_name=kwargs.get("testcase_name"),
    )
