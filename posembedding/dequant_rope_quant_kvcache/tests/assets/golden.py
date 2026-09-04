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
"""Independent Kernel golden for DequantRopeQuantKvcache."""

from __future__ import annotations

import numpy as np
import torch

try:
    import ml_dtypes
except ImportError as exc:  # pragma: no cover - environment error is explicit
    raise ImportError("DequantRopeQuantKvcache golden requires ml_dtypes") from exc


BF16_DTYPE = np.dtype(ml_dtypes.bfloat16)


def _as_torch(value):
    """Convert Kernel NumPy inputs to CPU torch tensors without losing BF16 bits."""

    if value is None or isinstance(value, torch.Tensor):
        return value
    value = np.asarray(value)
    if value.dtype == BF16_DTYPE:
        return torch.from_numpy(value.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(value)


def _as_numpy(value):
    """Convert a CPU torch tensor to NumPy without losing BF16 bits."""

    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().view(BF16_DTYPE)
    return value.numpy()


def _broadcast_quant_param(value, kv_head_num, head_dim):
    value = value.reshape(-1)
    if value.numel() == kv_head_num * head_dim:
        return value.reshape(1, 1, kv_head_num, head_dim)
    return value.reshape(1, 1, 1, head_dim)


def _apply_rope(value, cos, sin, compute_dtype):
    value = value.to(compute_dtype)
    cos = cos.to(compute_dtype)
    sin = sin.to(compute_dtype)
    half_dim = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half_dim:], value[..., :half_dim]), dim=-1)
    return torch.add(torch.mul(value, cos), torch.mul(rotated, sin))


def _quantize(value, scale, offset, kv_head_num, head_dim):
    value = value.to(torch.float32)
    scale = _broadcast_quant_param(scale, kv_head_num, head_dim).to(torch.float32)
    quantized = torch.div(value, scale)
    if offset is not None:
        offset = _broadcast_quant_param(offset, kv_head_num, head_dim).to(torch.float32)
        quantized = torch.add(quantized, offset)

    # Historical kernel: FP32 -> INT16 CAST_RINT -> FP16 -> saturating INT8.
    quantized = torch.clamp(torch.round(quantized), -32768, 32767).to(torch.int16)
    quantized = quantized.to(torch.float16)
    return torch.clamp(quantized, -128, 127).to(torch.int8)


def dequant_rope_quant_kvcache_golden(
    x,
    cos,
    sin,
    k_cache,
    v_cache,
    indices,
    scale_k,
    scale_v,
    offset_k=None,
    offset_v=None,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    *,
    size_splits,
    quant_mode="static",
    layout="BSND",
    kv_output=False,
    cache_mode="contiguous",
    **kwargs,
):
    """Compute all five outputs from inputs without consuming device results."""

    del quant_mode, layout, kwargs
    x = _as_torch(x)
    cos = _as_torch(cos)
    sin = _as_torch(sin)
    k_cache = _as_torch(k_cache).clone()
    v_cache = _as_torch(v_cache).clone()
    indices = _as_torch(indices).reshape(-1)
    scale_k = _as_torch(scale_k)
    scale_v = _as_torch(scale_v)
    offset_k = _as_torch(offset_k)
    offset_v = _as_torch(offset_v)
    weight_scale = _as_torch(weight_scale)
    activation_scale = _as_torch(activation_scale)
    bias = _as_torch(bias)

    size_splits = tuple(int(value) for value in size_splits)
    if len(size_splits) != 3:
        raise ValueError("size_splits must contain q, k and v sizes")

    x_was_2d = x.ndim == 2
    if x_was_2d:
        x = x.reshape(x.shape[0], 1, -1)
        cos = cos.reshape(cos.shape[0], 1, 1, -1)
        sin = sin.reshape(sin.shape[0], 1, 1, -1)

    batch_size, sequence_length = x.shape[:2]
    kv_head_num, head_dim = k_cache.shape[-2:]
    output_dtype = cos.dtype
    compute_dtype = torch.float16 if output_dtype == torch.float16 else torch.float32

    if x.dtype == torch.int32:
        if weight_scale is None:
            raise ValueError("weight_scale is required when x is int32")
        weight_scale = weight_scale.to(torch.float32).reshape(1, 1, -1)
        token_scale = None
        if activation_scale is not None:
            token_scale = activation_scale.to(torch.float32).reshape(
                batch_size, sequence_length, 1
            )

        if bias is not None and bias.dtype == torch.int32:
            # The device adds INT32 bias before converting the sum to FP32.
            dequantized = torch.add(x, bias.reshape(1, 1, -1)).to(torch.float32)
        else:
            dequantized = x.to(torch.float32)
        dequantized = torch.mul(dequantized, weight_scale)
        if token_scale is not None:
            dequantized = torch.mul(dequantized, token_scale)
        if bias is not None and bias.dtype != torch.int32:
            dequantized = torch.add(
                dequantized, bias.to(torch.float32).reshape(1, 1, -1)
            )

        rope_input = (
            dequantized.to(torch.float16)
            if output_dtype == torch.float16
            else dequantized
        )
        value_input = dequantized
    elif x.dtype == torch.bfloat16:
        rope_input = x.to(torch.float32)
        value_input = rope_input
    else:
        rope_input = x
        value_input = x

    q_size, k_size, v_size = size_splits
    q = rope_input[..., :q_size]
    k = rope_input[..., q_size : q_size + k_size]
    v = value_input[..., q_size + k_size : q_size + k_size + v_size]
    q = q.reshape(batch_size, sequence_length, -1, head_dim)
    k = k.reshape(batch_size, sequence_length, -1, head_dim)
    v = v.reshape(batch_size, sequence_length, -1, head_dim)

    q_out = _apply_rope(q, cos, sin, compute_dtype)
    k_out = _apply_rope(k, cos, sin, compute_dtype)
    quantized_k = _quantize(k_out, scale_k, offset_k, kv_head_num, head_dim)
    quantized_v = _quantize(v, scale_v, offset_v, kv_head_num, head_dim)

    if cache_mode == "page":
        k_cache_view = k_cache.reshape(-1, kv_head_num, head_dim)
        v_cache_view = v_cache.reshape(-1, kv_head_num, head_dim)
        quantized_k = quantized_k.reshape(-1, kv_head_num, head_dim)
        quantized_v = quantized_v.reshape(-1, kv_head_num, head_dim)
        for token, index in enumerate(indices):
            index = int(index)
            if index >= 0:
                k_cache_view[index] = quantized_k[token]
                v_cache_view[index] = quantized_v[token]
    else:
        for batch, index in enumerate(indices):
            index = int(index)
            k_cache[batch, index : index + sequence_length] = quantized_k[batch]
            v_cache[batch, index : index + sequence_length] = quantized_v[batch]

    q_out = q_out.to(output_dtype)
    if kv_output:
        k_result = k_out.to(output_dtype)
        v_result = v.to(output_dtype)
    else:
        k_result = torch.empty(0, dtype=output_dtype)
        v_result = torch.empty(0, dtype=output_dtype)
    if x_was_2d:
        q_out = q_out[:, 0]
        if kv_output:
            k_result = k_result[:, 0]
            v_result = v_result[:, 0]
    return [_as_numpy(value) for value in (q_out, k_result, v_result, k_cache, v_cache)]


def customize_inputs(*inputs, **kwargs):
    """Constrain indices to the valid ranges documented by the operator."""

    values = list(inputs)
    x, k_cache, indices = values[0], values[3], values[5]
    seed = sum(
        bytearray(
            kwargs.get("testcase_name", "dequant_rope_quant_kvcache").encode("utf-8")
        )
    )
    generator = np.random.default_rng(seed)
    if kwargs.get("cache_mode", "contiguous").lower() == "page":
        slot_count = k_cache.shape[0] * k_cache.shape[1]
        indices[...] = generator.permutation(slot_count)[: indices.size].reshape(
            indices.shape
        )
    else:
        sequence_length = 1 if x.ndim == 2 else x.shape[1]
        max_start = k_cache.shape[1] - sequence_length
        indices[...] = generator.integers(
            0, max_start + 1, size=indices.shape, dtype=np.int32
        )
    return tuple(values)


class DequantRopeQuantKvcacheTestSpec:
    """Raw Kernel test specification using the independent CPU golden."""

    golden = staticmethod(dequant_rope_quant_kvcache_golden)
    customize_inputs = staticmethod(customize_inputs)
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
    }


__spec__ = {
    "dequant_rope_quant_kvcache": "DequantRopeQuantKvcacheTestSpec",
}
