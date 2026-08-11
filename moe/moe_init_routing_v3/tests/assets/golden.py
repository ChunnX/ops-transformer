#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

__golden__ = {"kernel": {"moe_init_routing_v3": "MoeInitRoutingV3"}}

__input__ = {"kernel": {"moe_init_routing_v3": "MoeInitRoutingV3_input"}}

import numpy
from typing import List


def MoeInitRoutingV3_input(*input_arrays, **kwargs):
    input_arrays = list(input_arrays)
    expert_idx = input_arrays[1]
    if expert_idx is not None:
        expert_num = kwargs.get("expert_num", 256)
        expert_idx_arr = numpy.asarray(expert_idx)
        expert_idx_arr = numpy.random.randint(0, expert_num, size=expert_idx_arr.shape).astype(numpy.int32)
        invalid_ratio = kwargs.get("invalid_ratio", 0.0)
        if invalid_ratio > 0.0:
            mask = numpy.random.random(size=expert_idx_arr.shape) < invalid_ratio
            expert_idx_arr[mask] = -1
        input_arrays[1] = expert_idx_arr

    scale = input_arrays[2]
    if scale is not None:
        input_arrays[2] = numpy.random.uniform(-2, 2, size=scale.shape).astype(numpy.float32)

    offset = input_arrays[3]
    if offset is not None:
        input_arrays[3] = numpy.random.uniform(-2, 2, size=offset.shape).astype(numpy.float32)

    return tuple(input_arrays)


def _moe_init_routing_v3_numpy(x, expert_idx, scale, offset,
                               active_num, expert_capacity, expert_num, drop_pad_mode,
                               expert_tokens_num_type, expert_tokens_num_flag,
                               quant_mode, active_expert_range, row_idx_type):
    expert_start = active_expert_range[0]
    expert_end = active_expert_range[1]
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]
    total_length = num_rows * k
    if active_num == 0 or active_num == -1:
        x_out_num = total_length
    else:
        x_out_num = min(active_num, total_length)

    if drop_pad_mode == 1 and quant_mode == -1:
        expert_idx_flat = expert_idx.copy().reshape(-1)
        sorted_expert_indices = numpy.argsort(expert_idx_flat, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_flat[sorted_expert_indices]

        valid_mask = (sorted_expert_idx >= expert_start) & (sorted_expert_idx < expert_end)
        valid_sorted_row_idx = sorted_expert_indices[valid_mask]
        valid_sorted_expert_idx = sorted_expert_idx[valid_mask]

        sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=numpy.int64)
        expert_offset = numpy.zeros(expert_num, dtype=numpy.int64)
        for row_idx, expert_id in zip(valid_sorted_row_idx, valid_sorted_expert_idx):
            expert_id = int(expert_id)
            if expert_id < 0 or expert_id >= expert_num:
                continue
            offset_idx = expert_offset[expert_id]
            expert_offset[expert_id] += 1
            if offset_idx >= expert_capacity:
                continue
            sort_row_tmp[expert_id * expert_capacity + offset_idx] = row_idx

        expanded_row_idx = numpy.full((num_rows * k,), -1, dtype=numpy.int32)
        valid_capacity_mask = sort_row_tmp != -1
        expanded_row_idx[sort_row_tmp[valid_capacity_mask]] = numpy.arange(
            expert_num * expert_capacity, dtype=numpy.int32)[valid_capacity_mask]

        expanded_x = numpy.zeros((expert_num * expert_capacity, h), dtype=x.dtype)
        expanded_x[valid_capacity_mask] = x[sort_row_tmp[valid_capacity_mask] // k, :]
        expanded_x = expanded_x.reshape((expert_num, expert_capacity, h))

        expanded_scale = None
        if scale is not None:
            scale_shape = (expert_num * expert_capacity,) + scale.shape[1:]
            expanded_scale = numpy.zeros(scale_shape, dtype=scale.dtype)
            if scale.shape[0] == 1:
                expanded_scale[valid_capacity_mask] = scale[0]
            else:
                expanded_scale[valid_capacity_mask] = scale[sort_row_tmp[valid_capacity_mask] // k]
            if expanded_scale.ndim == 2 and expanded_scale.shape[-1] == 1:
                expanded_scale = expanded_scale.reshape(-1)

        if expert_tokens_num_flag:
            expert_tokens_count = numpy.bincount(
                valid_sorted_expert_idx - expert_start, minlength=expert_end - expert_start)
            expert_tokens_count = expert_tokens_count[:expert_end - expert_start].astype(numpy.int64)
        else:
            expert_tokens_count = numpy.array([], dtype=numpy.int64)
        return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale

    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = numpy.sum(
        (expert_idx_in >= expert_start) & (expert_idx_in < expert_end))

    expert_idx_in[(expert_idx_in < expert_start)] = numpy.int32(numpy.iinfo(numpy.int32).max)
    sorted_expert_indices = numpy.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices
    else:
        expanded_row_idx = numpy.ones(num_rows * k).astype(numpy.int32) * -1
        tmp_indices = numpy.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    if quant_mode == -1:
        if scale is None:
            expanded_scale = None
        else:
            expanded_scale = scale[sorted_expert_indices[:actual_expert_total_num] // k].flatten()
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]

    elif quant_mode == 0:
        expanded_scale = None
        x_fp16 = x.astype(numpy.float16)
        scale_fp16 = scale.astype(numpy.float16)
        expanded_x = x_fp16 * scale_fp16[0]
        offset_fp16 = offset.astype(numpy.float16)
        expanded_x = expanded_x + offset_fp16[0]
        expanded_x = numpy.rint(expanded_x)
        expanded_x = numpy.clip(expanded_x, -128, 127)
        expanded_x = expanded_x.astype(numpy.int8)

    elif quant_mode == 1:
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x.astype(numpy.float32)
        if scale is None:
            expanded_x = expanded_x[:actual_expert_total_num, :]
            x_abs = numpy.abs(expanded_x)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = expanded_x / expanded_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)
        else:
            expended_scale = scale[sorted_expert_idx[:actual_expert_total_num] - expert_start, :]
            expanded_x = expanded_x[:actual_expert_total_num, :]
            expanded_x = expanded_x * expended_scale
            x_abs = numpy.abs(expanded_x)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = expanded_x / expanded_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)

    elif quant_mode == 13:
        from tbetoolkits.utilities.dtypes import numpy_int4
        expanded_x = x[sorted_expert_indices // k, :].astype(numpy.float32)
        if scale is None:
            expanded_x = expanded_x[:actual_expert_total_num, :]
        else:
            scale_fp32 = scale.astype(numpy.float32)
            smooth_scale = scale_fp32.reshape(1, -1) if scale_fp32.ndim == 1 else scale_fp32
            expanded_x = expanded_x[:actual_expert_total_num, :] * smooth_scale
        x_abs = numpy.abs(expanded_x)
        x_max = numpy.max(x_abs, axis=-1, keepdims=True)
        expanded_scale = numpy.where(
            x_max == 0,
            numpy.float32(0.0),
            (x_max * numpy.float32(1.0 / 7.0)).astype(numpy.float32),
        )
        mul = numpy.where(
            expanded_scale == 0,
            numpy.float32(0.0),
            (numpy.float32(1.0) / expanded_scale).astype(numpy.float32),
        )
        expanded_x = numpy.rint((expanded_x * mul).astype(numpy.float32))
        expanded_x = numpy.clip(expanded_x, -8, 7).astype(numpy_int4())
        if actual_expert_total_num < x_out_num:
            expanded_x = numpy.concatenate(
                [expanded_x, numpy.ones((x_out_num - actual_expert_total_num, h), dtype=expanded_x.dtype)], axis=0)
            expanded_scale = numpy.concatenate(
                [expanded_scale.reshape(-1),
                 numpy.ones((x_out_num - actual_expert_total_num,), dtype=expanded_scale.dtype)])
        else:
            expanded_x = expanded_x[:x_out_num, :]
            expanded_scale = expanded_scale.reshape(-1)[:x_out_num]

    elif quant_mode in (2, 3, 16, 17):
        from tbetoolkits.utilities import _mxfp8_roundscale_quant_numpy
        expanded_x = x[sorted_expert_indices // k, :]
        expanded_x = expanded_x[:actual_expert_total_num, :]
        quant_mode_dtype_str_map = {2: "float8_e5m2", 3: "float8_e4m3fn", 16: "float8_e5m2", 17: "float8_e4m3fn"}
        expanded_scale, expanded_x = _mxfp8_roundscale_quant_numpy(
            expanded_x, quant_mode_dtype_str_map[quant_mode], clamp_amax=(quant_mode in (16, 17)))

    elif quant_mode in (4, 5, 14, 15):
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]
        expanded_x_f32 = expanded_x.astype(numpy.float32)
        num_rows_q, h_q = expanded_x_f32.shape
        group_size = 128
        num_groups = (h_q + group_size - 1) // group_size
        padded_h = num_groups * group_size
        expanded_x_padded = numpy.pad(expanded_x_f32, ((0, 0), (0, padded_h - h_q)),
                                       mode='constant', constant_values=0)
        grouped = expanded_x_padded.reshape(num_rows_q, num_groups, group_size)
        amax = numpy.max(numpy.abs(grouped), axis=-1)
        if quant_mode in (14, 15):
            amax = numpy.maximum(amax, numpy.float32(0.0001))
        fp8_max = {4: numpy.float32(57344.0), 5: numpy.float32(448.0),
                   14: numpy.float32(57344.0), 15: numpy.float32(448.0)}[quant_mode]
        raw_scale = numpy.array(amax / fp8_max, dtype=numpy.float32)
        raw_scale_int = raw_scale.view(numpy.int32)
        exp_bits = (raw_scale_int >> numpy.int32(23)) & numpy.int32(0xFF)
        mant_bits = raw_scale_int & numpy.int32(0x007FFFFF)
        mant_add = numpy.where(mant_bits != numpy.int32(0), numpy.int32(1), numpy.int32(0)).astype(numpy.int32)
        rounded_exp = exp_bits - numpy.int32(127) + mant_add
        biased_exp = rounded_exp + numpy.int32(127)
        round_scale_int = (biased_exp << numpy.int32(23)).astype(numpy.int32)
        round_scale = round_scale_int.view(numpy.float32).copy()
        nonzero_mask = raw_scale != numpy.float32(0.0)
        round_scale = numpy.where(nonzero_mask, round_scale, numpy.float32(0.0))
        divisor = numpy.where(round_scale != numpy.float32(0.0), round_scale, numpy.float32(1.0))
        divisor_expanded = divisor[:, :, numpy.newaxis]
        quantized_padded = grouped / divisor_expanded
        from ml_dtypes import float8_e5m2, float8_e4m3fn
        fp8_dtype = {4: float8_e5m2, 5: float8_e4m3fn, 14: float8_e5m2, 15: float8_e4m3fn}[quant_mode]
        expanded_x = quantized_padded.reshape(num_rows_q, padded_h)[:, :h_q].astype(fp8_dtype)
        expanded_scale = round_scale

    if expert_tokens_num_type == 0:
        counts = numpy.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start,
            minlength=expert_end - expert_start)
        expert_tokens_count = numpy.cumsum(counts).astype(numpy.int64)
    elif expert_tokens_num_type == 1:
        expert_tokens_count = numpy.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start)
        expert_tokens_count = numpy.concatenate(
            [expert_tokens_count, numpy.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(numpy.int64)])
    elif expert_tokens_num_type == 2:
        expert_id, counts = numpy.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
        expert_tokens_count = numpy.column_stack((expert_id, counts))
        if expert_tokens_count.shape[0] < expert_num:
            expert_tokens_count = numpy.concatenate((expert_tokens_count, [[0, 0]]), axis=0)
    return expanded_x, expanded_row_idx.astype(numpy.int32), expert_tokens_count.astype(numpy.int64), expanded_scale


def MoeInitRoutingV3(*input_arrays, **kwargs):
    x = input_arrays[0]
    expert_idx = input_arrays[1]
    scale = input_arrays[2] if len(input_arrays) > 2 else None
    offset = input_arrays[3] if len(input_arrays) > 3 else None

    active_num = kwargs.get("active_num", -1)
    expert_capacity = kwargs.get("expert_capacity", -1)
    expert_num = kwargs.get("expert_num", 256)
    drop_pad_mode = kwargs.get("drop_pad_mode", 0)
    expert_tokens_num_type = kwargs.get("expert_tokens_num_type", 1)
    expert_tokens_num_flag = kwargs.get("expert_tokens_num_flag", True)
    quant_mode = kwargs.get("quant_mode", -1)
    active_expert_range = kwargs.get("active_expert_range", [0, expert_num])
    row_idx_type = kwargs.get("row_idx_type", 1)

    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = _moe_init_routing_v3_numpy(
        x, expert_idx, scale, offset, active_num, expert_capacity, expert_num,
        drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag,
        quant_mode, active_expert_range, row_idx_type)
    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale
