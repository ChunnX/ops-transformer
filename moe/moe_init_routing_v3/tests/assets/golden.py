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

import numpy as np


def MoeInitRoutingV3_input(*input_arrays, **kwargs):
    input_arrays = list(input_arrays)
    expert_idx = input_arrays[1]
    if expert_idx is not None:
        expert_num = kwargs.get("expert_num", 256)
        expert_idx_arr = np.asarray(expert_idx)
        expert_idx_arr = np.random.randint(0, expert_num, size=expert_idx_arr.shape).astype(np.int32)
        invalid_ratio = kwargs.get("invalid_ratio", 0.0)
        if invalid_ratio > 0.0:
            mask = np.random.random(size=expert_idx_arr.shape) < invalid_ratio
            expert_idx_arr[mask] = -1
        input_arrays[1] = expert_idx_arr

    scale = input_arrays[2]
    if scale is not None:
        quant_mode = kwargs.get("quant_mode", -1)
        if quant_mode == -1:
            input_arrays[2] = np.random.uniform(-2, 2, size=scale.shape).astype(np.float32)
        elif quant_mode == 1:
            input_arrays[2] = np.random.uniform(-2, 2, size=scale.shape).astype(np.float32)

    offset = input_arrays[3]
    if offset is not None:
        input_arrays[3] = np.random.uniform(-2, 2, size=offset.shape).astype(np.float32)

    return tuple(input_arrays)


def MoeInitRoutingV3(x, expert_idx, scale, offset, **kwargs):
    active_num = kwargs.get("active_num", -1)
    expert_capacity = kwargs.get("expert_capacity", -1)
    expert_num = kwargs.get("expert_num", 256)
    drop_pad_mode = kwargs.get("drop_pad_mode", 0)
    expert_tokens_num_type = kwargs.get("expert_tokens_num_type", 1)
    expert_tokens_num_flag = kwargs.get("expert_tokens_num_flag", True)
    quant_mode = kwargs.get("quant_mode", -1)
    active_expert_range = kwargs.get("active_expert_range", [0, expert_num])
    row_idx_type = kwargs.get("row_idx_type", 1)

    x = np.asarray(x)
    expert_idx = np.asarray(expert_idx)
    scale = np.asarray(scale) if scale is not None else None
    offset = np.asarray(offset) if offset is not None else None

    expert_start = active_expert_range[0] if drop_pad_mode == 0 else 0
    expert_end = active_expert_range[1] if drop_pad_mode == 0 else expert_num
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]
    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = int(np.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end)))

    expert_idx_in[(expert_idx_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
    sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]

    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices
    else:
        expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
        tmp_indices = np.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    expanded_scale = None
    if drop_pad_mode == 0:
        if active_num == 0 or active_num == -1:
            active_num = actual_expert_total_num
        else:
            active_num = min(active_num, actual_expert_total_num)

    if quant_mode == -1:
        expanded_x = x[sorted_expert_indices[:actual_expert_total_num] // k, :]
        if scale is not None:
            expanded_scale = scale[sorted_expert_indices[:actual_expert_total_num] // k]

    elif quant_mode == 0:
        expanded_scale = None
        x_fp16 = x.astype(np.float16)
        scale_fp16 = scale.astype(np.float16)
        if scale_fp16.ndim == 1:
            scale_fp16 = scale_fp16[:, np.newaxis]
        expanded_x = x_fp16[sorted_expert_indices[:actual_expert_total_num] // k, :] * \
                     scale_fp16[sorted_expert_idx[:actual_expert_total_num] - expert_start, :]
        if offset is not None:
            offset_fp16 = offset.astype(np.float16)
            if offset_fp16.ndim == 1:
                offset_fp16 = offset_fp16[:, np.newaxis]
            expanded_x = expanded_x + \
                         offset_fp16[sorted_expert_idx[:actual_expert_total_num] - expert_start, :]
        expanded_x = np.rint(expanded_x)
        expanded_x = np.clip(expanded_x, -128, 127)
        expanded_x = expanded_x.astype(np.int8)

    elif quant_mode == 1:
        total_length = num_rows * k
        x_out_num = min(active_num, total_length) if active_num > 0 else total_length

        expanded_x = x[sorted_expert_indices // k, :].astype(np.float32)
        if scale is None:
            expanded_x = expanded_x[:actual_expert_total_num, :]
        else:
            smooth_scale = scale[sorted_expert_idx[:actual_expert_total_num] - expert_start, :]
            expanded_x = expanded_x[:actual_expert_total_num, :] * smooth_scale

        x_abs = np.abs(expanded_x)
        x_max = np.max(x_abs, axis=-1, keepdims=True)
        expanded_scale = x_max / 127
        expanded_x = expanded_x / expanded_scale
        expanded_x = np.round(expanded_x).astype(np.int8)

        if actual_expert_total_num < x_out_num:
            pad_val = np.ones(1, dtype=expanded_x.dtype)[0]
            scale_pad_val = np.float32(1.0)
            expanded_x = np.concatenate(
                [expanded_x, np.full((x_out_num - actual_expert_total_num, h), pad_val, dtype=expanded_x.dtype)],
                axis=0
            )
            expanded_scale = np.concatenate(
                [
                    expanded_scale.reshape(-1),
                    np.full((x_out_num - actual_expert_total_num,), scale_pad_val, dtype=expanded_scale.dtype),
                ]
            )
        else:
            expanded_x = expanded_x[:x_out_num, :]
            expanded_scale = expanded_scale.reshape(-1)[:x_out_num]

    if drop_pad_mode == 1:
        def _adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
            count = 0
            last = sorted_expert_idx[0]
            for i, val in enumerate(sorted_expert_idx):
                if last != val:
                    count = 1
                    last = val
                else:
                    count += 1
                    if count > capacity:
                        sorted_expert_idx[i] = -1
                        sorted_row_idx[i] = -1

        _adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)
        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
        offset_tmp = 0
        last_expert_id = 0
        for i, val in enumerate(sorted_expert_indices):
            if val != -1:
                if last_expert_id != sorted_expert_idx[i]:
                    offset_tmp = 0
                    last_expert_id = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                offset_tmp = offset_tmp + 1
        expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i
        expanded_x_mask = np.full((expert_num * expert_capacity, h), 1, dtype=int)
        expanded_x = np.full((expert_num * expert_capacity, h), 0, dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = np.full((h,), 0, dtype=int)
        if quant_mode == -1 and scale is not None:
            expanded_scale = np.full((expert_num * expert_capacity,), 0, dtype=scale.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_scale[i] = scale[val // k]
        expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

    if expert_tokens_num_flag:
        if expert_tokens_num_type == 1:
            expert_tokens_count = np.bincount(
                sorted_expert_idx[:actual_expert_total_num] - expert_start)
            expert_tokens_count = np.concatenate(
                [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
        elif expert_tokens_num_type == 2:
            expert_id, counts = np.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
            expert_tokens_count = np.column_stack((expert_id, counts))
            if expert_tokens_count.shape[0] < expert_num:
                expert_tokens_count = np.concatenate((expert_tokens_count, [[0, 0]]), axis=0)
        else:
            expert_tokens_count = np.bincount(
                sorted_expert_idx[:actual_expert_total_num] - expert_start)
            expert_tokens_count = np.concatenate(
                [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            expert_tokens_count = np.cumsum(expert_tokens_count)
        expert_tokens_count = expert_tokens_count.astype(np.int64)
    else:
        expert_tokens_count = None

    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale
