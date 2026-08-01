#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import operator
import random
from functools import reduce
import os
import math
import torch

INVALID_VALUE = -1


def get_torch_dtype_for_gen(type_str):
    type_dict = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    return type_dict[type_str]


SHAPE_KEY_TO_INDEX = {
    "query": 0,
    "key": 1,
    "value": 2,
    "sparse_indices": 3,
    "block_table": 4,
    "query_rope": 5,
    "key_rope": 6,
    "sinks": 7,
}


def gen_tensor_data(params, key):
    index = SHAPE_KEY_TO_INDEX.get(key, 0)
    shape_input = params["shape_input"][key]
    gen_dtype = params["dtype_input"][key]
    torch_dtype = get_torch_dtype_for_gen(gen_dtype)

    is_fix_value = False
    for param_key in params.keys():
        if f"tensor_data_{index}" in param_key and "required_" not in param_key:
            raw_data = params[param_key]
            if gen_dtype in ("bf16", "bfloat16"):
                input_i_data = (
                    torch.tensor(raw_data, dtype=torch.float32)
                    .reshape(shape_input)
                    .to(torch.bfloat16)
                )
            else:
                input_i_data = torch.tensor(raw_data, dtype=torch_dtype).reshape(
                    shape_input
                )
            is_fix_value = True

    if not is_fix_value:
        if len(params["range_input"][key]) == 1:
            input_i_data = gen_boundary_tensor_data(
                shape_input,
                params["range_input"][key],
                torch_dtype,
                params["dtype_input"][key],
            )
        else:
            range_input = params["range_input"][key]
            input_i_data = gen_nonbound_tensor_data(
                params, shape_input, range_input, torch_dtype, index
            )

    if input_i_data is None:
        return torch.tensor([], dtype=torch_dtype)
    return input_i_data


def gen_boundary_tensor_data(shape_input, input_range, torch_dtype, dtype_raw="fp32"):
    input_i_data = None
    if input_range[0] == "null":
        input_i_data = torch.tensor([], dtype=torch_dtype)
    elif input_range[0] == "nan":
        try:
            input_i_data = torch.full(shape_input, float("nan"), dtype=torch.float32).to(
                torch_dtype
            )
        except Exception:
            input_i_data = torch.full(shape_input, float("nan"), dtype=torch.float32).to(
                torch_dtype
            )
    elif input_range[0] == "inf":
        try:
            input_i_data = torch.full(shape_input, float("inf"), dtype=torch.float32).to(
                torch_dtype
            )
        except Exception:
            input_i_data = torch.full(shape_input, float("inf"), dtype=torch.float32).to(
                torch_dtype
            )
    elif input_range[0] == "-inf":
        try:
            input_i_data = torch.full(
                shape_input, float("-inf"), dtype=torch.float32
            ).to(torch_dtype)
        except Exception:
            input_i_data = torch.full(
                shape_input, float("-inf"), dtype=torch.float32
            ).to(torch_dtype)
    elif input_range[0] == "default":
        input_i_data = None
    else:
        val = float(input_range[0])
        input_i_data = torch.full(shape_input, val, dtype=torch.float32).to(torch_dtype)
    return input_i_data


def gen_nonbound_tensor_data(params, data_shape, range_input, torch_dtype, tensor_index):
    fuzz_value_type = ""
    try:
        if isinstance(range_input, dict):
            fuzz_value_type = "normal"
        elif isinstance(range_input, list):
            fuzz_value_type = "uniform"
        else:
            return INVALID_VALUE
        hash_seed = -1
        min_value, max_value = range_input
        if min_value == "-inf" and max_value == "inf":
            if torch_dtype in [torch.float32, torch.float64, torch.float16]:
                info = torch.finfo(torch_dtype)
                min_value = info.min
                max_value = info.max
            elif torch_dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                info = torch.iinfo(torch_dtype)
                min_value = info.min
                max_value = info.max
            else:
                min_value = False
                max_value = True
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            if torch_dtype == torch.float64:
                input_data = (
                    torch.rand(data_shape, dtype=torch.float64) * 2 - 1
                ) * max_value
            else:
                input_data = (
                    torch.rand(data_shape, dtype=torch.float32)
                    * (max_value - min_value)
                    + min_value
                ).to(torch_dtype)
            if data_shape == []:
                return gen_uniform_data(data_shape, -1, 1, torch_dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape)
            if shape_len == 0:
                return gen_uniform_data(data_shape, 0, 4, torch_dtype, hash_seed)
            flat = input_data.flatten()
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            num_change = torch.randint(0, min(shape_len, 4) + 1, (1,)).item()
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 1)
            indices = torch.randperm(shape_len)[:num_change]
            flat[indices] = min_value
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 2)
            num_change = torch.randint(0, min(shape_len, 4) + 1, (1,)).item()
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 3)
            indices = torch.randperm(shape_len)[:num_change]
            flat[indices] = max_value
            return input_data
        elif min_value == "nan":
            try:
                return torch.full(
                    data_shape, float("nan"), dtype=torch.float32
                ).to(torch_dtype)
            except Exception:
                return torch.full(
                    data_shape, float("nan"), dtype=torch.float32
                ).to(torch_dtype)
        elif min_value == "inf":
            try:
                return torch.full(
                    data_shape, float("inf"), dtype=torch.float32
                ).to(torch_dtype)
            except Exception:
                return torch.full(
                    data_shape, float("inf"), dtype=torch.float32
                ).to(torch_dtype)
        elif min_value == "-inf":
            try:
                return torch.full(
                    data_shape, float("-inf"), dtype=torch.float32
                ).to(torch_dtype)
            except Exception:
                return torch.full(
                    data_shape, float("-inf"), dtype=torch.float32
                ).to(torch_dtype)
        elif min_value == "default":
            return None
        elif min_value == "null":
            return torch.tensor([], dtype=torch_dtype)
        else:
            return gen_uniform_data(
                data_shape, min_value, max_value, torch_dtype, hash_seed
            )
    except MemoryError:
        print("[ERROR] MemoryError.")
        return INVALID_VALUE


def gen_uniform_data(data_shape, min_value, max_value, torch_dtype, hash_seed):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=torch_dtype)
    if hash_seed > 0:
        torch.manual_seed(hash_seed)
    if torch_dtype == torch.bool:
        return torch.randint(0, 2, data_shape, dtype=torch.bool)
    if torch_dtype == torch.complex64:
        if hash_seed > 0:
            torch.manual_seed(hash_seed)
        real = (
            torch.rand(data_shape, dtype=torch.float32) * (max_value - min_value)
            + min_value
        )
        if hash_seed > 0:
            torch.manual_seed(hash_seed + 1)
        imag = (
            torch.rand(data_shape, dtype=torch.float32) * (max_value - min_value)
            + min_value
        )
        return torch.complex(real, imag)
    data = (
        torch.rand(data_shape, dtype=torch.float32) * (max_value - min_value) + min_value
    )
    return data.to(torch_dtype)
