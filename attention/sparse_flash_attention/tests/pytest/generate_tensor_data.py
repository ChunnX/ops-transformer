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
import json
import math
import struct
import torch
import copy


INVALID_VALUE = -1


def get_torch_dtype(type_str):
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
    input_i_data = None
    index = SHAPE_KEY_TO_INDEX.get(key, 0)
    shape_input = params["shape_input"][key]
    gen_dtype = params["dtype_input"][key]
    dtype_input = get_torch_dtype(gen_dtype)

    is_fix_value = False
    for param_key in params.keys():
        if f"tensor_data_{index}" in param_key and "required_" not in param_key:
            input_i_data = torch.tensor(
                params[param_key], dtype=dtype_input
            ).reshape(shape_input)
            is_fix_value = True

    if not is_fix_value:
        tensor_duplicate_rat = None
        real_shape = None
        diff_data_size = None

        if len(params["range_input"][key]) == 1:
            input_i_data = gen_boundary_tensor_data(
                shape_input,
                params["range_input"][key],
                dtype_input,
            )
        else:
            range_input = params["range_input"][key]
            input_i_data = gen_nonbound_tensor_data(
                params, shape_input, range_input, dtype_input, index
            )

    return input_i_data


def gen_boundary_tensor_data(shape_input, input_range, dtype_input):
    input_i_data = None
    if input_range[0] == "null":
        input_i_data = torch.empty(0, dtype=dtype_input)
    elif input_range[0] == "nan":
        input_i_data = torch.full(shape_input, float("nan"), dtype=dtype_input)
    elif input_range[0] == "inf":
        input_i_data = torch.full(shape_input, float("inf"), dtype=dtype_input)
    elif input_range[0] == "-inf":
        input_i_data = torch.full(shape_input, float("-inf"), dtype=dtype_input)
    elif input_range[0] == "default":
        input_i_data = None
    else:
        input_i_data = torch.full(shape_input, input_range[0], dtype=dtype_input)
    return input_i_data


def gen_nonbound_tensor_data(params, data_shape, range_input, dtype, tensor_index):
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
            if dtype in [torch.float32, torch.float64, torch.float16]:
                min_value = torch.finfo(dtype).min
                max_value = torch.finfo(dtype).max
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                min_value = torch.iinfo(dtype).min
                max_value = torch.iinfo(dtype).max
            else:
                min_value = False
                max_value = True
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            if dtype == torch.float64:
                input_data = (
                    torch.empty(data_shape, dtype=torch.float64).uniform_(-1, 1)
                    * max_value
                )
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                input_data = torch.randint(
                    min_value, max_value, data_shape, dtype=dtype
                )
            else:
                input_data = torch.empty(data_shape, dtype=dtype).uniform_(
                    min_value, max_value
                )
            if data_shape == []:
                return gen_uniform_data(data_shape, -1, 1, dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape)
            if shape_len == 0:
                return gen_uniform_data(data_shape, 0, 4, dtype, hash_seed)
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            num_change = torch.randint(
                low=0, high=min(shape_len, 4), size=(1,)
            ).item()
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 1)
            inf_index = torch.unravel_index(
                torch.randperm(shape_len)[:num_change], data_shape
            )
            input_data[inf_index] = min_value
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 2)
            num_change = torch.randint(
                low=0, high=min(shape_len, 4), size=(1,)
            ).item()
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 3)
            inf_index = torch.unravel_index(
                torch.randperm(shape_len)[:num_change], data_shape
            )
            input_data[inf_index] = max_value
            return input_data
        elif min_value == "nan":
            return torch.full(data_shape, float("nan"), dtype=dtype)
        elif min_value == "inf":
            return torch.full(data_shape, float("inf"), dtype=dtype)
        elif min_value == "-inf":
            return torch.full(data_shape, float("-inf"), dtype=dtype)
        elif min_value == "default":
            return None
        elif min_value == "null":
            return torch.empty(0, dtype=dtype)
        else:
            return gen_uniform_data(
                data_shape, min_value, max_value, dtype, hash_seed
            )
    except MemoryError:
        print("[ERROR] MemoryError.")
        return INVALID_VALUE


def gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    if hash_seed > 0:
        torch.manual_seed(hash_seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, data_shape, dtype=torch.bool)
    if dtype == torch.complex64:
        real = torch.empty(data_shape, dtype=torch.float32).uniform_(
            min_value, max_value
        )
        if hash_seed > 0:
            torch.manual_seed(hash_seed + 1)
        imag = torch.empty(data_shape, dtype=torch.float32).uniform_(
            min_value, max_value
        )
        return torch.complex(real, imag)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return torch.randint(
            int(min_value), int(max_value), data_shape, dtype=dtype
        )
    return torch.empty(data_shape, dtype=dtype).uniform_(min_value, max_value)
