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


SHAPE_KEY_TO_INDEX = {
    'query': 0,
    'key': 1,
    'value': 2,
    'sparse_indices': 3,
    'block_table': 4,
    'query_rope': 5,
    'key_rope': 6,
}


def gen_tensor_data(params, key):
    input_i_data = None
    index = SHAPE_KEY_TO_INDEX.get(key, 0)
    shape_input = params["shape_input"][key]
    torch_dtype = params["dtype_input"][key]

    is_fix_value = False
    for param_key in params.keys():
        if f"tensor_data_{index}" in param_key and "required_" not in param_key:
            input_i_data = torch.tensor(params[param_key], dtype=torch_dtype).reshape(shape_input)
            is_fix_value = True

    if not is_fix_value:
        range_input = params["range_input"][key]
        if len(range_input) == 1:
            input_i_data = gen_boundary_tensor_data(shape_input, range_input, torch_dtype)
        else:
            input_i_data = gen_nonbound_tensor_data(
                params,
                shape_input,
                range_input,
                torch_dtype,
                index)

    return input_i_data


def gen_boundary_tensor_data(shape_input, input_range, torch_dtype):
    input_i_data = None
    if input_range[0] == "null":
        input_i_data = torch.tensor([], dtype=torch_dtype)
    elif input_range[0] == "nan":
        input_i_data = torch.full(shape_input, float('nan'), dtype=torch_dtype)
    elif input_range[0] == "inf":
        input_i_data = torch.full(shape_input, float('inf'), dtype=torch_dtype)
    elif input_range[0] == "-inf":
        input_i_data = torch.full(shape_input, float('-inf'), dtype=torch_dtype)
    elif input_range[0] == "default":
        input_i_data = None
    else:
        input_i_data = torch.full(shape_input, input_range[0], dtype=torch_dtype)
    return input_i_data


def gen_nonbound_tensor_data(params, data_shape, range_input, torch_dtype, tensor_index):
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
            if hash_seed > 0:
                torch.manual_seed(hash_seed)
            
            if torch_dtype in (torch.float32, torch.float16, torch.bfloat16):
                dtype_min = torch.finfo(torch_dtype).min
                dtype_max = torch.finfo(torch_dtype).max
                input_data = torch.rand(data_shape, dtype=torch_dtype)
                input_data = input_data * (dtype_max - dtype_min) + dtype_min
            elif torch_dtype in (torch.int8, torch.int32, torch.uint8):
                dtype_min = torch.iinfo(torch_dtype).min
                dtype_max = torch.iinfo(torch_dtype).max
                input_data = torch.randint(dtype_min, dtype_max + 1, data_shape, dtype=torch_dtype)
            elif torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # float8 类型: 先用 float32 生成，再转换
                input_data = torch.rand(data_shape, dtype=torch.float32)
                input_data = input_data.to(torch_dtype)
            else:
                input_data = torch.rand(data_shape, dtype=torch.float32).to(torch_dtype)
            
            if data_shape == []:
                return gen_uniform_data(data_shape, -1, 1, torch_dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape) if isinstance(data_shape, list) else data_shape
            if shape_len == 0:
                return gen_uniform_data(data_shape, 0, 4, torch_dtype, hash_seed)
            
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 1)
            num_change = torch.randint(0, min(shape_len, 4), (1,)).item()
            if num_change > 0 and torch_dtype in (torch.float32, torch.float16, torch.bfloat16):
                inf_index = torch.randperm(shape_len)[:num_change]
                dtype_min = torch.finfo(torch_dtype).min
                for idx in inf_index:
                    input_data.view(-1)[idx] = dtype_min
            
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 2)
            num_change = torch.randint(0, min(shape_len, 4), (1,)).item()
            if hash_seed > 0:
                torch.manual_seed(hash_seed + 3)
            if num_change > 0 and torch_dtype in (torch.float32, torch.float16, torch.bfloat16):
                inf_index = torch.randperm(shape_len)[:num_change]
                dtype_max = torch.finfo(torch_dtype).max
                for idx in inf_index:
                    input_data.view(-1)[idx] = dtype_max
            return input_data
        elif min_value == "nan":
            return torch.full(data_shape, float('nan'), dtype=torch_dtype)
        elif min_value == "inf":
            return torch.full(data_shape, float('inf'), dtype=torch_dtype)
        elif min_value == "-inf":
            return torch.full(data_shape, float('-inf'), dtype=torch_dtype)
        elif min_value == "default":
            return None
        elif min_value == "null":
            return torch.tensor([], dtype=torch_dtype)
        else:
            return gen_uniform_data(data_shape, min_value, max_value, torch_dtype, hash_seed)
    except MemoryError:
        print(f"[ERROR] MemoryError.")
        return INVALID_VALUE


def gen_uniform_data(data_shape, min_value, max_value, torch_dtype, hash_seed):
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=torch_dtype)
    if hash_seed > 0:
        torch.manual_seed(hash_seed)
    
    if torch_dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.rand(data_shape, dtype=torch_dtype) * (max_value - min_value) + min_value
    elif torch_dtype in (torch.int8, torch.int32):
        return torch.randint(int(min_value), int(max_value) + 1, data_shape, dtype=torch_dtype)
    elif torch_dtype == torch.uint8:
        return torch.randint(int(min_value), int(max_value) + 1, data_shape, dtype=torch_dtype)
    elif torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # float8 类型: 先用 float32 生成，再转换
        return (torch.rand(data_shape, dtype=torch.float32) * (max_value - min_value) + min_value).to(torch_dtype)
    
    return (torch.rand(data_shape, dtype=torch.float32) * (max_value - min_value) + min_value).to(torch_dtype)