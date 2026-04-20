import operator
import random
from functools import reduce
import datetime
import os
import sys
import numpy as np
import os
import json
import math
import struct
import torch
import tensorflow as tf
import copy
from ml_dtypes import float8_e4m3fn
import utils


INVALID_VALUE = -1

dtype_width_map = {
    "complex32": 4,
    "complex64": 8,
    "complex128": 16,
    "double": 8,
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "uint64": 8,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
    "uint1": 1,
    "bool": 1,
    "float4_e2m1": 0.5,
    "float4_e1m2": 0.5,
    "float8_e8m0": 1,
    "float8_e5m2": 1,
    "float8_e4m3fn": 1,
    "hifloat8": 1,
    None: 0
}
dtype_map = {
    "c32": "complex32",
    "c64": "complex64",
    "complex32": "complex32",
    "complex64": "complex64",

    "f64": "float64",
    "fp64": "float64",
    "float64": "float64",
    "double": "double",

    "f32": "float32",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",

    "f16": "float16",
    "fp16": "float16",
    "float16": "float16",

    "bf16": "bfloat16",
    "bfp16": "bfloat16",
    "bfloat16": "bfloat16",

    "s64": "int64",
    "int64": "int64",

    "u64": "uint64",
    "uint64": "uint64",

    "s32": "int32",
    "int32": "int32",

    "u32": "uint32",
    "uint32": "uint32",

    "s16": "int16",
    "int16": "int16",

    "u16": "uint16",
    "uint16": "uint16",

    "s8": "int8",
    "int8": "int8",

    "u8": "uint8",
    "uint8": "uint8",

    "s4": "int4",
    "int4": "int4",

    "u1": "uint1",
    "uint1": "uint1",

    "s1": "uint1",
    "int1": "uint1",

    "bool": "bool",

    "f4_e2m1": "float4_e2m1",
    "fp4_e2m1": "float4_e2m1",
    "float4_e2m1": "float4_e2m1",
    "float4_e2m1fn": "float4_e2m1",
    "fp4_e2m1fn": "float4_e2m1",
    "f4_e2m1fn": "float4_e2m1",

    "f4_e1m2": "float4_e1m2",
    "fp4_e1m2": "float4_e1m2",
    "float4_e1m2": "float4_e1m2",
    "float4_e1m2fn": "float4_e1m2",
    "fp4_e1m2fn": "float4_e1m2",
    "f4_e1m2fn": "float4_e1m2",

    "f8_e8m0": "float8_e8m0",
    "fp8_e8m0": "float8_e8m0",
    "float8_e8m0": "float8_e8m0",

    "f8_e5m2": "float8_e5m2",
    "fp8_e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",

    "f8_e4m3fn": "float8_e4m3fn",
    "fp8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",

    "hif8": "hifloat8",
    "hifp8": "hifloat8",
    "hifloat8": "hifloat8",

    None: None
}


def get_torch_dtype(type_str):
    type_dict = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "hf32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "int64_p2": torch.uint8
    }
    return type_dict[type_str]


def gen_tensor_data(params, index):
    input_i_data = None
    shape_input = params["shape_input"][index]
    gen_dtype = params["dtype_input"][index]
    dtype_input = utils.get_np_dtype(gen_dtype)


    no_contiguous_input_index=[i for i in range(len(params["shape_input"]))]

    is_fix_value = False
    for key in params.keys():
        if f"tensor_data_{index}" in key and "required_" not in key:
            input_i_data = np.array(params[key]).reshape(shape_input).astype(gen_dtype)
            is_fix_value = True

    if not is_fix_value:
        dtype_input = utils.get_np_dtype(gen_dtype)
        tensor_duplicate_rat = None
        real_shape = None
        diff_data_size = None

        if len(params["range_input"][index]) == 1:
            input_i_data = gen_boundary_tensor_data(shape_input, params["range_input"][index], dtype_input,params["dtype_input"][index])
        else:
            range_input = params["range_input"][index]
            input_i_data = gen_nonbound_tensor_data(
                params,
                shape_input,
                range_input,
                dtype_input,
                index)

            if params["dtype_input"][index] == "hifloat8":
                input_i_data = trans_np_float_tensor_to_hifuint8(in_tensor=input_i_data, round_mode="hybrid",over_mode=True)
            elif params["dtype_input"][index] == "float8_e8m0":
                from en_dtypes import float8_e8m0
                input_i_data = input_i_data.view(np.uint32)
                input_i_data = (input_i_data << 1) >> 24
                input_i_data = 2 ** (input_i_data.astype(np.float64) - 127)
                input_i_data = input_i_data.astype(float8_e8m0)
                # input_i_data = input_i_data.astype(np.uint8)
            elif params["dtype_input"][index] == "uint1":
                np_uint8 = gen_nonbound_tensor_data(
                    params,
                    shape_input,
                    range_input,
                    np.uint8,
                    index)
                np_bool = np_uint8.astype('bool', copy=False)
                input_i_data = np.packbits(np_bool)
            elif params["dtype_input"][index] == "fp4_e2m1_as_fp32":
                choices = [0, -0.5, 0.5, -1, 1, -2, 2, -1.5, 1.5, -3, 3, -4, 4, -6, 6]
                input_i_data = np.random.choice(choices, size=shape_input).astype(np.float32)

    return input_i_data


def trans_np_float_tensor_to_hifuint8(in_tensor,round_mode="round", over_mode=True):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    if multi_shape == 1.0: # shape为空时，输出multi_shape可能为浮点数 1.0
        multi_shape = int(multi_shape)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    if in_tensor.dtype == np.float32:
        print(f"[trans_np_float_tensor_to_hifuint8]float32->hif8 total size :{multi_shape}")
        for i in range(multi_shape):
            out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    else:
        print(f"[trans_np_float_tensor_to_hifuint8]float16->hif8 total size :{multi_shape}")
        for i in range(multi_shape):
            out_tensor[i] = cvt_float16_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.astype(np.uint8)
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor


def _get_hif8_fraction_bits_number(exponent):
    #return dot value(4bits), exponent size, fraction size
    if exponent < -22:
        #zero
        return -1, 3, 0
    if -22 <= exponent < -15:
        #dml
        return 0, 3, 0
    if exponent == 0:
        #d0
        return 1, 0, 3
    if abs(exponent) == 1:
        #d1
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        #d2
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        #d3
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        #d4
        return 12, 4, 1
    if exponent > 15:
        #over flow
        return 12, 4, -1


def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = 16383  # b11111111111111
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192 #10 0000 0000 0000
        t14_values = fraction32_int & t14_mask
        hif8_value = 0

    else:
        #fp32 fraction is 23,keep hif8_bits_num bits
        #print(f"fraction32_int:{format(fraction32_int, 'b')} hif8_bits_num:{hif8_bits_num}")
        hif8_value = fraction32_int >> (23 - hif8_bits_num)
        #print(f"hif8_value:{hif8_value}")
        f14_t14 = fraction32_int - (hif8_value << (23 - hif8_bits_num))
        #print(f"f14_t14:{f14_t14}")
        #current do not deal hif8_bits_num > 9
        f14_values = f14_t14 >> (23 - hif8_bits_num - 14)
        t14_values = f14_t14 & t14_mask
    #print(f"f14_values:{f14_values} t14_values:{t14_values}")
    if f14_values >= t14_values:
        #carry bits
        if hif8_value == pow(2, hif8_bits_num) - 1:
            #carry exponent:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    #fp32 fraction is 23,keep hif8_bits_num + 1 bits
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num+1))
    #print(f"fraction32_int:{format(fraction32_int, 'b')} hif8_value_tmp:{format(hif8_value_tmp, 'b')}")
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        #carry exponent
        return True, 0
    elif hif8_value_tmp == 0:
        #zero
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        #carrys bits
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1


def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    if x < 0.0:
        sign = True
        sign_int_value = 128
    if np.isinf(x) or x_abs >= over_value:
        #2^15 = 32768
        if sign:
            if over_mode:
                #b11101111 = 239
                return 239
            else:
                # b11101110 = 238
                return 238
        else:
            if over_mode:
                #b01101111 = 111
                return 111
            else:
                # b01101110 = 110
                return 110
    if np.isnan(x):
        if over_mode:
            #b10000000
            return 128
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    #precheck
    fraction_int = int(x_abs * pow(2, 23)*pow(2, -exponent) - pow(2, 23))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    #print(f"exponent:{exponent} fraction_int:{fraction_int} dot_hif8_value:{dot_hif8_value} exponent_hif8_bits:{exponent_hif8_bits} hif8_f_bits:{fraction_hif8_bits} cut_bit_type:{cut_bit_type}")
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknow round type")
        return 0
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = _get_hif8_fraction_bits_number(exponent)
        #print(f"after carry exponent:{exponent} dot_hif8_value:{dot_hif8_value} exponent_hif8_bits:{exponent_hif8_bits} hif8_f_bits:{fraction_hif8_bits_new}")
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < -23:
        #zero b00000000
        #print("zero")
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        #dml
        #print(f"dml exponet:{exponent}")
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        #d0
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
        #print(f"sign_int_value:{format(sign_int_value, 'b')} dot_int_value:{format(dot_int_value, 'b')} sig_exp:{format(sig_exp, 'b')} exponent_int_value:{format(exponent_int_value, 'b')} hif8_frac_value:{format(hif8_frac_value, 'b')}")
    return hif8_int_value


def float32_to_float4_e2m1fn(x: np.float32) -> int:
    # 解析 float32 的符号位、指数、尾数
    x_bits = np.array([x], dtype=np.float32).view(np.uint32)[0]
    sign = (x_bits >> 31) & 0x1  # 符号位
    exponent = ((x_bits >> 23) & 0xFF) - 127  # 指数部分（偏移为127）
    mantissa = (x_bits & 0x7FFFFF) / (1 << 23)  # 尾数部分（归一化）

    # 将 float32 转换为 float4_e2m1fn
    # 限制指数范围（例如只能表示 -1, 0, 1, 2 等）
    if exponent < -1:
        exponent = -1  # 下溢
    elif exponent > 2:
        exponent = 2  # 上溢

    # 尾数部分只保留一位（0或1）
    m = 1 if mantissa >= 0.5 else 0

    # 构建 float4_e2m1fn 的4位表示
    # 格式：sign (1 bit) | exponent (2 bits) | mantissa (1 bit)
    e2m1 = (sign << 3) | ((exponent + 1) << 1) | m  # 假设 exponent bias 为 1

    return e2m1


def gen_boundary_tensor_data(shape_input, input_range, dtype_input, dtype_raw='fp32'):
    input_i_data = None
    if input_range[0] == "null":
        input_i_data = np.array([], dtype=dtype_input)
    elif input_range[0] == "nan":
        try:
            input_i_data = np.full(shape_input, np.NaN, dtype=dtype_input)
        except Exception:
            input_i_data = np.full(shape_input, np.nan, dtype=dtype_input)
    elif input_range[0] == "inf":
        try:
            input_i_data = np.full(shape_input, np.Inf, dtype=dtype_input)
        except Exception:
            input_i_data = np.full(shape_input, np.inf, dtype=dtype_input)
    elif input_range[0] == "-inf":
        try:
            input_i_data = np.full(shape_input, -np.Inf, dtype=dtype_input)
        except Exception:
            input_i_data = np.full(shape_input, -np.inf, dtype=dtype_input)
    elif input_range[0] == "random" and dtype_raw in ['fp4_e1m2', 'float4_e1m2']:
        from libs.generate_float_data import random_fp4_2_uint8
        input_i_data = random_fp4_2_uint8(shape_input, 1)
    elif input_range[0] == "random" and dtype_raw in ['fp4_e2m1', 'float4_e2m1']:
        from libs.generate_float_data import random_fp4_2_uint8
        input_i_data = random_fp4_2_uint8(shape_input, 2)
    elif input_range[0] == "default":
        input_i_data = None
    else:
        input_i_data = np.full(shape_input, input_range[0], dtype=dtype_input)
    return input_i_data


def gen_nonbound_tensor_data(params, data_shape, range_input, dtype, tensor_index):
    fuzz_value_type = ""
    try:
        if isinstance(range_input, dict):
            fuzz_value_type = "normal"
        elif isinstance(range_input, list):
            fuzz_value_type = "uniform"
        else:
            utils.print_log(f"[ERROR] input data range {range_input} is invalid.")
            return INVALID_VALUE
        hash_seed = -1
        if 0 in data_shape:
            json_path = os.path.join(params['case_path'], params['case_name'] + ".json")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            index = 0
            for input_info in json_data["input"]:
                if "shape" in input_info.keys():
                    if input_info['name'] == f'input{index}' and 0 in input_info['shape']:
                        view_shape = input_info['shape']
                        inp = torch.randn(view_shape)
                        stride = inp.stride()
                        input_info['view_shape'] = list(view_shape)
                        input_info['stride'] = list(stride)
                        json_data["input"][index] = input_info
                elif "shapes" in input_info.keys():
                    if input_info['name'] == f'input{index}':
                        shape_list = input_info["shapes"]
                        view_shape_list = shape_list
                        stride_list = []
                        for view_shape in view_shape_list:
                            inp = torch.randn(view_shape)
                            stride = inp.stride()
                            stride_list.append(list(stride))
                        input_info['view_shape'] = view_shape_list
                        input_info['stride'] = stride_list
                        json_data["input"][index] = input_info
                index += 1
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)
        min_value, max_value = range_input
        if min_value == "-inf" and max_value == "inf":
            if dtype in [np.float32, np.float64, np.float16]:
                min_value = np.finfo(dtype).min
                max_value = np.finfo(dtype).max
            elif dtype in [np.int8, np.int16, np.int32, np.uint8]:
                min_value = np.iinfo(dtype).min
                max_value = np.iinfo(dtype).max
            else:
                min_value = False
                max_value = True
            if hash_seed > 0:
                utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed}")
                np.random.seed(hash_seed)
            if dtype in [np.float64]:
                input_data = (np.random.uniform(low=-1, high=1, size=data_shape).astype(np.float64) * max_value)
            else:
                input_data = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
            if (data_shape == []):  # shape=[] reduce处理会异常
                return gen_uniform_data(data_shape, -1, 1, dtype, hash_seed)
            shape_len = reduce(operator.mul, data_shape)
            if (shape_len == 0):  # high不能为0
                return gen_uniform_data(data_shape, 0, 4, dtype, hash_seed)
            if hash_seed > 0:
                utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed}")
                np.random.seed(hash_seed)
            num_change = np.random.randint(low=0, high=min(shape_len, 4))
            if hash_seed > 0:
                utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed + 1}")
                np.random.seed(hash_seed + 1)
            inf_index = np.unravel_index(np.random.choice(shape_len, num_change), data_shape)
            input_data[inf_index] = min_value
            if hash_seed > 0:
                utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed + 2}")
                np.random.seed(hash_seed + 2)
            num_change = np.random.randint(low=0, high=min(shape_len, 4))
            if hash_seed > 0:
                utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed + 3}")
                np.random.seed(hash_seed + 3)
            inf_index = np.unravel_index(np.random.choice(shape_len, num_change), data_shape)
            input_data[inf_index] = max_value
            return input_data
        elif min_value == "nan":
            try:
                return np.full(data_shape, np.NaN, dtype=dtype)
            except Exception:
                return np.full(data_shape, np.nan, dtype=dtype)
        elif min_value == "inf":
            try:
                return np.full(data_shape, np.Inf, dtype=dtype)
            except Exception:
                return np.full(data_shape, np.inf, dtype=dtype)
        elif min_value == "-inf":
            try:
                return np.full(data_shape, -np.Inf, dtype=dtype)
            except Exception:
                    return np.full(data_shape, -np.inf, dtype=dtype)
        elif min_value == "default":
            return None
        elif min_value == "null":
            return np.array([], dtype=dtype)
        else:
            return gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed)
    except MemoryError:
        print(f"[ERROR] MemoryError.")
        return INVALID_VALUE



def gen_uniform_data(data_shape, min_value, max_value, dtype, hash_seed):
    print("----------- gen uniform data -----------")
    if min_value == 0 and max_value == 0:
        # print('--------using np.zeros to generate----------')
        return np.zeros(data_shape, dtype=dtype)
    if hash_seed > 0:
        utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed}")
        np.random.seed(hash_seed)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    if dtype == np.complex64:
        real = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(np.float32)
        real = torch.tensor(real)
        if hash_seed > 0:
            utils.print_log(f"[gen_uniform_data]set numpy seed:{hash_seed + 1}")
            np.random.seed(hash_seed + 1)
        imag = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(np.float32)
        imag = torch.tensor(imag)
        input_complex64 = torch.complex(real, imag)
        data = input_complex64.detach().numpy().astype(np.complex64)
    data = np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(dtype)
    utils.print_log(f"[gen_uniform_data]gen data shape:{data.shape},dtype:{data.dtype}")
    return data


def gen_normal_data(data_shape, mean, std, dtype, hash_seed):
    utils.print_log("----------- gen normal data -----------")
    if hash_seed > 0:
        utils.print_log(f"[gen_normal_data]set numpy seed:{hash_seed}")
        np.random.seed(hash_seed)
    if dtype == np.complex64:
        real = np.random.normal(mean, std, size=data_shape).astype(np.float32)
        real = torch.tensor(real)
        if hash_seed > 0:
            utils.print_log(f"[gen_normal_data]set numpy seed:{hash_seed + 1}")
            np.random.seed(hash_seed + 1)
            imag = np.random.normal(mean, std, size=data_shape).astype(np.float32)
        input_complex64 = torch.complex(real, imag)
        data = input_complex64.detach().numpy().astype(np.complex64)
    else:
        data = np.random.normal(mean, std, size=data_shape).astype(dtype)

    utils.print_log(f"[gen_normal_data]gen data shape:{data.shape},dtype:{data.dtype}")
    return data


def add_robustness_noise(data, noise_ratio=0.001, scale=1000):
    data = np.array(data, copy=True)
    total_elements = data.size
    utils.print_log(f"[add_robustness_noise]total_elements: {total_elements}")
    num_noisy_points = int(total_elements * noise_ratio)

    if num_noisy_points == 0:
        utils.print_log(f"[add_robustness_noise] warning: {noise_ratio} * {total_elements} = 0, no need inject noisy.")
        return data
    indices = np.random.choice(total_elements, size=num_noisy_points, replace=False)
    utils.print_log(f"[add_robustness_noise]inject noisy indices: {indices}")
    flat_data = data.ravel()
    flat_data[indices] *= scale
    return flat_data.reshape(data.shape)


def generate_input_tensors(params):
    input_tensors = {}
    
    for idx, tensor_name in utils.INPUT_TENSOR_INDEX_MAP.items():
        if idx >= len(params['shape_input']):
            continue
        np_dtype = utils.get_np_dtype(params['dtype_input'][idx])
        x_min = params['range_input'][idx]
        type_input = params['type_input'][idx]
        if type_input == 'tensor':
            input_file_i = os.path.join(params['case_path'], params['case_name'] + '_input_%s.bin' % idx)
            input_data_np = gen_tensor_data(params, idx)
            if idx in {15, 17} and "k_antiquant_mode" in params and params["k_antiquant_mode"] == 6 and len(gen_tensor_data(params, 21).shape) == 5:
                utils.print_log(f"for Nz, input_data{idx}.shape={input_data_np.shape}")
                input_data_np_Nz = input_data_np.transpose(0, 1, 2, 4, 3)
                utils.dump_file(input_data_np_Nz, input_file_i, params['dtype_input'][idx], params)
            else:
                utils.print_log("generate_input_tensors")
                utils.print_log(f"tensor_name={tensor_name}, shape={input_data_np.shape}")
                utils.dump_file(input_data_np, input_file_i, params['dtype_input'][idx], params)
            input_tensors[tensor_name] = utils.numpy_to_torch(input_data_np)
            utils.print_log('generating data completed: %s' % input_file_i)
    
    input_attrs = {
        "scale_value": params.get("scalevalue", 1.0),
        "layout_query": params.get("layout_query", "BSND"),
        "layout_kv": params.get("layout_kv", "PA_BSND"),
        "sparse_mode": params.get("sparsemode", 3),
        "key_quant_mode": params.get("key_quant_mode", 2),
        "value_quant_mode": params.get("value_quant_mode", 2),
        "sparse_block_size": params.get("sparse_blocksize", 1),
        "attention_mode": params.get("attention_mode", 2),
        "quant_scale_repo_mode": params.get("quant_scale_repo_mode", 1),
        "tile_size": params.get("tile_size", 128),
        "rope_head_dim": params.get("rope_head_dim", 64),
        "kv_dtype": params.get("kv_dtype", "float8_e4m3fn"),
        "pre_tokens": params.get("pre_tokens", (1 << 63) - 1),
        "next_tokens": params.get("next_tokens", (1 << 63) - 1),
    }
    
    test_case_data = {
        "testcase_name": params.get("case_name", ""),
        "params": params,
        "input_tensors": input_tensors,
        "input_attrs": input_attrs,
        "cpu_output": None,
    }
    
    return test_case_data


def generate_input_tensor_list(params):
    return generate_input_tensors(params)
