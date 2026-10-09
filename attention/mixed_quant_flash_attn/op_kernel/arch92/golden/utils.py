#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import random
import ctypes
import numpy as np
import datetime
import tensorflow as tf
import torch
import os
import sys


def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i + 1,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx + 1,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


# fuzz 中precision_method == 1的精度对比方式
def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return 100.0, "Pass"
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.size, data_compe.size)
        )
        return 0.0, result
    overflows_count = (
        data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    )

    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                data_compe[np.isinf(data_compe)][0:10],
                data_compe[np.isnan(data_compe)][0:10],
            )
        )

    if data_type == "bfloat16":
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.0078125
        atol = 0.0001
        max_error_idx = 10000000
    else:
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.005
        atol = 0.000025
        max_error_idx = 10000000

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    has_nan_inf = False
    if (
        "nan" in str(real_data)
        or "inf" in str(real_data)
        or "nan" in str(data_compe)
        or "inf" in str(data_compe)
    ):
        has_nan_inf = True

    if str(real_data.dtype) == "bfloat16":
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    elif str(real_data.dtype) == "float8_e4m3fn":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    elif str(real_data.dtype) == "float8_e5m2":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        nan_pos_inf = np.isposinf(real_data)
        real_data[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(real_data)
        real_data[nan_neg_inf] = -57344

        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        nan_pos_inf = np.isposinf(data_compe)
        data_compe[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(data_compe)
        data_compe[nan_neg_inf] = -57344

        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return fulfill_percent, result


def get_np_dtype(type_str):
    type_dict = {
        "fp64": np.float64,
        "fp32": np.float32,
        "fp16": np.float16,
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
        "uint64": np.uint64,
        "uint32": np.uint32,
        "uint16": np.uint16,
        "uint8": np.uint8,
        "bool": np.bool_,
        "complex64": np.complex64,
        "complex128": np.complex128,
        "complex32": np.float16,
        "bf16": tf.bfloat16.as_numpy_dtype,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
        "float4_e2m1": np.uint8,
        "float4_e1m2": np.uint8,
        "float8_e8m0": np.uint8,
        "hifloat8": np.uint8,
        "fp4_e1m2": np.uint8,
        "fp4_e2m1": np.uint8,
        "fp8_e8m0": np.uint8,
        "float32": np.float32,
        "float16": np.float16,
        "qint8": np.int8,
        "qint32": np.int32,
        "quint8": np.uint8,
        "qint16": np.int16,
        "uint1": np.uint8,
        "quint16": np.uint16,
        "fp4_e2m1_as_fp32": np.float32,
        "fp32_e6m2e1e16": np.float32,
    }
    if type_str == "int4":
        from ml_dtypes import int4

        return int4
    elif type_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    elif type_str == "fp8_e4m3" or type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    elif type_str == "fp8_e8m0" or type_str == "float8_e8m0fn":
        from ml_dtypes import float8_e8m0fnu

        return float8_e8m0fnu
    else:
        return type_dict[type_str]


def get_npu_dtype(type_str):
    import torch_npu

    type_dict = {
        "int8": torch_npu.int8,
        "fp16": torch_npu.float16,
        "bf16": torch_npu.bfloat16,
        "fp4_e1m2": torch_npu.float4_e1m2fn_x2,
        "fp4_e2m1": torch_npu.float4_e2m1fn_x2,
        "fp8_e8m0": torch_npu.float8_e8m0fnu,
        "fp8_e4m3": torch_npu.float8_e4m3fn,
        "float8_e5m2": torch_npu.float8_e5m2,
        "hifloat8": torch_npu.hifloat8,
        "int32": torch_npu.int32,
        "fp32": torch_npu.float32,
        "fp32_e6m2e1e16": torch_npu.float32,
    }
    return type_dict[type_str]


fp4_e1m2_values = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    -0.25,
    -0.5,
    -0.75,
    -1.0,
    -1.25,
    -1.5,
    -1.75,
]

fp4_e2m1_values = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# 数值 <-> 编码映射
value_to_code_e1m2 = {val: idx for idx, val in enumerate(fp4_e1m2_values)}
value_to_code_e2m1 = {val: idx for idx, val in enumerate(fp4_e2m1_values)}
code_to_value_e1m2 = {idx: val for idx, val in enumerate(fp4_e1m2_values)}
code_to_value_e2m1 = {idx: val for idx, val in enumerate(fp4_e2m1_values)}


def generate_fp4_tensor(shape, dtype):
    """
    生成FP4_E1M2/E2M1类型的Tensor（编码为uint8格式存储）
    """
    num_elements = torch.prod(torch.tensor(shape)).item()
    if dtype == "fp4_e1m2":
        sampled_values = random.choices(fp4_e1m2_values, k=num_elements)
        encoded_codes = [value_to_code_e1m2[val] for val in sampled_values]
    elif dtype == "fp4_e2m1":
        sampled_values = random.choices(fp4_e2m1_values, k=num_elements)
        encoded_codes = [value_to_code_e2m1[val] for val in sampled_values]
    else:
        print("error fp4 dtype")
    fp4_tensor_encoded = torch.tensor(encoded_codes, dtype=torch.uint8).reshape(shape)
    fp4_tensor = torch.tensor(sampled_values, dtype=torch.float32).reshape(shape)
    return fp4_tensor_encoded, fp4_tensor


def decode_fp4_tensor(nptensor, dtype):
    if dtype in ["float4_e1m2", "fp4_e1m2"]:
        decode_func = np.vectorize(lambda code: code_to_value_e1m2[code])
    elif dtype in ["float4_e2m1", "fp4_e2m1"]:
        decode_func = np.vectorize(lambda code: code_to_value_e2m1[code])
    else:
        print("error fp4 dtype")
    decoded_np = decode_func(nptensor).astype(np.float32)
    return decoded_np


def cvt_fp4_e2m1_to_bfloat16(x):
    Fp4e2m1ToBf16 = {
        "0": 0.0,
        "1": 0.5,
        "2": 1.0,
        "3": 1.5,
        "4": 2.0,
        "5": 3.0,
        "6": 4.0,
        "7": 6.0,
        "8": -0.0,
        "9": -0.5,
        "10": -1.0,
        "11": -1.5,
        "12": -2.0,
        "13": -3.0,
        "14": -4.0,
        "15": -6.0,
    }

    # x_abs = math.fabs(x)
    # # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    # x = eval(float_to_hex(x_abs))
    # x = x >> 16
    x = int(x)
    # first_fp4val = (x >> 4 )& 0x0f
    # second_fp4val = x & 0x0f
    first_fp4val = x & 0x0F

    # print("======first_fp4val====",first_fp4val)
    # print("======second_fp4val====",second_fp4val)
    first_fp4str = str(first_fp4val)

    # print("======first_fp4str====",first_fp4str)

    return Fp4e2m1ToBf16[first_fp4str]


def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {
        "0": 0.0,
        "1": 0.25,
        "2": 0.5,
        "3": 0.75,
        "4": 1.0,
        "5": 1.25,
        "6": 1.5,
        "7": 1.75,
        "8": -0.0,
        "9": -0.25,
        "10": -0.5,
        "11": -0.75,
        "12": -1.0,
        "13": -1.25,
        "14": -1.5,
        "15": -1.75,
    }

    # x_abs = math.fabs(x)
    # # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    # x = eval(float_to_hex(x_abs))
    # x = x >> 16
    x = int(x)
    # first_fp4val = (x >> 4 )& 0x0f
    # second_fp4val = x & 0x0f
    first_fp4val = x & 0x0F
    first_fp4str = str(first_fp4val)
    # print("======first_fp4str====",first_fp4str)
    return Fp4e1m2ToBf16[first_fp4str]


def new_trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)


def new_trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """Convert binary data into the floating point value"""
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2**b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2**exponent) * mant_mult


def trans_tensor_fp8_e8m0_to_bf16(tensor):
    # import pdb;pdb.set_trace()
    from ml_dtypes import bfloat16

    # 创建一个与输入tensor形状相同的新tensor
    new_tensor = np.zeros_like(tensor).astype(bfloat16)
    # 使用nditer遍历tensor中的每个元素
    with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            # 将元素传递给trans函数进行处理
            # new_value = new_value.astype(bfloat16)

            # x = float(x)
            # new_value = 2 ** (x - 127)
            new_value = ieee_754_conversion(0, int(x), 0)
            # 将处理后的结果回填到新tensor中
            new_tensor[it.multi_index] = new_value
    return new_tensor


def float32_to_float8_e6m2(x):
    # support normal only
    if x < 2 ** (-48):
        x = 2 ** (-48)
    if x > 1.5 * 2 ** (15):
        x = 1.5 * 2 ** (15)

    x_ptr = ctypes.pointer(ctypes.c_float(x))
    x_u32 = ctypes.c_uint32
    uint32_ptr = ctypes.cast(x_ptr, ctypes.POINTER(x_u32))
    x_u32 = uint32_ptr.contents.value
    f32_e = x_u32 >> 23
    f8_e = f32_e - 127 + 48
    m = (x_u32 >> 21) & 0x3
    res = np.uint8(f8_e << 2 | m)
    # print("x, res", x, res)
    return res


def gen_tensor_fp32_e6m2e1e16(shape_input, data_range):
    input_min, input_max = data_range
    # Range设定E8_1取值范围
    tensor = np.random.uniform(low=input_min, high=input_max, size=shape_input).astype(
        np.float32
    )
    new_tensor = np.zeros(shape_input).astype(np.uint32)
    with np.nditer(tensor, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            E6M2_1 = float32_to_float8_e6m2(x)
            if input_min == input_max:  # 特殊场景, E1_8 E1_16全1
                E1_8 = np.uint8(255)
                E1_16 = np.uint16(65535)
            else:
                E1_8 = np.uint8(np.random.randint(0, 255))
                E1_16 = np.uint16(np.random.randint(0, 65535))
                max_pos_e8 = np.random.randint(0, 8)
                E1_8 = E1_8 | (1 << max_pos_e8)  # 确保E1_8中必然有个1
                for k in range(8):
                    if (
                        E1_8 & (1 << k)
                    ) != 0:  # E1_8 为1 对应的2个E1_16 [2k, 2k+1]中必然有1
                        pos = np.random.randint(0, 2)
                        E1_16 = E1_16 | (1 << (2 * k + pos))

            EG = np.uint32(E6M2_1) | np.uint32(E1_8) << 8 | np.uint32(E1_16) << 16
            # print(f"EG:{EG:X} E8_1:{E8_1:X} E1_8:{E1_8:X} E1_16:{E1_16:X}")
            new_tensor[it.multi_index] = EG
    return new_tensor.view(np.float32)


def trans_tensor_e6m2e1e16_to_bf16(tensor):
    # import pdb;pdb.set_trace()
    from ml_dtypes import bfloat16

    # 每4个共用一个
    new_tensor = np.zeros(
        [
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
            tensor.shape[4] * 16,
        ]
    ).astype(bfloat16)

    # 使用nditer遍历tensor中的每个元素
    tensor_u32 = tensor.view(np.uint32)
    with np.nditer(tensor_u32, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            E6M2_1 = x & 0xFF

            e6m2_e = E6M2_1 >> 2
            e6m2_m0 = E6M2_1 & 0x1
            e6m2_m1 = (E6M2_1 & 0x3) >> 1

            e6m2_val = 2.0 ** (e6m2_e - 48) * (1 + e6m2_m1 * 0.5 + e6m2_m0 * 0.25)

            E1_8 = np.zeros(8).astype(np.int32)
            for k in range(8):
                E1_8[k] = (x >> (8 + k)) & 0x1  # 8~15bit

            E1_8x2 = np.zeros(16).astype(np.int32)
            for k in range(8):
                E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]

            E1_16 = np.zeros(16).astype(np.int32)
            for k in range(16):
                E1_16[k] = (x >> (16 + k)) & 0x1  # 16~31bit
            E16G = E1_16 + E1_8x2  # Fused 16 exp vals

            S16G = np.zeros(16).astype(bfloat16)
            for k in range(16):
                S16G[k] = (e6m2_val * 2 ** (E16G[k])).astype(bfloat16)

            # print(f"u32 hex: {x:X}")
            # print("E1_16  = ", E1_16)
            # print("E1_8x2 = ", E1_8x2)
            # print("E8_1   = ", E8_1)
            # print("E16G   = ", E16G)
            # print("S16G   =",  S16G)

            new_tensor[it.multi_index[0]][it.multi_index[1]][it.multi_index[2]][
                it.multi_index[3]
            ][it.multi_index[4] * 16 : it.multi_index[4] * 16 + 16] = S16G

    return new_tensor


def pack_int4(src: np.ndarray):
    """
    Pack int4 numpy array (each int4 number stored in one byte actually) to be continuous bytes stored in uint8
    """
    pack_size = 2
    shift = np.array([0, 4], dtype=np.uint8)
    array = src
    if src.size % pack_size != 0:
        array = np.pad(
            src.flatten(), (0, pack_size - src.size % pack_size), mode="constant"
        )
    reshaped = array.reshape([-1, 2])
    # bitwise_and is for arm
    out = np.sum(
        np.bitwise_and(reshaped.view(np.uint8), 0b00001111) << shift,
        axis=1,
        dtype=np.uint8,
    )
    return out


def unpack_int4(src: np.ndarray):
    from ml_dtypes import int4

    """
    Unpack uint8 numpy array to int4 array
    """
    # int4 = numpy_int4()
    shift = np.array([0, 4], dtype=np.uint8)
    src.dtype = np.uint8
    return (
        np.bitwise_and(src.reshape([-1, 1]) >> shift, 0b00001111)
        .astype(int4)
        .reshape([-1])
    )


def pack_fp4_indices(indices_fp4):
    """将FP4索引对打包为uint8数组。
    假设indices_fp4是uint8数组，每个元素在[0,15]区间，低4位有效。
    结果数组形状为 (N//2,)，若原始长度为奇数，最后一个字节的高4位将无效。
    """
    # 检查并确保输入是正确的uint8类型
    if indices_fp4.dtype != np.uint8:
        indices_fp4 = indices_fp4.astype(np.uint8)

    # 确保长度是偶数，否则用0填充
    if len(indices_fp4) % 2 != 0:
        indices_fp4 = np.append(indices_fp4, [0])

    # 拆分为奇偶索引
    even_indices = indices_fp4[0::2]  # 低位 (索引0,2,4...)
    odd_indices = indices_fp4[1::2] << 4  # 奇索引左移4位到高半字节

    # 打包：低半字节(偶索引) + 高半字节(奇索引)
    packed = (odd_indices | even_indices).astype(np.uint8)
    return packed


def numpy_bfloat16():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            # noinspection PyUnresolvedReferences
            import tensorflow

            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                "Please install with `pip3 install ml-dtypes` "
                "or `pip3 install tensorflow`"
            )
    # some older TF version (v1.15.0) bfp16 needs to convert to fp32 to calculate in numpy
    global BFP16_NEEDS_FP32_FOR_NPY
    if BFP16_NEEDS_FP32_FOR_NPY is None:
        test_array = np.array([1, 2, 3], dtype=bfloat16)
        try:
            np.max(test_array)
        except TypeError:
            # TypeError: No loop matching the specified signature
            # and casting was found for ufunc maximum
            BFP16_NEEDS_FP32_FOR_NPY = True
        else:
            BFP16_NEEDS_FP32_FOR_NPY = False
    return bfloat16


def trans_tensor_e8e1e16_to_bf16(tensor):
    # 每4个共用一个
    new_tensor = np.zeros(
        [
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
            tensor.shape[4] * 16,
        ]
    ).astype(numpy_bfloat16())

    # 使用nditer遍历tensor中的每个元素
    tensor_u32 = tensor.view(np.uint32)
    with np.nditer(tensor_u32, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for x in it:
            E8_1 = (x & 0xFF) - 127  # 0~7bit
            E1_8 = np.zeros(8).astype(np.int32)
            for k in range(8):
                E1_8[k] = (x >> (8 + k)) & 0x1  # 8~15bit

            E1_8x2 = np.zeros(16).astype(np.int32)
            for k in range(8):
                E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]

            E1_16 = np.zeros(16).astype(np.int32)
            for k in range(16):
                E1_16[k] = (x >> (16 + k)) & 0x1  # 16~31bit
            E16G = E1_16 + E1_8x2 + E8_1  # Fused 16 exp vals

            S16G = np.zeros(16).astype(numpy_bfloat16())
            for k in range(16):
                S16G[k] = 2 ** (E16G[k]).astype(numpy_bfloat16())

            new_tensor[
                it.multi_index[0],
                it.multi_index[1],
                it.multi_index[2],
                it.multi_index[3],
                it.multi_index[4] * 16 : it.multi_index[4] * 16 + 16,
            ] = S16G

    return new_tensor


import numpy as np


def split_float32_to_uint16(arr: np.ndarray) -> np.ndarray:
    # 确保输入为 float32 类型
    arr = arr.astype(np.float32, copy=False)
    # 视为 uint32 二进制表示
    uint32_view = arr.view(np.uint32)
    # 提取高 16 位和低 16 位
    high = (uint32_view >> 16).astype(np.uint16)
    low = (uint32_view & 0xFFFF).astype(np.uint16)
    # 沿新轴堆叠
    return np.concatenate([low, high], axis=-1)


def combine_uint16_to_float32(arr_uint16: np.ndarray) -> np.ndarray:
    """
    将拆分后的 uint16 数组聚合回 float32 数组。

    参数:
        arr_uint16: shape (n, s, d, 2) 的 uint16 数组，最后一维 [high, low]

    返回:
        shape (n, s, d) 的 float32 数组
    """
    high = arr_uint16[..., 0].astype(np.uint32) << 16
    low = arr_uint16[..., 1].astype(np.uint32)
    combined = high | low
    return combined.view(np.float32)
