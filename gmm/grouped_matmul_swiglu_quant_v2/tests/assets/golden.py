#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GroupedMatmulSwigluQuantV2 Golden"""
import math
import numpy as np

__golden__ = {"kernel": {"grouped_matmul_swiglu_quant_v2": "grouped_matmul_swiglu_quant_v2_golden"}}


def numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
        return bfloat16
    except ModuleNotFoundError:
        raise RuntimeError("ml-dtypes is needed to support bfloat16 dtype!!! "
                           "Please install with `pip3 install ml-dtypes`")


def float_to_hex(f):
    import struct
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def cvt_bfloat16_to_fp4_e2m1(x):
    import math
    sRet = 0
    if x < 0.0:
        sRet = 1
    x_abs = math.fabs(x)
    x = eval(float_to_hex(x_abs))
    x = x >> 16
    ef = (x >> 7) & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 1
    maxExp = 3
    expBias = 1
    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0):
        eNorm = ef - 127 + 1
    else:
        eNorm = ef - 127
    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 1)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = 1 if ((mf & (0x1 << mLenDelta)) != 0) and ((mf & ((0x1 << (mLenDelta - 1)) - 1)) != 0 or (mf & (0x1 << (mLenDelta - 1))) != 0) else 0
        mRet = (mf >> mLenDelta)
        if needRound:
            mRet += 1
    else:
        eRet = (eNorm + expBias)
        needRound = 1 if ((mf & (0x1 << mLenDelta)) != 0) and ((mf & ((0x1 << (mLenDelta - 1)) - 1)) != 0 or (mf & (0x1 << (mLenDelta - 1))) != 0) else 0
        mRet = (mf >> mLenDelta)
        if needRound:
            mRet += 1
        if (((mRet & 0b10) != 0) and needRound):
            eRet += 1
            mRet = 0
    if (eRet >= 3):
        eRet = 3
    elif (eRet == 0 and mRet == 0b10):
        eRet += 1
        mRet = 0
    return (((sRet) << 3) | ((eRet) << 1) | ((mRet) & 1))


def cvt_bfloat16_to_fp4_e1m2(x):
    import math
    sRet = 0
    if x < 0.0:
        sRet = 1
    x_abs = math.fabs(x)
    x = eval(float_to_hex(x_abs))
    x = x >> 16
    ef = x >> 7 & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 2
    maxExp = 1
    expBias = 1
    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0):
        eNorm = ef - 127 + 1
    else:
        eNorm = ef - 127
    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = 1 if ((mf & (0x1 << mLenDelta)) != 0) and ((mf & ((0x1 << (mLenDelta - 1)) - 1)) != 0 or (mf & (0x1 << (mLenDelta - 1))) != 0) else 0
        mRet = (mf >> mLenDelta)
        if needRound:
            mRet += 1
    else:
        eRet = (eNorm + expBias)
        needRound = 1 if ((mf & (0x1 << mLenDelta)) != 0) and ((mf & ((0x1 << (mLenDelta - 1)) - 1)) != 0 or (mf & (0x1 << (mLenDelta - 1))) != 0) else 0
        mRet = (mf >> mLenDelta)
        if needRound:
            mRet += 1
        if (((mRet & 0b100) != 0) and needRound):
            eRet += 1
            mRet = 0
    if (eRet >= 1):
        eRet = 1
    elif (eRet == 0 and mRet == 0b100):
        eRet += 1
        mRet = 0
    return (((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3))


def trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    import struct
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)

    Fp4e2m1ToBf16 = {0: 0x0, 1: 0x3F00, 2: 0x3F80, 3: 0x3FC0,
                      4: 0x4000, 5: 0x4040, 6: 0x4080, 7: 0x40C0,
                      8: 0x8000, 9: 0xBF00, 10: 0xBF80, 11: 0xBFC0,
                      12: 0xC000, 13: 0xC040, 14: 0xC080, 15: 0xC0C0}
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)

    for i in range(multi_shape):
        first_fp4val = in_tensor[i] & 0x0f
        second_fp4val = (in_tensor[i] >> 4) & 0x0f
        bfloat16_tensor[i * 2] = Fp4e2m1ToBf16.get(int(first_fp4val), 0)
        bfloat16_tensor[i * 2 + 1] = Fp4e2m1ToBf16.get(int(second_fp4val), 0)
        fp32_tensor[i * 2] = struct.unpack('!f', struct.pack('!I', bfloat16_tensor[i * 2] << 16))[0]
        fp32_tensor[i * 2 + 1] = struct.unpack('!f', struct.pack('!I', bfloat16_tensor[i * 2 + 1] << 16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


def trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    import struct
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)

    Fp4e1m2ToBf16 = {0: 0x0, 1: 0x3E80, 2: 0x3F00, 3: 0x3F40,
                      4: 0x3F80, 5: 0x3FA0, 6: 0x3FC0, 7: 0x3FE0,
                      8: 0x8000, 9: 0xBE80, 10: 0xBF00, 11: 0xBF40,
                      12: 0xBF80, 13: 0xBFA0, 14: 0xBFC0, 15: 0xBFE0}
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)

    for i in range(multi_shape):
        first_fp4val = in_tensor[i] & 0x0f
        second_fp4val = (in_tensor[i] >> 4) & 0x0f
        bfloat16_tensor[i * 2] = Fp4e1m2ToBf16.get(int(first_fp4val), 0)
        bfloat16_tensor[i * 2 + 1] = Fp4e1m2ToBf16.get(int(second_fp4val), 0)
        fp32_tensor[i * 2] = struct.unpack('!f', struct.pack('!I', bfloat16_tensor[i * 2] << 16))[0]
        fp32_tensor[i * 2 + 1] = struct.unpack('!f', struct.pack('!I', bfloat16_tensor[i * 2 + 1] << 16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


def convert_to_high_precision(input_tensor, input_type):
    import torch
    if input_type in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))
    elif input_type in ("int4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.int32)).to(torch.int32)
    else:
        input_tensor = torch.from_numpy(input_tensor).to(torch.int32)
    return input_tensor


def transform_tensor(input_tensor):
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def single_group_mm_cal(x1, x2):
    import torch
    return torch.matmul(x1, x2)


def swiglu(gmm_out):
    import torch
    c_temp4, gate = gmm_out.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)
    c_temp6 = c_temp5 * gate
    return c_temp6


def _mx_reshape_to_blocks(fp_array, axis, block_size):
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, 'constant')
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_calculate_share_exp(fp_array, scale_axis, mx_quant_dtype):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    ele_emax = 0
    if mx_quant_dtype == 0:
        ele_emax = 8
    elif mx_quant_dtype == 1:
        ele_emax = 15
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis + 1, keepdims=True)
    res = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax

    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array, round_mode):
    if round_mode in ("rint", "even"):
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        fp_array = np.sign(fp_array) * np.floor(
            np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
    elif round_mode == "floor":
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
    exp_bits = 0
    mantissa_bits = 0
    emax = 0
    max_norm = 0
    if mx_ele_dtype == 0:
        exp_bits = 4
        mantissa_bits = 3
        emax = 8
        max_norm = 448
    elif mx_ele_dtype == 1:
        exp_bits = 5
        mantissa_bits = 2
        emax = 15
        max_norm = 57344

    ret = fp_array / (2 ** share_exp + (share_exp == -float("inf"))) - (share_exp == -float("inf")) * fp_array
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))
                            ).astype(fp_array.dtype, copy=False)
    min_exp = -(2 ** (exp_bits - 1)) + exp_bits
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    return ret


def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array


def numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                            "Please install with `pip3 install en-dtypes`")


def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                            "Please install with `pip3 install ml-dtypes`")


def numpy_float8_e5m2():
    try:
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                            "Please install with `pip3 install ml-dtypes`")


def quant(swiglu_out, mx_quant_dtype):
    axis = 1
    block_size = 32
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(swiglu_out, axis, block_size)
    round_mode = "rint"
    share_exp = _mx_calculate_share_exp(fp_array, axis, mx_quant_dtype)
    scale_emax = 2 ** (8 - 1) - 1
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_quant_dtype, round_mode)
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)

    target_quant_dtype = ""
    if mx_quant_dtype == 0:
        target_quant_dtype = "float8_e4m3fn"
    elif mx_quant_dtype == 1:
        target_quant_dtype = "float8_e5m2"

    ele_dtype_np = eval(f"numpy_{target_quant_dtype}()")
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = scale_array.astype(numpy_float8_e8m0(), copy=False)
    return ele_array, scale_array


def grouped_matmul_swiglu_quant_v2_golden(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7,
                                        input_dtypes, output_dtypes, **kwargs):
    import torch

    x1, x_scale, group_list, x2, weight_scale, _, _, _ = input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
    x1_dtype, x_scale_dtype, _, x2_dtype, weight_scale_dtype, _, _, _ = input_dtypes[0], input_dtypes[1], input_dtypes[2], input_dtypes[3], input_dtypes[4], input_dtypes[5], input_dtypes[6], input_dtypes[7]

    trans_b = kwargs.get("transpose_weight", False)
    group_list_type = kwargs.get("group_list_type", 0)
    dequant_mode = kwargs.get("dequant_mode", 0)
    dequant_dtype = kwargs.get("dequant_dtype", 0)
    shape_m = x1.shape[0]
    shape_n = x2.shape[-1] if not trans_b else x2.shape[-2]
    shape_k = x1.shape[1]
    mx_quant_dtype = kwargs.get("quant_dtype", 0)

    grouped_quant_output = []
    grouped_quant_scale_output = []
    group_num = len(group_list)
    is_mx_quant = x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2") and weight_scale_dtype == "float8_e8m0"

    if group_list_type == 1:
        group_list = np.cumsum(group_list)
    if is_mx_quant:
        pertoken_scale_mx = x_scale
        m, k0, k1 = pertoken_scale_mx.shape
        pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
        x1_local = input_0
        pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
        x1_dims = len(x1_local.shape)
        x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1_local.shape[-1]
        x1_local = np.pad(x1_local, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x1_local = x1_local * pertoken_scale_mx_broadcast
        x1_local = convert_to_high_precision(x1_local, x1_dtype)
        for i in range(group_num):
            scale_g = weight_scale[i]
            if trans_b == False:
                scale_g = transform_tensor(scale_g)
            else:
                n, k0, k1 = scale_g.shape
                scale_g = scale_g.reshape(n, k0 * k1)

            x2_local = input_3[i]
            if trans_b:
                x2_local = np.swapaxes(x2_local, -1, -2)
                scale_g = scale_g.transpose()

            deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
            x2_dims = len(x2_local.shape)
            x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2_local.shape[-2]
            x2_local = np.pad(x2_local, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len), (0, 0)], mode='constant', constant_values=0)
            x2_local = x2_local * deq_scale_mx_broadcast

            if i == 0:
                x1_temp = x1_local[:group_list[i], :]
            else:
                x1_temp = x1_local[group_list[i - 1]:group_list[i], :]
            x2_local = convert_to_high_precision(x2_local, x2_dtype)
            gmm_out = single_group_mm_cal(x1_temp, x2_local)
            torch.set_printoptions(threshold=torch.inf)
            has_inf = torch.isinf(gmm_out).any()
            has_nan = torch.isnan(gmm_out).any()

            print("Whether contains inf:", has_inf.item())
            print("Whether contains nan:", has_nan.item())
            swiglu_out = swiglu(gmm_out)
            quant_output, quant_scale_output = quant(swiglu_out, mx_quant_dtype)
            grouped_quant_output.append(quant_output)
            grouped_quant_scale_output.append(quant_scale_output)

    data_output = grouped_quant_output if not grouped_quant_output else np.concatenate(grouped_quant_output, axis=0)
    data_output = data_output.reshape(group_list[-1], int(shape_n / 2))

    quant_scale_output = grouped_quant_scale_output if not grouped_quant_output else np.concatenate(grouped_quant_scale_output, axis=0)

    quant_scale_output = quant_scale_output.reshape(group_list[-1], int(shape_n / 2 / 64), 2)
    return data_output, quant_scale_output
