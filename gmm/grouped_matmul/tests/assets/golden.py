#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GroupedMatmul Golden"""
import math
import numpy as np

__golden__ = {"kernel": {"grouped_matmul": "grouped_matmul_golden"}}


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def bfloat16_conversion(container):
    from ml_dtypes import bfloat16
    ret = list(container)
    special_dtypes = ("bfloat16", "int4",
                      "float8_e5m2", "float8_e4m3fn", "float8_e8m0",
                      "float4_e2m1", "float4_e1m2",
                      "hifloat8", "hifloat4")
    for sd in special_dtypes:
        for idx, dtype in enumerate(ret):
            if not isinstance(dtype, str):
                continue
            if sd == dtype:
                ret[idx] = bfloat16
    return ret


def numpy_bfloat16():
    from ml_dtypes import bfloat16
    return bfloat16


def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


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


def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _mx_calculate_share_exp(fp_array, scale_axis, mx_quant_dtype):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    ele_emax = 0
    if mx_quant_dtype == 0:
        ele_emax = 8
    elif mx_quant_dtype == 1:
        ele_emax = 15
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis+1, keepdims=True)
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
    import struct
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
    import struct
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
    if input_type in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8", "hifloat4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))
    elif input_type in ("int4"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.int32)).to(torch.int32)
    else:
        input_tensor = torch.from_numpy(input_tensor).to(torch.int32)
    return input_tensor


def scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32_deq_scale &= 0XFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
    return fp32_deq_scale


def get_total_scale_k_for_split_k(k, group_size, group_num):
    return k // group_size + group_num


def infer_quant_mode(scale, pertoken_scale, trans_b, shape_m, shape_k, shape_n, group_num, split_m, split_k, gs_k, gs_n):
    x_quant_type = set()
    w_quant_type = set()
    if split_m:
        expect_scale_k = math.ceil(shape_k / gs_k)
    elif split_k:
        expect_scale_k = get_total_scale_k_for_split_k(shape_k, gs_k, group_num)

    if pertoken_scale is not None:
        if split_m:
            if len(pertoken_scale.shape) == 1:
                if pertoken_scale.shape[-1] == group_num:
                    x_quant_type.add("perTensor")
                elif pertoken_scale.shape[-1] == shape_m:
                    x_quant_type.add("perToken")
            else:
                if pertoken_scale.shape[-1] == expect_scale_k and pertoken_scale.shape[-2] == shape_m:
                    x_quant_type.add("perGroup")
                if pertoken_scale.shape[-1] == 1 and pertoken_scale.shape[-2] == group_num:
                    x_quant_type.add("perTensor")
                if pertoken_scale.shape[-1] == 1 and pertoken_scale.shape[-2] == shape_m:
                    x_quant_type.add("perToken")
        elif split_k:
            if len(pertoken_scale.shape) == 1:
                if pertoken_scale.shape[-1] == group_num:
                    x_quant_type.add("perTensor")
            else:
                if pertoken_scale.shape[-1] == 1 and pertoken_scale.shape[-2] == group_num:
                    x_quant_type.add("perTensor")
                if pertoken_scale.shape[-1] == shape_m and pertoken_scale.shape[-2] == group_num:
                    x_quant_type.add("perToken")
                if pertoken_scale.shape[-1] == shape_m and pertoken_scale.shape[-2] == expect_scale_k:
                    x_quant_type.add("perGroup")

    if len(scale.shape) == 1:
        if scale.shape[-1] == group_num:
             w_quant_type.add("perTensor")
    else:
        scale_k = scale.shape[-1] if trans_b else scale.shape[-2]
        scale_n = scale.shape[-2] if trans_b else scale.shape[-1]
        if scale.shape[-1] == 1 and scale.shape[-2] == group_num:
            w_quant_type.add("perTensor")
        if scale.shape[-1] == shape_n and scale.shape[-2] == group_num:
            w_quant_type.add("perChannel")
        if scale_k == expect_scale_k and scale_n == math.ceil(shape_n / gs_n):
            w_quant_type.add("perBlock")

    print(f"Quant mode list: X:{x_quant_type} W:{w_quant_type}")

    if len(w_quant_type) == 1 and len(x_quant_type) == 1:
        return next(iter(x_quant_type)), next(iter(w_quant_type))
    if "perGroup" in x_quant_type and "perBlock" in w_quant_type:
        return "perGroup", "perBlock"
    if "perTensor" in x_quant_type and "perTensor" in w_quant_type:
        return "perTensor", "perTensor"

    if "perGroup" in x_quant_type:
        x_quant_type.remove("perGroup")
    if "perBlock" in w_quant_type:
        w_quant_type.remove("perBlock")

    res_x = next(iter(x_quant_type)) if x_quant_type else None
    res_w = next(iter(w_quant_type)) if w_quant_type else None
    return res_x, res_w


def single_group_mm_cal(x1, x2, bias, pertoken_scale_slice, deq_scale_tensor, two_scale_tensor,
                        out_dtype, bias_dtype,
                        is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible):
    import torch
    out = torch.matmul(x1, x2)
    if bias is not None and bias_dtype == "int32":
        bias = torch.from_numpy(bias).to(torch.int32)
        out = torch.add(out, bias)
    if out_dtype == 'float16' or out_dtype == 'float32':
        if is_mx_quant:
            pass
        elif is_pertoken:
            out = out * deq_scale_tensor * pertoken_scale_slice
        elif is_two_scale:
            out = out * two_scale_tensor
        else:
            out = out * deq_scale_tensor
        if bias is not None and (bias_dtype == "float16" or bias_dtype == "float32"):
            bias_fp32 = torch.from_numpy(bias.astype(np.float32))
            out = (out + bias_fp32).numpy().astype(out_dtype)
        else:
            out = out.numpy().astype(out_dtype)
    else:
        if is_mx_quant or is_hif4:
            pass
        elif is_pertoken:
            out = out * deq_scale_tensor * pertoken_scale_slice
        elif is_two_scale:
            out = out * two_scale_tensor
        elif is_compatible:
            if deq_scale_tensor.shape[-1] == 1 and (bias_dtype != "bfloat16" and bias_dtype != "float32"):
                scale = scale_generate(deq_scale_tensor.numpy())
                deq_scale_tensor = torch.unsqueeze(torch.from_numpy(scale), dim=1).to(torch.float32)
            out = out * deq_scale_tensor
        else:
            out = out * deq_scale_tensor
        if bias is not None and (bias_dtype == "bfloat16" or bias_dtype == "float16" or bias_dtype == "float32"):
            bias_fp32 = torch.from_numpy(bias.astype(np.float32))
            out = (out + bias_fp32).numpy().astype(out_dtype)
        else:
            out = out.numpy().astype(out_dtype)
    return out


def pertile_mm_cal(x1, x2, pertoken_scale, scale, out_dtype):
    import torch
    group_size_k = 128
    group_size_n = 128
    M = x1.shape[0]
    K = x2.shape[-2]
    N = x2.shape[-1]
    out = torch.zeros(M, N)
    scale_n = torch.repeat_interleave(scale, repeats=group_size_n, dim=-1)
    scale_n = scale_n[:, :N]
    for k_idx in range((K + group_size_k - 1) // group_size_k):
        k_start = k_idx * group_size_k
        k_end = min((k_idx + 1) * group_size_k, K)
        scale_mul = pertoken_scale[:, k_idx].unsqueeze(1) * scale_n[k_idx, :].unsqueeze(0)
        out += torch.matmul(x1[:, k_start:k_end], x2[k_start:k_end, :]) * scale_mul
    if out_dtype == 'bfloat16':
        out_dtype = bfloat16_conversion([out_dtype])[0]
    out = (out).numpy().astype(out_dtype)
    return out


def transform_tensor(input_tensor):
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def reshape_last_two_dims(x, c0=32):
    def ceil_align(d, align):
        return (d + align - 1) // align * align

    original_shape = np.array(x.shape)
    ndim = x.ndim

    if ndim < 2:
        raise ValueError("Input array must have at least 2 dimensions")

    d1, d2 = original_shape[-2], original_shape[-1]
    aligned_d1 = ceil_align(d1, 16)
    aligned_d2 = ceil_align(d2, c0)
    pad_d1 = aligned_d1 - d1
    pad_d2 = aligned_d2 - d2
    pad_width = [(0, 0)] * (ndim - 2) + [(0, pad_d1), (0, pad_d2)]
    x_padded = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    new_d1 = aligned_d1 // 16
    new_d2 = aligned_d2 // c0
    new_shape = list(original_shape[:-2]) + [new_d1, 16, new_d2, c0]
    x_reshaped = x_padded.reshape(new_shape)
    transpose_order = list(range(ndim - 2)) + [ndim, ndim - 2, ndim - 1, ndim + 1]
    x_transposed = np.transpose(x_reshaped, axes=transpose_order)
    return x_transposed


def trans_nz2nd(input_data):
    nd_data = input_data.transpose(0, 2, 3, 1, 4).reshape(input_data.shape[0],
                                                          input_data.shape[2] * input_data.shape[3],
                                                          input_data.shape[1] * input_data.shape[4])
    return nd_data


def trans_weight_fp4(x1_dtype, x2_dtype, x2):
    if x2_dtype == 'float4_e1m2':
        if x1_dtype == 'bfloat16':
            x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)
        else:
            x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32).astype(np.float16)
    else:
        if x1_dtype == 'bfloat16':
            x2 = trans_np_fp4_e2m1_tensor_to_bfloat16(x2).astype(np.float32)
        else:
            x2 = trans_np_fp4_e2m1_tensor_to_bfloat16(x2).astype(np.float32).astype(np.float16)
    return x2


def get_per_group_size(antiquant_scale, k_size, trans_b):
    gn = antiquant_scale.shape[-2] if not trans_b else antiquant_scale.shape[-1]
    assert gn > 0 and k_size % gn == 0
    group_size = k_size // gn
    print(f"antiquant groupSize: {group_size}, k_size: {k_size}", flush=True)
    return group_size


def get_mx_scale(antiquant_scale, group_size, pertoken_scale, trans_b, x1_dtype):
    import torch
    print(f"kankan antiquant_scale 00000000000 {antiquant_scale.dtype}", flush=True)
    antiquant_scale = antiquant_scale.astype(np.float32)
    antiquant_scale_broadcast = np.repeat(antiquant_scale, group_size, axis=-2) if not trans_b else np.repeat(
        antiquant_scale, group_size, axis=-1)
    print(f"kankan antiquant_scale_broadcast shape 00000000000 {antiquant_scale_broadcast.shape}", flush=True)
    antiquant_scale_broadcast = torch.from_numpy(antiquant_scale_broadcast)
    if x1_dtype == "float16":
        antiquant_scale_broadcast = antiquant_scale_broadcast.to(torch.float16)

    if x1_dtype == "float8_e4m3fn":
        pertoken_scale_broadcast = torch.from_numpy(
            np.repeat(pertoken_scale.astype(np.float32), group_size, axis=-1))
        print(f"kankan pertoken_scale_broadcast shape 00000000000 {pertoken_scale_broadcast.shape}",
              flush=True)
    else:
        pertoken_scale_broadcast = None

    return antiquant_scale_broadcast, pertoken_scale_broadcast


def get_group_m(group_list, group_list_type, i):
    if group_list_type == 0:
        start_m = group_list[i - 1] if i > 0 else 0
        end_m = group_list[i]
    else:
        start_m = sum(group_list[:i])
        end_m = start_m + group_list[i]
    print("GMM GOLDEN: ")
    print(start_m)
    print(end_m)
    return start_m, end_m


def cal_weight_quant_s8s4(x1, x2_ori, bias, scale, antiquant_scale, pertoken_scale, group_list, group_list_type,
                          y_dtype):
    import torch
    shape_k = x2_ori.shape[-2]
    group_size = get_per_group_size(antiquant_scale, shape_k, False)
    print("antiquant groupSize: {}, shape_k: {}".format(group_size, shape_k))

    x2_tmp = x2_ori.astype(np.float16)
    antiquant_scale = antiquant_scale.astype(np.float16)
    antiquant_scale_broadcast = np.repeat(antiquant_scale, group_size, axis=-2)
    x2_tmp = x2_tmp * antiquant_scale_broadcast

    x1 = torch.from_numpy(x1.astype(np.int32)).to(torch.int32)
    x2 = torch.from_numpy(np.around(x2_tmp).astype(np.int32)).to(torch.int32)

    outs = []
    for i in range(len(group_list)):
        start_m, end_m = get_group_m(group_list, group_list_type, i)
        x1_group = x1[start_m:end_m, :]
        weight_group = x2[i]

        group_out = torch.matmul(x1_group, weight_group).to(torch.int32)
        group_out = group_out.numpy().astype(np.float32) * scale[i].reshape(1, -1) * \
                    pertoken_scale.reshape(-1, 1)[start_m:end_m, :]

        if (bias is not None) and (len(bias) != 0):
            group_out += bias[i, :].reshape(1, -1)

        if y_dtype == "bfloat16":
            outs.append(group_out.astype(bfloat16_conversion([y_dtype])[0]))
        else:
            outs.append(group_out.astype(np.float16))

    return outs


def cal_weight_quant_mx(x1, x2, bias, antiquant_scale, pertoken_scale, group_list, group_list_type,
                        x1_dtype, x2_dtype, y_dtype, trans_b):
    import torch
    x1 = torch.from_numpy(x1.astype(np.float32))
    x2 = trans_weight_fp4(x1_dtype, x2_dtype, x2)
    print("cal_weight_quant_mx x2 shape 00000000000 ", x2.shape, flush=True)
    x2 = trans_nz2nd(x2)
    print("cal_weight_quant_mx x2 shape 11111111111 ", x2.shape, flush=True)
    x2 = torch.from_numpy(x2).to(torch.float16) if x1_dtype == "float16" else torch.from_numpy(x2)
    print(f"cal_weight_quant_mx x2 dtype 11111111111 {x2.dtype}", flush=True)

    shape_k = x2.shape[-2] if not trans_b else x2.shape[-1]
    shape_n = x2.shape[-1] if not trans_b else x2.shape[-2]
    print(f"cal_weight_quant_mx k{shape_k} n{shape_n}")

    if (bias is not None) and (len(bias) != 0):
        bias = torch.from_numpy(bias.astype(np.float32))
    else:
        bias = torch.from_numpy(np.zeros([x2.shape[0], shape_n], dtype=np.float32))

    group_size = get_per_group_size(antiquant_scale, shape_k, trans_b)
    assert group_size == 32

    antiquant_scale_broadcast, pertoken_scale_broadcast = get_mx_scale(antiquant_scale, group_size,
                                                                       pertoken_scale, trans_b, x1_dtype)
    outs = []
    for i in range(len(group_list)):
        start_m, end_m = get_group_m(group_list, group_list_type, i)
        x1_group = x1[start_m:end_m, :]
        weight_group = x2[i]
        antiquant_scale_broadcast_i = antiquant_scale_broadcast[i]
        weight_group = weight_group * antiquant_scale_broadcast_i
        if trans_b:
            weight_group = weight_group.T

        if x1_dtype == "float8_e4m3fn":
            x1_group = x1_group * pertoken_scale_broadcast[start_m:end_m, :]

        if x1_dtype == "bfloat16":
            x1_group = x1_group.to(torch.bfloat16)
            weight_group = weight_group.to(torch.bfloat16)

        group_out = torch.matmul(x1_group.to(torch.float32), weight_group.to(torch.float32))
        group_out += bias[i, :].reshape(1, -1).to(torch.float32)
        if y_dtype == "bfloat16":
            outs.append(group_out.numpy().astype(bfloat16_conversion([y_dtype])[0]))
        else:
            outs.append(group_out.numpy().astype(np.float16))

    return outs


def cal_weight_quant_a16w8(x1, x2, bias, antiquant_scale, antiquant_offset, group_list, group_list_type,
                           x1_dtype, x2_dtype, y_dtype, trans_b):
    import torch
    print("weight quant a16w8")
    assert x1_dtype == y_dtype
    x1 = torch.from_numpy(x1.astype(np.float32))
    is_f8_input = x2_dtype in ["hifloat8", "float8_e5m2", "float8_e4m3fn"]

    if is_f8_input:
        if x1_dtype == "bfloat16":
            x2 = torch.from_numpy(x2.astype(np.float32)).to(torch.bfloat16)
        else:
            x2 = torch.from_numpy(x2.astype(np.float16))
    else:
        x2 = torch.from_numpy(x2.astype(np.int32)).to(torch.int32)

    print(f"x1 = {x1}", flush=True)
    print(f"x2 = {x2}", flush=True)

    if (bias is not None) and (len(bias) != 0):
        bias = torch.from_numpy(bias.astype(np.float32))
    else:
        n_size = x2.shape[-1] if not trans_b else x2.shape[-2]
        bias = torch.from_numpy(np.zeros([x2.shape[0], n_size], dtype=np.float32))

    if (antiquant_offset is not None) and (len(antiquant_offset) != 0):
        pass
    else:
        antiquant_offset = np.zeros(antiquant_scale.shape, dtype=antiquant_scale.dtype)

    if x1_dtype == "bfloat16":
        x2 = x2.to(torch.bfloat16)
        antiquant_scale = torch.from_numpy(antiquant_scale.astype(np.float32)).to(torch.bfloat16)
        antiquant_offset = torch.from_numpy(antiquant_offset.astype(np.float32)).to(torch.bfloat16)
    elif x1_dtype == "float16":
        x2 = x2.to(torch.float16)
        antiquant_scale = torch.from_numpy(antiquant_scale)

    outs = []
    for i in range(len(group_list)):
        start_m, end_m = get_group_m(group_list, group_list_type, i)
        x1_group = x1[start_m:end_m, :]

        weight_group = x2[i]
        if trans_b:
            weight_group = weight_group.T

        antiquant_scale_i = antiquant_scale[i].reshape(1, antiquant_scale[i].shape[0])
        antiquant_offset_i = antiquant_offset[i].reshape(1, antiquant_offset[i].shape[0])

        dequant_weight = (weight_group + antiquant_offset_i) * antiquant_scale_i

        if x1_dtype == "bfloat16":
            x1_group = x1_group.to(torch.bfloat16)
            dequant_weight = dequant_weight.to(torch.bfloat16)

        group_out = torch.matmul(x1_group.to(torch.float32), dequant_weight.to(torch.float32))
        group_out += bias[i].to(torch.float32)

        if y_dtype == "bfloat16":
            outs.append(group_out.numpy().astype(bfloat16_conversion([y_dtype])[0]))
        else:
            outs.append(group_out.numpy().astype(np.float16))

    return outs


def repack_hif4_scale(scale):
    original_shape = scale.shape
    new_shape = list(original_shape[:-1]) + [original_shape[-1] * 64]
    result = np.zeros(new_shape, dtype=np.float32)

    for idx in np.ndindex(original_shape):
        scale_value = np.uint32(scale[idx])
        for k in range(64):
            new_idx = list(idx)
            new_idx[-1] = idx[-1] * 64 + k
            new_idx = tuple(new_idx)
            high_8bits = (scale_value) & 0xFF
            bits_9_to_16 = (scale_value >> 8) & 0xFF
            secd_idx = k // 8
            secd_bit = (bits_9_to_16 >> secd_idx) & 0x1
            high_16 = (scale_value >> 16) & 0xFFFF
            third_idx = k // 4
            third_bit = (high_16 >> third_idx) & 1
            m2 = high_8bits & 0x3
            e6 = high_8bits >> 2
            result[new_idx] = np.float32(np.power(np.float32(2), (e6 - 48) + secd_bit + third_bit) * np.float32(1 + (m2 >> 1) * (0.5) + (m2 & 0x1) * (0.5) ** (2)))
    return result


def get(container, idx):
    if container is None:
        return None
    if isinstance(container, (list, tuple)):
        if idx < len(container):
            return container[idx]
        return None
    return None


def grouped_matmul_golden(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8,
                          input_dtypes, output_dtypes, original_input_arrays=None, **kwargs):
    import torch
    import os

    x1, x2, bias, scale, _, antiquant_scale, antiquant_offset, group_list, pertoken_scale = input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8
    x1_dtype, x2_dtype, bias_dtype, scale_dtype, _, antiquant_scale_dtype, _, _, pertoken_scale_dtype = input_dtypes[0], input_dtypes[1], input_dtypes[2], input_dtypes[3], input_dtypes[4], input_dtypes[5], input_dtypes[6], input_dtypes[7], input_dtypes[8]

    trans_a = kwargs.get("transpose_x", False)
    trans_b = kwargs.get("transpose_weight", False)
    group_type = kwargs.get("group_type", -1)
    group_list_type = kwargs.get("group_list_type", 0)

    out_dtype = output_dtypes[0]
    if out_dtype == 'bfloat16':
        out_dtype = bfloat16_conversion(output_dtypes)[0]

    shape_m = x1.shape[0] if not trans_a else x1.shape[1]
    shape_n = x2.shape[-1] if not trans_b else x2.shape[-2]
    shape_k = x1.shape[-1] if not trans_a else x1.shape[-2]
    split_m = group_type == 0
    split_k = group_type == 2

    is_quant = scale.any()
    weight_quant_s8s4_flag = x1_dtype == "int8" and x2_dtype == "int4"
    if weight_quant_s8s4_flag:
        is_quant = False
    weight_quant_mxfp4_flag = "float4" in x2_dtype and "float8" in antiquant_scale_dtype
    print(f"weight_quant_s8s4_flag: {weight_quant_s8s4_flag}, weight_quant_mxfp4_flag: {weight_quant_mxfp4_flag}")
    is_weight_quant = antiquant_scale.any()
    print(f"is_quant: {is_quant}, is_weight_quant: {is_weight_quant}")

    outs = []
    group_num = len(group_list)
    if is_quant:
        gs_n = 128
        gs_k = 128
        x_quant_type = None
        w_quant_type = None
        is_mx_quant = x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2") and scale_dtype == "float8_e8m0"
        is_hif4 = x1_dtype in ("hifloat4") and scale_dtype == "float32"
        is_compatible = x1_dtype == 'int8' and scale_dtype not in ("uint64", "int64")

        if not is_mx_quant:
            x_quant_type, w_quant_type = infer_quant_mode(scale, pertoken_scale, trans_b, shape_m, shape_k, shape_n, group_num, split_m, split_k, gs_k, gs_n)
        print(f"Qunat mode is :{x_quant_type} {w_quant_type}")

        is_two_scale = x_quant_type == "perTensor" and w_quant_type == "perTensor"
        is_pertile = x_quant_type == "perGroup" and w_quant_type == "perBlock"
        is_pertoken = x_quant_type == "perToken"

        if w_quant_type == "perTensor":
            scale = scale.reshape([group_num, 1])
        if x_quant_type == "perTensor":
            pertoken_scale = pertoken_scale.reshape([group_num, 1])

        if group_list_type == 1:
            group_list = np.cumsum(group_list)
        if is_mx_quant:
            if split_m:
                pertoken_scale_mx = input_8
                m, k0, k1 = pertoken_scale_mx.shape
                pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
                x1_local = input_0
                if x1_dtype == 'float4_e2m1':
                    x1_local = trans_np_fp4_e2m1_tensor_to_bfloat16(x1_local).astype(np.float32)
                elif x1_dtype == 'float4_e1m2':
                    x1_local = trans_np_fp4_e1m2_tensor_to_bfloat16(x1_local).astype(np.float32)
                if trans_a:
                    x1_local = np.swapaxes(x1_local, -1, -2)
                    pertoken_scale_mx = pertoken_scale_mx.transpose()
                pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
                x1_dims = len(x1_local.shape)
                x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1_local.shape[-1]
                x1_local = np.pad(x1_local, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
                x1_local = x1_local * pertoken_scale_mx_broadcast
                x1_local = convert_to_high_precision(x1_local, x1_dtype)
                for i in range(group_num):
                    scale_g = input_3[i]
                    if trans_b == False:
                        scale_g = transform_tensor(scale_g)
                    else:
                        n, k0, k1 = scale_g.shape
                        scale_g = scale_g.reshape(n, k0 * k1)
                    x2_local = input_1[i]
                    if x2_dtype == 'float4_e2m1':
                        x2_local = trans_np_fp4_e2m1_tensor_to_bfloat16(x2_local).astype(np.float32)
                    elif x2_dtype == 'float4_e1m2':
                        x2_local = trans_np_fp4_e1m2_tensor_to_bfloat16(x2_local).astype(np.float32)
                    if trans_b:
                        x2_local = np.swapaxes(x2_local, -1, -2)
                        scale_g = scale_g.transpose()
                    bias_g = input_2[i] if len(bias.shape) == 2 else None
                    deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
                    x2_dims = len(x2_local.shape)
                    x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2_local.shape[-2]
                    x2_local = np.pad(x2_local, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
                    x2_local = x2_local * deq_scale_mx_broadcast
                    if i == 0:
                        x1_temp = x1_local[:group_list[i], :]
                    else:
                        x1_temp = x1_local[group_list[i - 1]:group_list[i], :]
                    x2_local = convert_to_high_precision(x2_local, x2_dtype)
                    out = single_group_mm_cal(x1_temp, x2_local, bias_g, None, None, None, out_dtype, bias_dtype, is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
            elif split_k:
                pertoken_scale_mx = input_8
                x1_local = input_0
                x2_local = input_1
                deq_scale_mx = input_3
                M = x1_local.shape[-1]
                N = x2_local.shape[-1]
                scale_k_offset = 0
                pertoken_scale_mx = transform_tensor(pertoken_scale_mx)
                deq_scale_mx = transform_tensor(deq_scale_mx)
                for i in range(group_num):
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    cur_scale_k = (cur // 64 + i + 1) * 2
                    if cur - pre == 0:
                        out = np.zeros((M, N), dtype=out_dtype)
                        outs.append(out)
                        if i != group_num - 1:
                            if group_list[i + 1] != 0:
                               scale_k_offset = cur_scale_k
                        continue
                    x1_g = x1_local[pre:cur, :]
                    x2_g = x2_local[pre:cur, :]
                    pertoken_scale_mx_g = pertoken_scale_mx[scale_k_offset:cur_scale_k, :]
                    deq_scale_mx_g = deq_scale_mx[scale_k_offset:cur_scale_k, :]
                    scale_k_offset = cur_scale_k
                    if trans_a:
                        x1_g = np.swapaxes(x1_g, -1, -2)
                        pertoken_scale_mx_g = pertoken_scale_mx_g.transpose()
                    pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx_g, 32, axis=-1)
                    x1_dims = len(x1_g.shape)
                    x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1_g.shape[-1]
                    x1_g = np.pad(x1_g, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
                    x1_g = x1_g * pertoken_scale_mx_broadcast
                    x1_g = convert_to_high_precision(x1_g, x1_dtype)
                    if trans_b:
                        x2_g = np.swapaxes(x2_g, -1, -2)
                        deq_scale_mx_g = deq_scale_mx_g.transpose()
                    deq_scale_mx_broadcast = np.repeat(deq_scale_mx_g, 32, axis=-2)
                    x2_dims = len(x2_g.shape)
                    x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2_g.shape[-2]
                    x2_g = np.pad(x2_g, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
                    x2_g = x2_g * deq_scale_mx_broadcast
                    x2_g = convert_to_high_precision(x2_g, x2_dtype)
                    out = single_group_mm_cal(x1_g, x2_g, None, None, None, None, out_dtype, None, is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
        elif is_hif4:
            pertoken_scale_hif4 = input_8
            deq_scale_hif4 = input_3
            x1_local = input_0
            x2_local = input_1
            x1_local = trans_np_fp4_e1m2_tensor_to_bfloat16(x1_local).astype(np.float32)
            x2_local = trans_np_fp4_e1m2_tensor_to_bfloat16(x2_local).astype(np.float32)
            pertoken_scale_hif4 = repack_hif4_scale(pertoken_scale_hif4)
            x1_dims = len(x1_local.shape)
            x2_dims = len(x2_local.shape)
            x1_pad_len = pertoken_scale_hif4.shape[-1] - x1_local.shape[-1]
            x1_local = np.pad(x1_local, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
            x1_local = x1_local * pertoken_scale_hif4
            x1_local = torch.from_numpy(x1_local.astype(np.float32))
            for i in range(group_num):
                scale_g = input_3[i]
                scale_g = repack_hif4_scale(scale_g)
                x2 = input_1[i]
                x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)
                x2 = np.swapaxes(x2, -1, -2)
                scale_g = scale_g.transpose()
                x2_dims = len(x2.shape)
                x2_pad_len = scale_g.shape[-2] - x2.shape[-2]
                x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
                x2 = x2 * scale_g
                if i == 0:
                    x1_temp = x1_local[:group_list[i], :]
                else:
                    x1_temp = x1_local[group_list[i - 1]:group_list[i], :]
                x2 = torch.from_numpy(x2.astype(np.float32))
                out = torch.matmul(x1_temp, x2)
                outs.append(out)
        elif is_pertile:
            if trans_a:
                x1 = np.swapaxes(x1, -1, -2)
                pertoken_scale = np.swapaxes(pertoken_scale, -1, -2)
            if trans_b:
                x2 = np.swapaxes(x2, -1, -2)
                scale = np.swapaxes(scale, -1, -2)
            x1 = convert_to_high_precision(x1, x1_dtype)
            x2 = convert_to_high_precision(x2, x2_dtype)
            pertoken_scale = torch.from_numpy(pertoken_scale).to(torch.float32)
            scale = torch.from_numpy(scale).to(torch.float32)
            if split_m:
                for i in range(group_num):
                    scale_g = scale[i]
                    x2_g = x2[i]
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    x1_g = x1[pre:cur, :]
                    pertoken_scale_g = pertoken_scale[pre:cur, :]
                    out = pertile_mm_cal(x1_g, x2_g, pertoken_scale_g, scale_g, out_dtype)
                    outs.append(out)
            elif split_k:
                for i in range(group_num):
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    k_g = cur - pre
                    if k_g == 0:
                        out = np.zeros((shape_m, shape_n), dtype=out_dtype)
                        outs.append(out)
                        continue
                    x1_g = x1[:, pre:cur]
                    x2_g = x2[pre:cur, :]
                    cur_scale_k = math.ceil(k_g / gs_k)
                    scale_k_offset = 0 if i == 0 else group_list[i - 1] // gs_k + i
                    pertoken_scale_g = pertoken_scale[:, scale_k_offset:(scale_k_offset + cur_scale_k)]
                    scale_g = scale[scale_k_offset:(scale_k_offset + cur_scale_k), :]
                    out = pertile_mm_cal(x1_g, x2_g, pertoken_scale_g, scale_g, out_dtype)
                    outs.append(out)
        else:
            x1 = convert_to_high_precision(x1, x1_dtype)
            x2 = convert_to_high_precision(x2, x2_dtype)
            if is_mx_quant:
                pass
            elif scale_dtype == "uint64" or scale_dtype == "int64":
                deq_scale = np.load(kwargs.get('testcase_name', '') + "_deq_scale.npy")
                deq_scale_tensor = torch.from_numpy(deq_scale)
            else:
                deq_scale = scale.astype(np.float32)
                deq_scale_tensor = torch.from_numpy(deq_scale)

            if is_pertoken and group_type == 0:
                pertoken_scale_slice = torch.unsqueeze(torch.from_numpy(pertoken_scale), dim=1).to(torch.float32)
            elif is_two_scale:
                two_scale = scale_generate(pertoken_scale * deq_scale)
                two_scale_tensor = torch.unsqueeze(torch.from_numpy(two_scale), dim=1).to(torch.float32)
            elif x_quant_type == "perTensor":
                deq_scale_tensor *= torch.from_numpy(pertoken_scale).to(torch.float32)
            pertoken_scale_temp = None
            deq_scale_temp = None
            two_scale_temp = None
            bias_temp = None
            if group_type == 0:
                for i in range(group_num):
                    if i == 0:
                        x1_temp = x1[:group_list[i], :]
                    else:
                        x1_temp = x1[group_list[i - 1]:group_list[i], :]
                    x2_temp = x2[i]
                    if trans_b:
                        x2_temp = np.swapaxes(x2_temp, -1, -2)
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    if bias.any():
                        bias_temp = bias[i]
                    if is_pertoken:
                        pertoken_scale_temp = pertoken_scale_slice[pre:cur]
                        deq_scale_temp = deq_scale_tensor[i]
                    elif is_two_scale:
                        two_scale_temp = two_scale_tensor[i]
                    else:
                        deq_scale_temp = deq_scale_tensor[i]
                    out = single_group_mm_cal(x1_temp, x2_temp, bias_temp, pertoken_scale_temp, deq_scale_temp, two_scale_temp,
                                              out_dtype, bias_dtype,
                                              is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
            elif group_type == 2:
                for i in range(group_num):
                    M = x1.shape[-1]
                    N = x2.shape[-1]
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    if cur - pre == 0:
                        out = np.zeros((M, N), dtype=out_dtype)
                        outs.append(out)
                        continue
                    x1_g = x1[pre:cur, :]
                    if trans_a:
                        x1_g = np.swapaxes(x1_g, -1, -2)
                    x2_g = x2[pre:cur, :]

                    if bias != []:
                        bias_temp = bias[i]
                    if is_pertoken:
                        pertoken_scale_temp = pertoken_scale[i].flatten()
                        pertoken_scale_temp = torch.unsqueeze(torch.from_numpy(pertoken_scale_temp), dim=1).to(torch.float32)
                        deq_scale_temp = deq_scale_tensor[i]
                    elif is_two_scale:
                        two_scale_temp = two_scale_tensor[i]
                    else:
                        deq_scale_temp = deq_scale_tensor[i]
                    out = single_group_mm_cal(x1_g, x2_g, bias_temp, pertoken_scale_temp, deq_scale_temp, two_scale_temp,
                                              out_dtype, bias_dtype,
                                              is_pertoken, is_two_scale, is_mx_quant, is_hif4, is_compatible)
                    outs.append(out)
    elif is_weight_quant:
        if weight_quant_s8s4_flag:
            x2_ori = get(original_input_arrays, 1) if original_input_arrays else input_1
            print("x1 shape: {}, x1_dtype: {}, x2 shape: {}, x2 ori shape: {}, x2_dtype: {}".format(
                  x1.shape, x1_dtype, x2.shape, x2_ori.shape, x2_dtype))
            print("scale shape: {}, dtype: {}".format(scale.shape, scale.dtype))
            print("per-group scale shape: {}, dtype: {}".format(antiquant_scale.shape, scale.dtype))
            print("per-token scale shape: {}, dtype: {}".format(pertoken_scale.shape, pertoken_scale.dtype))

            assert np.array_equal(reshape_last_two_dims(x2_ori), x2)
            assert not trans_a and not trans_b

            outs = cal_weight_quant_s8s4(x1, x2_ori, bias, scale, antiquant_scale, pertoken_scale, group_list,
                                         group_list_type, output_dtypes[0])
        elif weight_quant_mxfp4_flag:
            assert x2_format == "FRACTAL_NZ" if 'x2_format' in dir() else True
            outs = cal_weight_quant_mx(x1, x2, bias, antiquant_scale, pertoken_scale, group_list, group_list_type,
                                       x1_dtype, x2_dtype, output_dtypes[0], trans_b)
        else:
            outs = cal_weight_quant_a16w8(x1, x2, bias, antiquant_scale, antiquant_offset, group_list, group_list_type,
                                          x1_dtype, x2_dtype, output_dtypes[0], trans_b)
    else:
        if group_list_type == 1:
            group_list = np.cumsum(group_list)
        if group_type == 0:
            _, w_outer, w_inner = x2.shape
            if x2_dtype == "float16" or x2_dtype == "bfloat16":
                w_outer_align = (w_outer + 16) // w_outer * w_outer
                w_inner_align = (w_inner + 16) // w_inner * w_inner

            if trans_a:
                x1 = np.swapaxes(x1, -1, -2)
            x1 = torch.from_numpy(x1.astype(np.float32))
            for i in range(group_num):
                x2_local = input_1[i]
                if x2_format == "FRACTAL_NZ" if 'x2_format' in dir() else False:
                    x2_local = x2_local.transpose((1, 2, 0, 3)).reshape(w_outer_align, w_inner_align)
                    x2_local = x2_local[:w_outer, :w_inner]

                if trans_b:
                    x2_local = np.swapaxes(x2_local, -1, -2)
                x2_local = torch.from_numpy(x2_local.astype(np.float32))
                if i == 0:
                    x1_temp = x1[:group_list[i], :]
                else:
                    x1_temp = x1[group_list[i - 1]:group_list[i], :]
                bias_temp = None
                if bias.any():
                    bias_temp = bias[i]

                out = torch.matmul(x1_temp, x2_local)
                if bias_temp is not None:
                    bias_temp = torch.from_numpy(bias_temp.astype(np.float32))
                    out += bias_temp
                outs.append(out.numpy().astype(out_dtype))
        elif group_type == 2:
            x1 = torch.from_numpy(x1.astype(np.float32))
            x2 = torch.from_numpy(x2.astype(np.float32))
            for i in range(group_num):
                M = x1.shape[-1]
                N = x2.shape[-1]
                pre = 0 if i == 0 else group_list[i - 1]
                cur = group_list[i]
                if cur - pre == 0:
                    out = np.zeros((M, N), dtype=out_dtype)
                    outs.append(out)
                    continue
                x1_g = x1[pre:cur, :]
                if trans_a:
                    x1_g = np.swapaxes(x1_g, -1, -2)
                x2_g = x2[pre:cur, :]
                out = torch.matmul(x1_g, x2_g).numpy().astype(out_dtype)
                outs.append(out)
    real_out = outs if not outs else np.concatenate(outs, axis=0)
    return real_out
