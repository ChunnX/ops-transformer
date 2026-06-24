
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_tools
   Description :
   Author :       d30004511
   date：          2022/11/18 11:12
-------------------------------------------------
   Change Activity:
                    2022/11/18 11:12
-------------------------------------------------
"""
import math
import random
import struct
from typing import Type, Sequence, Any
from ml_dtypes import uint4
from numpy import float32, floating
import numpy as np
import libs.tools as tools
from ml_dtypes import int4

__version__ = '0.0.1'
__all__ = [
    '__version__',
    'FP4E1M2',
    "FP4E2M1", "trans_np_float_tensor_to_float_n"
]


class FP4E1M2(uint4):
    """dtype object"""

    pass


class FP4E2M1(uint4):
    pass


class FP8E8M0(np.uint8):
    pass


def fp32_to_fp8_bin(value, E=4, M=3):
    """
    :param value: float32类型的值
    :param E: fp8的指数位数
    :param M: fp8的尾数位数
    :return: fp8 binary format,such as: 1.34-->00111100
    """

    value = float(value)
    bias = 2 ** (E - 1) - 1
    # 将float32转换为uint32
    float_as_uint32 = struct.unpack('I', struct.pack('f', value))[0]
    # 提取float32的符号位
    if E == 8:
        sign = 0
    else:
        sign = ((float_as_uint32 >> 31) & 0x1)
    # 18.5: 0 10000011 00101000000000000000000
    # 3.14  0 10000000 10010001111010111000011
    # 提取float32的指数部分，并减去127（float32的偏移量），然后将指数转换为FP8所需的格式
    exponent = ((float_as_uint32 >> 23) & 0xFF) - 127
    # # FP8有
    if exponent > (2 ** E - 1) - bias:
        exponent = (2 ** E - 1) - bias
    elif exponent < -bias:
        exponent = -bias
    exponent += bias
    if E == 8:
        return bin((float_as_uint32 >> 23) & 0xFF)
    tmp_fraction = format(float_as_uint32, 'b').zfill(32)[9:]

    # 对小数部分进行裁剪和对齐，并最终合成FP8的表示
    sign = bin(sign)[2:]
    exponent = ('0' * 8 + bin(exponent)[2:])[-8:][-E:]
    # # 小数部分
    fraction = tmp_fraction[:M - 1]
    fraction_end = tmp_fraction[M:]
    if str(fraction_end).find("1") >= 0:
        end = "1"
    else:
        end = "0"
    fp8 = sign + exponent + fraction + end
    return fp8


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, m_len=23):
    """
    :param sign:  0 1
    :param exponent_raw: 指数
    :param mantissa: 尾数
    :param exp_len: 指数长度
    :param m_len: 尾数长度
    :return:
    """

    sign_value = -1 if sign == 1 else 1

    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(m_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (m_len - b))

    return sign_value * (2 ** exponent) * mant_mult


# sign 的取值范围
# 0 or 1
def e2m1(sign, exponent_raw, mantissa):
    """
    :param sign:  0 整数   1  是负数
    :param exponent_raw:
    :param mantissa:
    :return:
    """
    return ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=2, m_len=1)


def e1m2(sign, exponent_raw, mantissa):
    return ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=1, m_len=2)


def t_float16(val: float):
    """ Print Float16 in a binary form """
    m = struct.unpack('H', struct.pack('e', np.float16(val)))[0]
    return format(m, 'b').zfill(16)


def str_to_binary(string):
    binary = ' '.join(format(ord(char), '08b') for char in string)
    return binary


def random_fp4_2_uint8(shape, style=1):
    """
     FP4：
    （1）FP4可以表示的数有限，可以全部遍历出来。然后基于这些固定数进行随机生成FP16的数据
    （2）使用gen_fp4_from_fp16将fp16的数据转换成fp4，保存下来作为kernel侧的输入
    :param style: 1  e2m1   2 :e1m2
    :return:
    """
    e1m2_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
    e2m1_list = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6]
    random_values = np.random.choice(e1m2_list, size=shape) if style == 1 else np.random.choice(e2m1_list, size=shape)
    tensor = np.array(random_values)
    tensor = tensor.astype(np.uint8)
    print("gen a fp4 tensor!")
    return tensor


def trans_np_float_tensor_to_float_n(in_tensor, style=1):
    """
    Args:
        in_tensor:

        style: 1 fp4e1m2 2:fp4e2m1 3 fp8e8m0

    Returns:
    """
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        if style == 1:
            out_tensor[i] = convert_uint4_to_float4(in_tensor[i], style=1)
        elif style == 2:
            out_tensor[i] = convert_uint4_to_float4(in_tensor[i], style=2)
        elif style == 3:
            out_tensor[i] = in_tensor[i]
        else:
            raise Exception("not support style ")
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor


def trans_fp4_tensor_to_uint4(in_tensor, style=1):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = convert_float4_to_int4(in_tensor[i], style=style)
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor


def convert_uint4_to_float4(value, style=1):
    """
    :param value:
    :param style:  1 e1m2 2 e2m1
    :return:
    """
    if value > 15:
        raise Exception("uint8 max value is 15")
    else:
        if style == 1:
            float4_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
        elif style == 2:
            float4_list = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        else:
            raise Exception("style must be 1 or  2")
        # import pdb;pdb.set_trace()
        return float4_list[int(value)]


def convert_float4_to_int4(value, style=1):
    int8_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if style == 1:
        float4_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
    elif style == 2:
        float4_list = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    else:
        raise Exception("style  must be 1 or 2")
    try:
        index_data = float4_list.index(value)
        return int8_list[index_data]
    except Exception as e:
        raise Exception("convert error")
        # treturn random.choice(int8_list)


def convert_uint4_to_float4_e1m2(value):
    return convert_uint4_to_float4(value)


def convert_uint4_to_float4_e2m1(value):
    return convert_uint4_to_float4(value, style=2)


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def cvt_bfloat16_to_fp4_e2m1(x):
    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    x = eval(float_to_hex(x_abs))
    x = x >> 16
    # print("x=========",x)

    # sRet = x >> 15 & 0b1
    ef = (x >> 7) & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 1  #
    maxExp = 3  # max E encoding value of e2m1 is 3
    expBias = 1  # Exponent Bias value of e2m1/e1m2 is 1
    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0):
        eNorm = ef - 127 + 1  # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127  # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 1)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta)  # determine if need to carry
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
    else:
        eRet = (eNorm + expBias)
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1

        if (((mRet & 0b10) != 0) and (needRound)):
            eRet += 1
            mRet = 0

    if (eRet >= 3):
        eRet = 3
        # if (mRet >= 1) :
        #     mRet = 0
    elif (eRet == 0 and mRet == 0b10):
        eRet += 1
        mRet = 0
    # print("eRet=========",eRet)
    # print("mRet=========",mRet)

    return (((sRet) << 3) | ((eRet) << 1) | ((mRet) & 1))


def IsRoundOne(sign, man, truncLen):
    roundingTruncLen = 64
    # mask0 = (truncLen >= roundingTruncLen) ? 0 : 0x1ul << truncLen;
    if truncLen >= roundingTruncLen:
        mask0 = 0
    else:
        mask0 = 0x1 << truncLen
    # mask1 = (truncLen > roundingTruncLen) ? 0 : 0x1ul << (truncLen - 1);
    if (truncLen > roundingTruncLen):
        mask1 = 0
    else:
        mask1 = 0x1 << (truncLen - 1)

    mask2 = mask1 - 1

    # ROUND_TO_NEAREST
    lastBit = (man & mask0) > 0  # Last bit after conversion
    truncHighBit = (man & mask1) > 0  # Highest bit in the truncated part
    truncLeft = (man & mask2) > 0  # Truncated left part (except for the highest bit)
    return truncHighBit and (truncLeft or lastBit)


def cvt_bfloat16_to_fp4_e1m2(x):
    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    x = eval(float_to_hex(x_abs))
    x = x >> 16
    # print("x=========",x)

    ef = x >> 7 & 0xff
    mf = x & 0x7f
    mLenDelta = 7 - 2  #
    maxExp = 1  # max E encoding value of e1m2 is 3
    expBias = 1  # Exponent Bias value of e2m1/e1m2 is 1

    eRet = 0
    mRet = 0
    eNorm = 0
    if (ef == 0 and mf != 0):
        eNorm = ef - 127 + 1  # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127  # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)):
        return ((sRet << 3) | 0b111)
    elif eNorm <= -(expBias):
        eRet = 0
        mf = (mf | 0x80)
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta)  # determine if need to carry
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
    else:
        eRet = (eNorm + expBias)
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = (mf >> mLenDelta)
        if (needRound):
            mRet += 1
        if (((mRet & 0b100) != 0) and (needRound)):
            eRet += 1
            mRet = 0

    if (eRet >= 1):
        eRet = 1
    elif (eRet == 0 and mRet == 0b100):
        eRet += 1
        mRet = 0

    # print("eRet=========",eRet)
    # print("mRet=========",mRet)
    return (((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3))


def cvt_fp4_e2m1_to_bfloat16(x):
    Fp4e2m1ToBf16 = {'0': 0.0, '1': 0.5, '2': 1.0, '3': 1.5,
                     '4': 2.0, '5': 3.0, '6': 4.0, '7': 6.0,
                     '8': -0.0, '9': -0.5, '10': -1.0, '11': -1.5,
                     '12': -2.0, '13': -3.0, '14': -4.0, '15': -6.0
                     }

    # x_abs = math.fabs(x)
    # # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    # x = eval(float_to_hex(x_abs))
    # x = x >> 16
    x = int(x)
    # first_fp4val = (x >> 4 )& 0x0f
    # second_fp4val = x & 0x0f
    first_fp4val = x & 0x0f

    # print("======first_fp4val====",first_fp4val)
    # print("======second_fp4val====",second_fp4val)
    first_fp4str = str(first_fp4val)

    # print("======first_fp4str====",first_fp4str)

    return Fp4e2m1ToBf16[first_fp4str]


def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {'0': 0.0, '1': 0.25, '2': 0.5, '3': 0.75,
                     '4': 1.0, '5': 1.25, '6': 1.5, '7': 1.75,
                     '8': -0.0, '9': -0.25, '10': -0.5, '11': -0.75,
                     '12': -1.0, '13': -1.25, '14': -1.5, '15': -1.75
                     }

    # x_abs = math.fabs(x)
    # # print("===format(x_abs,'x')=====",float_to_hex(x_abs))
    # x = eval(float_to_hex(x_abs))
    # x = x >> 16
    x = int(x)
    # first_fp4val = (x >> 4 )& 0x0f
    # second_fp4val = x & 0x0f
    first_fp4val = x & 0x0f
    first_fp4str = str(first_fp4val)
    # print("======first_fp4str====",first_fp4str)
    return Fp4e1m2ToBf16[first_fp4str]


def new_trans_np_bfloat16_tensor_to_fp4_e2m1(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    new_in_tensor = in_tensor.reshape(multi_shape)
    out_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e2m1(new_in_tensor[i])
    out_tensor = out_tensor.astype(np.uint8)
    return out_tensor.reshape(shape_tensor)

def new_trans_np_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    new_in_tensor = in_tensor.reshape(multi_shape)
    out_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e1m2(new_in_tensor[i])
    out_tensor = out_tensor.astype(np.uint8)
    return out_tensor.reshape(shape_tensor)

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


# -----------------------最新方案
def unpacked_float4e2m1_to_float32(x: np.ndarray) -> np.ndarray:
    """Evaluate the numerical value of an array of unpacked float4e2m1 values (as uint8)
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        x: an array of uint8 elements representing a float4e2m1 (using the 4 LSB)

    Returns:
        An array of float32 elements representing the values of the float4e2m1 input.
    """
    # x is stored in 4 LSB of int
    sign = np.where(np.bitwise_and(x, 0x08), -1, 1)
    mantissa = x & 0x01
    exponent = (x & 0x06) >> 1

    val = np.where(
        exponent == 0,
        sign * (mantissa / 2.0),
        sign * (1.0 + mantissa / 2.0) * 2.0 ** (exponent - 1),
    )  # denormalized, normalized
    return val


#
# def unpack_float4e2m1(
#         data: np.int32 | np.ndarray,
#         dims
# ) -> np.ndarray:
#     """Converts ndarray of float4e2m1 (as packed uint8) to f32
#     See :ref:`onnx-detail-float4` for technical details.
#
#     Args:
#         data: A numpy array, empty dimensions are allowed if dims is
#             None.
#         dims: The dimensions are used to reshape the unpacked buffer
#
#     Returns:
#         A numpy array of float32 reshaped to dims.
#     """
#     res_high, res_low = subbyte.unpack_single_4bitx2(data.ravel(), False)
#     res = np.empty((res_high.size + res_low.size,), dtype=np.float32)
#
#     res[0::2] = unpacked_float4e2m1_to_float32(res_high)
#     res[1::2] = unpacked_float4e2m1_to_float32(res_low)
#
#     if (
#             res.size == np.prod(dims) + 1
#     ):  # handle single-element padding due to odd number of elements
#         res = res.ravel()[:-1]
#     res = res.reshape(dims)
#     return res
#
#
# def unpack_float4e2m1_to_uint8(
#         data: np.int32 | np.ndarray,
#         dims,
# ) -> np.ndarray:
#     """Converts ndarray of float4e2m1 (as packed uint8) to uint8
#     See :ref:`onnx-detail-float4` for technical details.
#
#     Args:
#         data: A numpy array, empty dimensions are allowed if dims is
#             None.
#         dims: The dimensions are used to reshape the unpacked buffer
#
#     Returns:
#         A numpy array of uint8 reshaped to dims.
#     """
#     res_high, res_low = subbyte.unpack_single_4bitx2(data.ravel(), False)
#     res = np.empty((res_high.size + res_low.size,), dtype=np.uint8)
#
#     res[0::2] = res_high
#     res[1::2] = res_low
#
#     if (
#             res.size == np.prod(dims) + 1
#     ):  # handle single-element padding due to odd number of elements
#         res = res.ravel()[:-1]
#     res = res.reshape(dims)
#     return res
#
#
# def unpack_single_4bitx2(
#         x, signed: bool
# ):
#     unpack_signed = lambda x: np.where((x >> 3) == 0, x, x | 0xF0)  # noqa: E731
#     """Unpack a single byte 4bitx2 to two 4 bit elements
#     Args:
#         x: Input data
#         signed: boolean, whether to interpret as signed int4.
#     Returns:
#         A tuple of ndarrays containing int4 elements (sign-extended to int8/uint8)
#     """
#     if not isinstance(x, np.ndarray):
#         x = np.asarray(x)
#     x_low = x & 0x0F
#     x_high = x >> 4
#     x_low = unpack_signed(x_low) if signed else x_low
#     x_high = unpack_signed(x_high) if signed else x_high
#     dtype = np.int8 if signed else np.uint8
#     return (x_low.astype(dtype), x_high.astype(dtype))

def convert_common_int4_to_fp4(value: int4, style=1):
    """

    Args:
        value:
        style:  1 fp4e1m2  2: fp4e2m1
    Returns:
    """
    if style == 1:
        float4_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75]
    elif style == 2:
        float4_list = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    else:
        raise Exception("style must in [1,2] list")
    value_int4_list = range(-8, 8)
    if value not in value_int4_list:
        raise Exception("not in the range of int4")
    else:
        return float4_list[int(value)]


fp4_e1m2_values = [
    0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75,
    -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75
]

fp4_e2m1_values = [
    0, 0.5, 1 ,1.5, 2 ,3 ,4, 6, 
    0, -0.5, -1, -1.5, -2, -3 ,-4 ,-6
]

# 数值 <-> 编码映射
value_to_code_e1m2 = {val: idx for idx, val in enumerate(fp4_e1m2_values)}
value_to_code_e2m1 = {val: idx for idx, val in enumerate(fp4_e2m1_values)}
code_to_value_e1m2 = {idx: val for idx, val in enumerate(fp4_e1m2_values)}
code_to_value_e2m1 = {idx: val for idx, val in enumerate(fp4_e2m1_values)}


import torch
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
    return fp4_tensor_encoded


def decode_fp4_tensor(nptensor, dtype):
    if dtype in ["float4_e1m2", "fp4_e1m2"]:
        decode_func = np.vectorize(lambda code: code_to_value_e1m2[code])
    elif dtype in ["float4_e2m1", "fp4_e2m1"]:
        decode_func = np.vectorize(lambda code: code_to_value_e2m1[code])
    else:
        print("error fp4 dtype")
    decoded_np = decode_func(nptensor).astype(np.float32)
    return decoded_np


if __name__ == '__main__':
    import numpy as np

    t = np.random.uniform(low=-100, high=100, size=(1, 100)).astype(int4)
    print(t)
