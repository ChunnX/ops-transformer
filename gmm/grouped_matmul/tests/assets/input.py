#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GroupedMatmul Input Generator"""
import numpy as np

__input__ = {"kernel": {"grouped_matmul": "grouped_matmul_input"}}


def numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
        return bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow as tf
            return tf.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError("ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")


def numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                            "Please install with `pip3 install en-dtypes`")


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


def trans_np_bfloat16_tensor_to_fp4_e2m1(in_tensor):
    from ml_dtypes import bfloat16
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.uint8)
    in_tensor = in_tensor.reshape(multi_shape)

    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e2m1(in_tensor[i])

    out_tensor = out_tensor.astype(np.uint8)

    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape // 2).astype(np.uint8)
    for i in range(multi_shape // 2):
        fp4_tensor[i] = (out_tensor[i * 2 + 1] << 4) | out_tensor[i * 2]

    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor


def trans_np_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    from ml_dtypes import bfloat16
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.uint8)
    in_tensor = in_tensor.reshape(multi_shape)

    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e1m2(in_tensor[i])

    out_tensor = out_tensor.astype(np.uint8)

    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape // 2).astype(np.uint8)
    for i in range(multi_shape // 2):
        fp4_tensor[i] = (out_tensor[i * 2 + 1] << 4) | out_tensor[i * 2]
    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor


def scale_generate(testcase_name):
    deq_scale_shape = kwargs.get('input_shapes', [[]])[3] if kwargs else [[]]
    if deq_scale_shape and len(deq_scale_shape) > 0 and deq_scale_shape[0]:
        deq_scale_shape = deq_scale_shape[0]
    else:
        deq_scale_shape = (1,)
    has_offset = kwargs.get('input_arrays', [None] * 9)[4] is not None if kwargs else False
    out_dtype = kwargs.get('output_dtypes', ['float32'])[0] if kwargs else 'float32'

    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=deq_scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
    uint32_deq_scale &= 0XFFFFE000

    if has_offset:
        offset_shape = kwargs.get('input_shapes', [[]])[4] if kwargs else [[]]
        if offset_shape and len(offset_shape) > 0 and offset_shape[0]:
            offset_shape = offset_shape[0]
        else:
            offset_shape = (1,)
        fp32_offset = np.random.uniform(low=-5, high=5, size=offset_shape).astype(np.float32)

    if out_dtype != "int8":
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(deq_scale_shape)
        np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
    elif out_dtype == "int8":
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(deq_scale_shape)
        np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
        s9_offset = 0
        if has_offset:
            np.save(testcase_name + "_offset.npy", fp32_offset)
            s9_offset = f32_2_s9(fp32_offset).astype(int).reshape(offset_shape)
            s9_offset &= 0X1FF
            s9_offset = s9_offset[0] if deq_scale_shape[-1] < offset_shape[-1] else s9_offset
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        uint64_deq_scale |= np.uint64(s9_offset << 37)
        uint64_deq_scale |= 1 << 46
    return uint64_deq_scale


def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip


def get_quant_pre(scale, offset):
    import struct
    scale_binary = struct.pack('f', scale)
    scale_int = int.from_bytes(scale_binary, byteorder='little')
    offset_round = round(offset)
    offset_round_clip = min(max(-256, offset_round), 255)
    offset_binary = struct.pack('i', offset_round_clip)
    offset_int = int.from_bytes(offset_binary, byteorder='little') & 0x1FF

    quant_pre_u64 = (1 << 46) + (offset_int << 37) + scale_int

    return quant_pre_u64


def process_quant_inf_nan(input_range_left, input_range_right, low, high):
    if ("-inf" in str(input_range_left)):
        input_range_left = low
    elif ("inf" in str(input_range_left)):
        input_range_left = high
    elif ("nan" in str(input_range_left)):
        input_range_left = 0
    else:
        input_range_left = input_range_left

    if ("-inf" in str(input_range_right)):
        input_range_right = low
    elif ("inf" in str(input_range_right)):
        input_range_right = high
    elif ("nan" in str(input_range_left)):
        input_range_right = 0
    else:
        input_range_right = input_range_right

    return input_range_left, input_range_right


kwargs = {}


def grouped_matmul_input(inputs, attrs, **kw):
    global kwargs
    kwargs = kw

    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8]
    x1, x2, bias, input_deq_scale, _, antiquant_scale, _, group_list_shape, pertoken_scale = input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8

    out_dtype = kw.get('output_dtypes', ['float32'])[0]

    group_size = kw.get('group_size', 0)
    if (x1.dtype == 'hifloat4' or x1.dtype == "float4_e1m2") and input_deq_scale.dtype == "float32":
        mx_gm_hif4scale = 0b10000000 << 0 + 0b00000000
        pertoken_scale = np.random.randint(128, 130, size=pertoken_scale.shape, dtype=np.uint32)
        input_deq_scale = np.random.randint(128, 130, size=input_deq_scale.shape, dtype=np.uint32)

        random_bits = np.random.randint(0, 2**24 - 0, size=pertoken_scale.shape, dtype=np.uint32)
        random_upper = random_bits << 8
        lower_bits_pertoken_scale = pertoken_scale & 0xFF
        pertoken_scale[:] = random_upper | lower_bits_pertoken_scale

        random_bits = np.random.randint(0, 2**24 - 1, size=input_deq_scale.shape, dtype=np.uint32)
        random_upper = random_bits << 8
        lower_bits_input_deq_scale = input_deq_scale & 0xFF
        input_deq_scale[:] = random_upper | lower_bits_input_deq_scale

    bfloat16_dtype = numpy_bfloat16()
    if x1.dtype == 'float4_e2m1':
        if out_dtype == "float32":
            x1_ori = np.random.uniform(0, 7, x1.shape).astype(np.float64)
        else:
            x1_ori = np.random.uniform(-2, 2, x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e2m1(x1_ori.astype(bfloat16_dtype))
    elif x1.dtype == 'float4_e1m2' or x1.dtype == 'hifloat4':
        if out_dtype == "float32":
            x1_ori = np.random.uniform(0, 2, x1.shape).astype(np.float64)
        else:
            x1_ori = np.random.uniform(-2, 2, x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e1m2(x1_ori.astype(bfloat16_dtype))

    if x2.dtype == 'float4_e2m1':
        if out_dtype == "float32":
            x2_ori = np.random.uniform(0, 7, x2.shape).astype(np.float64)
        else:
            x2_ori = np.random.uniform(-7, 7, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e2m1(x2_ori.astype(bfloat16_dtype))
    elif x2.dtype == 'float4_e1m2' or x2.dtype == 'hifloat4':
        if out_dtype == "float32":
            x2_ori = np.random.uniform(0, 2, x2.shape).astype(np.float64)
        else:
            x2_ori = np.random.uniform(-2, 2, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e1m2(x2_ori.astype(bfloat16_dtype))

    if (input_deq_scale.dtype == "uint64" or input_deq_scale.dtype == "int64") and pertoken_scale is None:
        testcase_name = kw.get('testcase_name', '')
        input_deq_scale = scale_generate(testcase_name)
    elif input_deq_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8)
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=input_deq_scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        input_deq_scale = x2_mx.astype("float8_e8m0")
    elif antiquant_scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(124, 130, size=pertoken_scale.shape).astype(np.uint8)
        x1_mx = 2 ** (x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

    if antiquant_scale.dtype == "float8_e8m0":
        x2_mx_gm = np.random.uniform(124, 130, size=antiquant_scale.shape).astype(np.uint8)
        x2_mx = 2 ** (x2_mx_gm.astype(np.float64) - 127)
        antiquant_scale = x2_mx.astype("float8_e8m0")

    group_num = group_list_shape.shape[0]
    group_list = []
    group_type = kw.get('group_type', 0)
    group_list_type = kw.get('group_list_type', 0)
    group_list_expect = kw.get('group_list_expect', [])

    if group_list_expect:
        group_list = group_list_expect
    elif group_num == 1:
        group_list.append(x1.shape[0])
    else:
        avgGroup = (x1.shape[0] + group_num - 1) // group_num
        tmp_num = x1.shape[0]
        print(" Input func x1 shape: ", x1.shape)
        for i in range(group_num - 1):
            tmp = min(tmp_num, avgGroup)
            tmp_num -= tmp
            group_list.append(tmp)
        group_list.append(tmp_num)
        if group_list_type == 0:
            for i in range(1, group_num):
                group_list[i] += group_list[i - 1]
    print("GMM INPUT FUNC, group_list: ", group_list)
    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))

    return [x1, x2, bias, input_deq_scale, None, antiquant_scale, None, np.array(group_list), pertoken_scale]
