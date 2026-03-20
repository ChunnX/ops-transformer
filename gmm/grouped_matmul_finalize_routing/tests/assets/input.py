#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GroupedMatmulFinalizeRouting Input Generator"""
import numpy as np

__input__ = {"kernel": {"grouped_matmul_finalize_routing": "grouped_matmul_finalize_routing_input"}}


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


def grouped_matmul_finalize_routing_input(inputs, attrs, **kwargs):
    import torch
    bfloat16_dtype = numpy_bfloat16()

    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]
    x1, x2, scale, bias, pertoken_scale, group_list_shape, shared_input, logit, row_index, _ = input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9

    row_index_test = []
    for i in range(len(row_index) // group_list_shape.shape[0]):
        row_index_test.extend([i] * group_list_shape.shape[0])
    remain = len(row_index) % group_list_shape.shape[0]
    if remain:
        row_index_test.extend([0] * remain)
    row_index_test = np.array(row_index_test, dtype=np.int64)
    row_index = row_index_test

    if x1.dtype == 'float4_e2m1':
        x1_ori = np.random.uniform(0, 1, x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e2m1(x1_ori.astype(bfloat16_dtype))
    elif x1.dtype == 'float4_e1m2':
        x1_ori = np.random.uniform(0, 1, x1.shape).astype(np.float64)
        x1 = trans_np_bfloat16_tensor_to_fp4_e1m2(x1_ori.astype(bfloat16_dtype))

    if x2.dtype == 'float4_e2m1':
        x2_ori = np.random.uniform(0, 1, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e2m1(x2_ori.astype(bfloat16_dtype))
    elif x2.dtype == 'float4_e1m2':
        x2_ori = np.random.uniform(0, 1, x2.shape).astype(np.float64)
        x2 = trans_np_bfloat16_tensor_to_fp4_e1m2(x2_ori.astype(bfloat16_dtype))

    if scale.dtype == "float8_e8m0" and pertoken_scale.dtype == "float8_e8m0":
        x1_mx_gm = np.random.uniform(127, 130, size=pertoken_scale.shape).astype(np.uint8)
        x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
        pertoken_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=scale.shape).astype(np.uint8)
        x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
        scale = x2_mx.astype("float8_e8m0")

    group_num = group_list_shape.shape[0]
    group_list = []
    group_list_type = kwargs.get("group_list_type", 0)
    group_list_expect = kwargs.get("group_list_expect", [])
    if group_list_expect:
        group_list = group_list_expect
    group_list_tmp = group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(group_list)
    if group_list_tmp and group_list_tmp[-1] > x1.shape[0]:
        raise Exception('sum of grouplist: ({}) can not be greater than x1[0]: ({})'.format(group_list_tmp[-1], x1.shape[0]))

    bias_n = bias.shape[-1]
    bias = np.zeros((x2.shape[0], bias_n)).astype(bfloat16_dtype)

    return [x1, x2, scale, bias, pertoken_scale, np.array(group_list), shared_input, logit, row_index, None]
