#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__input__ = {
    "kernel": {
        "grouped_matmul": "grouped_matmul_inputs"
    },
}

import numpy as np


def _unwrap_dtype(dt):
    if isinstance(dt, (list, tuple)):
        return dt[0] if len(dt) == 1 else dt
    return dt


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


def scale_generate(deq_scale_shape, has_offset, out_dtype, testcase_name):
    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=deq_scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
    uint32_deq_scale &= 0XFFFFE000

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
            fp32_offset = np.random.uniform(low=-5, high=5, size=deq_scale_shape).astype(np.float32)
            np.save(testcase_name + "_offset.npy", fp32_offset)
            s9_offset = f32_2_s9(fp32_offset).astype(int).reshape(deq_scale_shape)
            s9_offset &= 0X1FF
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        uint64_deq_scale |= np.uint64(s9_offset << 37)
        uint64_deq_scale |= 1 << 46
    return uint64_deq_scale


def grouped_matmul_inputs(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                          group_list, per_token_scale,
                          transpose_x=False, transpose_weight=False,
                          group_type=0, group_list_type=0, **kwargs):
    testcase_name = kwargs['testcase_name']
    output_dtypes = kwargs['output_dtypes']
    input_dtypes = kwargs['input_dtypes']
    out_dtype = _unwrap_dtype(output_dtypes[0])
    input_deq_scale = scale
    x1_dtype = _unwrap_dtype(input_dtypes[0])
    scale_dtype = _unwrap_dtype(input_dtypes[3])
    pertoken_scale_dtype = _unwrap_dtype(input_dtypes[8])
    antiquant_scale_dtype = _unwrap_dtype(input_dtypes[5])
    group_list_shape = group_list.shape if not isinstance(group_list, (list, tuple)) else (len(group_list),)

    if x1_dtype == "float4_e1m2" and scale_dtype == "float32":
        per_token_scale = np.random.randint(128, 130, size=per_token_scale.shape, dtype=np.uint32)
        input_deq_scale = np.random.randint(128, 130, size=input_deq_scale.shape, dtype=np.uint32)

        random_bits = np.random.randint(0, 2**24 - 0, size=per_token_scale.shape, dtype=np.uint32)
        random_upper = random_bits << 8
        lower_bits_pertoken_scale = per_token_scale & 0xFF
        per_token_scale[:] = random_upper | lower_bits_pertoken_scale

        random_bits = np.random.randint(0, 2**24 - 1, size=input_deq_scale.shape, dtype=np.uint32)
        random_upper = random_bits << 8
        lower_bits_input_deq_scale = input_deq_scale & 0xFF
        input_deq_scale[:] = random_upper | lower_bits_input_deq_scale

    if (scale_dtype == "uint64" or scale_dtype == "int64") and per_token_scale is None:
        deq_scale_shape = (input_deq_scale.shape
                           if not isinstance(input_deq_scale, (list, tuple))
                           else input_deq_scale[0].shape)
        has_offset = offset is not None
        input_deq_scale = scale_generate(deq_scale_shape, has_offset, out_dtype, testcase_name)
    elif scale_dtype == "float8_e8m0" and pertoken_scale_dtype == "float8_e8m0":
        pt_shape = per_token_scale.shape if not isinstance(per_token_scale, (list, tuple)) else per_token_scale[0].shape
        sc_shape = input_deq_scale.shape if not isinstance(input_deq_scale, (list, tuple)) else input_deq_scale[0].shape
        x1_mx_gm = np.random.uniform(127, 130, size=pt_shape).astype(np.uint8)
        x1_mx = 2 ** (x1_mx_gm.astype(np.float64) - 127)
        per_token_scale = x1_mx.astype("float8_e8m0")

        x2_mx_gm = np.random.randint(127, 130, size=sc_shape).astype(np.uint8)
        x2_mx = 2 ** (x2_mx_gm.astype(np.float64) - 127)
        input_deq_scale = x2_mx.astype("float8_e8m0")
    elif antiquant_scale_dtype == "float8_e8m0" and pertoken_scale_dtype == "float8_e8m0":
        pt_shape = per_token_scale.shape if not isinstance(per_token_scale, (list, tuple)) else per_token_scale[0].shape
        x1_mx_gm = np.random.uniform(124, 130, size=pt_shape).astype(np.uint8)
        x1_mx = 2 ** (x1_mx_gm.astype(np.float64) - 127)
        per_token_scale = x1_mx.astype("float8_e8m0")

    if antiquant_scale_dtype == "float8_e8m0":
        as_shape = antiquant_scale.shape if not isinstance(antiquant_scale, (list, tuple)) else antiquant_scale[0].shape
        x2_mx_gm = np.random.uniform(124, 130, size=as_shape).astype(np.uint8)
        x2_mx = 2 ** (x2_mx_gm.astype(np.float64) - 127)
        antiquant_scale = x2_mx.astype("float8_e8m0")

    group_num = group_list_shape[0]
    new_group_list = []
    group_list_expect = kwargs.get('group_list_expect', [])

    x1_shape = x.shape if not isinstance(x, list) else x[0].shape
    x1_dim0 = x1_shape[0] if not isinstance(x, list) else sum(s[0] for s in [x[i].shape for i in range(len(x))])

    if group_list_expect:
        new_group_list = group_list_expect
    elif group_num == 1:
        new_group_list.append(x1_dim0)
    else:
        avgGroup = (x1_dim0 + group_num - 1) // group_num
        tmp_num = x1_dim0
        for i in range(group_num - 1):
            tmp = min(tmp_num, avgGroup)
            tmp_num -= tmp
            new_group_list.append(tmp)
        new_group_list.append(tmp_num)
        if group_list_type == 0:
            for i in range(1, group_num):
                new_group_list[i] += new_group_list[i - 1]
    group_list_tmp = new_group_list
    if group_list_type == 1:
        group_list_tmp = np.cumsum(new_group_list)
    if group_list_tmp[-1] > x1_dim0:
        raise Exception(
            'sum of grouplist: ({}) can not be greater than x[0]: ({})'.format(
                group_list_tmp[-1], x1_dim0))

    group_list_dtype = group_list.dtype
    return (x, weight, bias, input_deq_scale, offset, antiquant_scale, antiquant_offset,
            np.array(new_group_list, dtype=group_list_dtype), per_token_scale)
