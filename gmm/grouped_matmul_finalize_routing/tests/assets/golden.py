#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""GroupedMatmulFinalizeRouting Golden"""
import math
import numpy as np

__golden__ = {"kernel": {"grouped_matmul_finalize_routing": "grouped_matmul_finalize_routing_golden"}}


def numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
        return bfloat16
    except ModuleNotFoundError:
        raise RuntimeError("ml-dtypes is needed to support bfloat16 dtype!!! "
                           "Please install with `pip3 install ml-dtypes`")


def bfloat16_conversion(container):
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
                ret[idx] = numpy_bfloat16()
    return ret


def float_to_hex(f):
    import struct
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def cvt_fp4_e2m1_to_bfloat16(x):
    Fp4e2m1ToBf16 = {0: 0x0, 1: 0x3F00, 2: 0x3F80, 3: 0x3FC0,
                      4: 0x4000, 5: 0x4040, 6: 0x4080, 7: 0x40C0,
                      8: 0x8000, 9: 0xBF00, 10: 0xBF80, 11: 0xBFC0,
                      12: 0xC000, 13: 0xC040, 14: 0xC080, 15: 0xC0C0}
    x = int(x)
    first_fp4val = x & 0x0f
    second_fp4val = (x >> 4) & 0x0f
    return Fp4e2m1ToBf16.get(int(first_fp4val), 0), Fp4e2m1ToBf16.get(int(second_fp4val), 0)


def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {0: 0x0, 1: 0x3E80, 2: 0x3F00, 3: 0x3F40,
                      4: 0x3F80, 5: 0x3FA0, 6: 0x3FC0, 7: 0x3FE0,
                      8: 0x8000, 9: 0xBE80, 10: 0xBF00, 11: 0xBF40,
                      12: 0xBF80, 13: 0xBFA0, 14: 0xBFC0, 15: 0xBFE0}
    x = int(x)
    first_fp4val = x & 0x0f
    second_fp4val = (x >> 4) & 0x0f
    return Fp4e1m2ToBf16.get(int(first_fp4val), 0), Fp4e1m2ToBf16.get(int(second_fp4val), 0)


def trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    import struct
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)

    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i * 2], bfloat16_tensor[i * 2 + 1] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
        fp32_tensor[i * 2] = struct.unpack('!f', struct.pack('!I', bfloat16_tensor[i * 2] << 16))[0]
        fp32_tensor[i * 2 + 1] = struct.unpack('!f', struct.pack('!I', bfloat16_tensor[i * 2 + 1] << 16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


def trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    import struct
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape)
    in_tensor = in_tensor.reshape(multi_shape)

    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i * 2], bfloat16_tensor[i * 2 + 1] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
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


def single_group_mm_cal(x1, x2, out_dtype):
    import torch
    out = torch.matmul(x1, x2)
    torch.set_printoptions(threshold=torch.inf)
    has_inf = torch.isinf(out).any()
    has_nan = torch.isnan(out).any()

    out = out.numpy().astype(out_dtype)
    return out


def transform_tensor(input_tensor):
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def combine_func(x, logits, residual, resid_scale, source_row, output_bs, offset):
    top_k = x.shape[0] // output_bs
    remain_logits = len(logits) % top_k
    if remain_logits:
        logits = logits[:len(logits) - remain_logits]
    out = x * logits.reshape(-1, 1)
    remain_sr = len(source_row) % top_k
    if remain_sr:
        source_row = source_row[:len(source_row) - remain_sr]
    index = np.argsort(source_row)
    out = out[index].reshape(output_bs, top_k, x.shape[-1]).sum(axis=1)
    if residual is not None:
        out[offset:offset + residual.shape[0], :] += resid_scale * residual.astype(np.float32)
    return out


def grouped_matmul_finalize_routing_golden(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9,
                                            input_dtypes, output_dtypes, **kwargs):
    import torch

    x1, x2_all, scale, bias, pertoken_scale, group_list, shared_input, logit, row_index, _ = input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9

    trans_b = kwargs.get("transpose_w", False)
    group_list_type = kwargs.get("group_list_type", 0)
    shared_input_weight = kwargs.get("shared_input_weight")
    shared_input_offset = kwargs.get("shared_input_offset")
    output_bs = kwargs.get("output_bs", x1.shape[0] // len(group_list))

    outs = []
    group_num = len(group_list)

    if group_list_type == 1:
        group_list = np.cumsum(group_list)
    M = x1.shape[0]
    N = x2_all.shape[-1]
    pertoken_scale_mx = pertoken_scale
    m, k0, k1 = pertoken_scale_mx.shape
    pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
    pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
    x1_dims = len(x1.shape)

    if x1.dtype == 'float4_e2m1':
        x1 = trans_np_fp4_e2m1_tensor_to_bfloat16(x1).astype(np.float32)
    elif x1.dtype == 'float4_e1m2':
        x1 = trans_np_fp4_e1m2_tensor_to_bfloat16(x1).astype(np.float32)

    x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]

    x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
    x1 = x1 * pertoken_scale_mx_broadcast
    x1 = convert_to_high_precision(x1, str(x1.dtype))

    for i in range(group_num):
        scale_g = scale[i]
        if trans_b == False:
            scale_g = transform_tensor(scale_g)
        else:
            n, k0, k1 = scale_g.shape
            scale_g = scale_g.reshape(n, k0 * k1)

        x2 = x2_all[i]

        if x2.dtype == 'float4_e2m1':
            x2 = trans_np_fp4_e2m1_tensor_to_bfloat16(x2).astype(np.float32)
        elif x2.dtype == 'float4_e1m2':
            x2 = trans_np_fp4_e1m2_tensor_to_bfloat16(x2).astype(np.float32)

        if trans_b:
            x2 = np.swapaxes(x2, -1, -2)
            scale_g = scale_g.transpose()

        deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
        x2_dims = len(x2.shape)
        x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2.shape[-2]
        x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)
        x2 = x2 * deq_scale_mx_broadcast

        if i == 0:
            x1_temp = x1[:group_list[i], :]
        else:
            x1_temp = x1[group_list[i - 1]:group_list[i], :]
        x2 = convert_to_high_precision(x2, str(x2.dtype))
        out = single_group_mm_cal(x1_temp, x2, output_dtypes[0])
        if bias is not None:
            out += bias[i, :].reshape(1, -1)
        outs.append(out)

    gmm_out = outs if not outs else np.concatenate(outs, axis=0)
    gmm_new = torch.from_numpy(gmm_out).to(torch.float32)
    final_out = combine_func(gmm_out, logit, shared_input, shared_input_weight, row_index, output_bs, shared_input_offset)
    final_out_new = torch.from_numpy(final_out).to(torch.float32)
    return final_out
