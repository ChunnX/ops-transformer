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
__golden__ = {
    "kernel": {
        "grouped_matmul": "grouped_matmul_golden"
    },
}

import math
import os

import ml_dtypes
import numpy as np
import torch


def convert_to_high_precision(input_tensor, input_type):
    if input_type in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
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


def infer_quant_mode(scale, pertoken_scale, trans_b, shape_m, shape_k, shape_n,
                     group_num, split_m, split_k, gs_k, gs_n):
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
                        is_pertoken, is_two_scale, is_mx_quant, is_compatible):
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
    elif out_dtype == 'int32' or out_dtype == 'int8':
        if out_dtype == 'int32':
            if bias is not None and bias_dtype == "int32":
                pass
            out = out.numpy().astype(np.int32)
        else:
            if is_pertoken:
                out = out * deq_scale_tensor * pertoken_scale_slice
            elif is_two_scale:
                out = out * two_scale_tensor
            else:
                out = out * deq_scale_tensor
            out = torch.clamp(torch.round(out), -128, 127).numpy().astype(np.int8)
    else:
        if is_mx_quant:
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
        out_dtype_np = ml_dtypes.bfloat16
    else:
        out_dtype_np = out_dtype
    out = (out).numpy().astype(out_dtype_np)
    return out


def transform_tensor(input_tensor):
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def trans_nz2nd(input_data):
    nd_data = input_data.transpose(0, 2, 3, 1, 4).reshape(input_data.shape[0],
                                                          input_data.shape[2] * input_data.shape[3],
                                                          input_data.shape[1] * input_data.shape[4])
    return nd_data


def get_per_group_size(antiquant_scale, k_size, trans_b):
    gn = antiquant_scale.shape[-2] if not trans_b else antiquant_scale.shape[-1]
    assert gn > 0 and k_size % gn == 0
    group_size = k_size // gn
    return group_size


def get_mx_scale(antiquant_scale, group_size, pertoken_scale, trans_b, x1_dtype):
    antiquant_scale = antiquant_scale.astype(np.float32)
    antiquant_scale_broadcast = np.repeat(antiquant_scale, group_size, axis=-2) if not trans_b else np.repeat(
        antiquant_scale, group_size, axis=-1)
    antiquant_scale_broadcast = torch.from_numpy(antiquant_scale_broadcast)
    if x1_dtype == "float16":
        antiquant_scale_broadcast = antiquant_scale_broadcast.to(torch.float16)

    if x1_dtype == "float8_e4m3fn":
        pertoken_scale_broadcast = torch.from_numpy(
            np.repeat(pertoken_scale.astype(np.float32), group_size, axis=-1))
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
    return start_m, end_m


def cal_weight_quant_s8s4(x1, x2, bias, scale, antiquant_scale, pertoken_scale, group_list, group_list_type,
                          y_dtype):
    shape_k = x2.shape[-2]
    group_size = get_per_group_size(antiquant_scale, shape_k, False)

    x2_tmp = x2.astype(np.float16)
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

        # Handle per-tensor scale (shape=(1,)) vs per-group scale (shape=(num_groups,))
        if scale.shape[0] == 1:
            scale_val = scale[0].reshape(1, -1)
        else:
            scale_val = scale[i].reshape(1, -1)

        # Handle per-tensor pertoken_scale (shape=(1,)) vs per-token (shape=(M,))
        if pertoken_scale.shape[0] == 1:
            pertoken_val = pertoken_scale[0]
        else:
            pertoken_val = pertoken_scale.reshape(-1, 1)[start_m:end_m, :]

        group_out = group_out.numpy().astype(np.float32) * scale_val * pertoken_val

        if (bias is not None) and (len(bias) != 0):
            if bias.shape[0] == 1:
                group_out += bias[0, :].reshape(1, -1)
            else:
                group_out += bias[i, :].reshape(1, -1)

        if y_dtype == "bfloat16":
            outs.append(group_out.astype(ml_dtypes.bfloat16))
        elif y_dtype == "int32":
            outs.append(group_out.astype(np.int32))
        elif y_dtype == "int8":
            outs.append(group_out.astype(np.int8))
        else:
            outs.append(group_out.astype(np.float16))

    return outs


def cal_weight_quant_a16w8(x1, x2, bias, antiquant_scale, antiquant_offset, group_list, group_list_type,
                           x1_dtype, x2_dtype, y_dtype, trans_b):
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
            outs.append(group_out.numpy().astype(ml_dtypes.bfloat16))
        else:
            outs.append(group_out.numpy().astype(np.float16))

    return outs



def _unwrap_dtype(dt):
    if isinstance(dt, (list, tuple)):
        return dt[0] if len(dt) == 1 else dt
    return dt


def grouped_matmul_golden(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                          group_list, per_token_scale,
                          transpose_x=False, transpose_weight=False,
                          group_type=0, group_list_type=0, **kwargs):
    testcase_name = kwargs['testcase_name']
    input_dtypes = kwargs['input_dtypes']
    input_formats = kwargs['input_formats']
    output_dtypes = kwargs['output_dtypes']

    out_dtype = _unwrap_dtype(output_dtypes[0])
    if out_dtype == 'bfloat16':
        out_dtype = ml_dtypes.bfloat16

    x1_dtype = _unwrap_dtype(input_dtypes[0])
    x2_dtype = _unwrap_dtype(input_dtypes[1])
    bias_dtype = _unwrap_dtype(input_dtypes[2])
    scale_dtype = _unwrap_dtype(input_dtypes[3])
    antiquant_scale_dtype = _unwrap_dtype(input_dtypes[5])
    pertoken_scale_dtype = _unwrap_dtype(input_dtypes[8])

    x2_format = _unwrap_dtype(input_formats[1])

    trans_a = transpose_x
    trans_b = transpose_weight

    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) == 1 else np.concatenate(x, axis=0)
    if isinstance(weight, (list, tuple)):
        if len(weight) == 1:
            weight = weight[0]
        else:
            weight = np.stack(weight, axis=0)
    if isinstance(bias, (list, tuple)):
        bias = bias[0] if len(bias) == 1 else np.stack(bias, axis=0)
    if isinstance(scale, (list, tuple)):
        scale = scale[0] if len(scale) == 1 else np.stack(scale, axis=0)
    if isinstance(antiquant_scale, (list, tuple)):
        antiquant_scale = antiquant_scale[0] if len(antiquant_scale) == 1 else np.stack(antiquant_scale, axis=0)
    if isinstance(antiquant_offset, (list, tuple)):
        antiquant_offset = antiquant_offset[0] if len(antiquant_offset) == 1 else np.stack(antiquant_offset, axis=0)
    if isinstance(group_list, (list, tuple)):
        group_list = np.array(group_list)
    if isinstance(per_token_scale, (list, tuple)):
        per_token_scale = per_token_scale[0] if len(per_token_scale) == 1 else np.stack(per_token_scale, axis=0)

    shape_m = x.shape[0] if not trans_a else x.shape[1]
    shape_n = weight.shape[-1] if not trans_b else weight.shape[-2]
    shape_k = x.shape[-1] if not trans_a else x.shape[-2]
    split_m = group_type == 0
    split_k = group_type == 2

    is_quant = scale.size > 0 and scale.any()
    weight_quant_s8s4_flag = x1_dtype == "int8" and x2_dtype == "int4"
    if weight_quant_s8s4_flag:
        is_quant = False
    weight_quant_mxfp4_flag = "float4" in x2_dtype and "float8" in antiquant_scale_dtype
    is_weight_quant = antiquant_scale.size > 0 and antiquant_scale.any()

    outs = []
    group_num = len(group_list)

    if is_quant:
        gs_n = 128
        gs_k = 128
        x_quant_type = None
        w_quant_type = None
        is_mx_quant = (x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2")
                       and scale_dtype == "float8_e8m0")
        is_compatible = x1_dtype == 'int8' and scale_dtype not in ("uint64", "int64")

        if not is_mx_quant:
            x_quant_type, w_quant_type = infer_quant_mode(
                scale, per_token_scale, trans_b, shape_m, shape_k, shape_n,
                group_num, split_m, split_k, gs_k, gs_n)

        is_two_scale = x_quant_type == "perTensor" and w_quant_type == "perTensor"
        is_pertile = x_quant_type == "perGroup" and w_quant_type == "perBlock"
        is_pertoken = x_quant_type == "perToken"

        if w_quant_type == "perTensor":
            scale = scale.reshape([group_num, 1])
        if x_quant_type == "perTensor":
            per_token_scale = per_token_scale.reshape([group_num, 1])

        if group_list_type == 1:
            group_list = np.cumsum(group_list)

        if is_mx_quant:
            if split_m:
                pertoken_scale_mx = per_token_scale
                m, k0, k1 = pertoken_scale_mx.shape
                pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
                if trans_a:
                    x = np.swapaxes(x, -1, -2)
                    pertoken_scale_mx = pertoken_scale_mx.transpose()
                pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
                x1_dims = len(x.shape)
                x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x.shape[-1]
                x = np.pad(x, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)],
                            mode='constant', constant_values=0)
                x = x * pertoken_scale_mx_broadcast
                x = convert_to_high_precision(x, x1_dtype)
                for i in range(group_num):
                    scale_g = scale[i]
                    if not trans_b:
                        scale_g = transform_tensor(scale_g)
                    else:
                        n, k0, k1 = scale_g.shape
                        scale_g = scale_g.reshape(n, k0 * k1)
                    x2_g = weight[i]
                    if trans_b:
                        x2_g = np.swapaxes(x2_g, -1, -2)
                        scale_g = scale_g.transpose()
                    bias_g = bias[i] if len(bias.shape) == 2 else None
                    deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
                    x2_dims = len(x2_g.shape)
                    x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2_g.shape[-2]
                    x2_g = np.pad(x2_g, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
                                  mode='constant', constant_values=0)
                    x2_g = x2_g * deq_scale_mx_broadcast
                    if i == 0:
                        x1_temp = x[:group_list[i], :]
                    else:
                        x1_temp = x[group_list[i - 1]:group_list[i], :]
                    x2_g = convert_to_high_precision(x2_g, x2_dtype)
                    out = single_group_mm_cal(x1_temp, x2_g, bias_g, None, None, None,
                                              out_dtype, bias_dtype,
                                              is_pertoken, is_two_scale, is_mx_quant, is_compatible)
                    outs.append(out)
            elif split_k:
                pertoken_scale_mx = per_token_scale
                deq_scale_mx = scale
                M = x.shape[-1]
                N = weight.shape[-1]
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
                    x1_g = x[pre: cur, :]
                    x2_g = weight[pre: cur, :]
                    pertoken_scale_mx_g = pertoken_scale_mx[scale_k_offset: cur_scale_k, :]
                    deq_scale_mx_g = deq_scale_mx[scale_k_offset: cur_scale_k, :]
                    scale_k_offset = cur_scale_k
                    if trans_a:
                        x1_g = np.swapaxes(x1_g, -1, -2)
                        pertoken_scale_mx_g = pertoken_scale_mx_g.transpose()
                    pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx_g, 32, axis=-1)
                    x1_dims = len(x1_g.shape)
                    x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1_g.shape[-1]
                    x1_g = np.pad(x1_g, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)],
                                  mode='constant', constant_values=0)
                    x1_g = x1_g * pertoken_scale_mx_broadcast
                    x1_g = convert_to_high_precision(x1_g, x1_dtype)
                    if trans_b:
                        x2_g = np.swapaxes(x2_g, -1, -2)
                        deq_scale_mx_g = deq_scale_mx_g.transpose()
                    deq_scale_mx_broadcast = np.repeat(deq_scale_mx_g, 32, axis=-2)
                    x2_dims = len(x2_g.shape)
                    x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2_g.shape[-2]
                    x2_g = np.pad(x2_g, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
                                  mode='constant', constant_values=0)
                    x2_g = x2_g * deq_scale_mx_broadcast
                    x2_g = convert_to_high_precision(x2_g, x2_dtype)
                    out = single_group_mm_cal(x1_g, x2_g, None, None, None, None,
                                              out_dtype, None,
                                              is_pertoken, is_two_scale, is_mx_quant, is_compatible)
                    outs.append(out)
        elif is_pertile:
            if trans_a:
                x = np.swapaxes(x, -1, -2)
                per_token_scale = np.swapaxes(per_token_scale, -1, -2)
            if trans_b:
                weight = np.swapaxes(weight, -1, -2)
                scale = np.swapaxes(scale, -1, -2)
            x = convert_to_high_precision(x, x1_dtype)
            weight = convert_to_high_precision(weight, x2_dtype)
            per_token_scale = torch.from_numpy(per_token_scale).to(torch.float32)
            scale = torch.from_numpy(scale).to(torch.float32)
            if split_m:
                for i in range(group_num):
                    scale_g = scale[i]
                    x2_g = weight[i]
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    x1_g = x[pre: cur, :]
                    pertoken_scale_g = per_token_scale[pre: cur, :]
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
                    x1_g = x[:, pre: cur]
                    x2_g = weight[pre: cur, :]
                    cur_scale_k = math.ceil(k_g / gs_k)
                    scale_k_offset = 0 if i == 0 else group_list[i - 1] // gs_k + i
                    pertoken_scale_g = per_token_scale[:, scale_k_offset: (scale_k_offset + cur_scale_k)]
                    scale_g = scale[scale_k_offset: (scale_k_offset + cur_scale_k), :]
                    out = pertile_mm_cal(x1_g, x2_g, pertoken_scale_g, scale_g, out_dtype)
                    outs.append(out)
        else:
            x = convert_to_high_precision(x, x1_dtype)
            weight = convert_to_high_precision(weight, x2_dtype)
            if is_mx_quant:
                pass
            elif scale_dtype == "uint64" or scale_dtype == "int64":
                deq_scale = np.load(testcase_name + "_deq_scale.npy")
                deq_scale_tensor = torch.from_numpy(deq_scale)
            else:
                deq_scale = scale.astype(np.float32)
                deq_scale_tensor = torch.from_numpy(deq_scale)

            if is_pertoken and group_type == 0:
                pertoken_scale_slice = torch.unsqueeze(torch.from_numpy(per_token_scale), dim=1).to(torch.float32)
            elif is_two_scale:
                two_scale = scale_generate(per_token_scale * deq_scale)
                two_scale_tensor = torch.unsqueeze(torch.from_numpy(two_scale), dim=1).to(torch.float32)
            elif x_quant_type == "perTensor":
                deq_scale_tensor *= torch.from_numpy(per_token_scale).to(torch.float32)
            pertoken_scale_temp = None
            deq_scale_temp = None
            two_scale_temp = None
            bias_temp = None
            if group_type == 0:
                for i in range(group_num):
                    if i == 0:
                        x1_temp = x[:group_list[i], :]
                    else:
                        x1_temp = x[group_list[i - 1]:group_list[i], :]
                    x2_temp = weight[i]
                    if trans_b:
                        x2_temp = np.swapaxes(x2_temp, -1, -2)
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    if bias is not None and bias.size > 0:
                        if bias.shape[0] == 1:
                            bias_temp = bias[0]
                        else:
                            bias_temp = bias[i]
                    if is_pertoken:
                        pertoken_scale_temp = pertoken_scale_slice[pre:cur]
                        if deq_scale_tensor.shape[0] == 1:
                            deq_scale_temp = deq_scale_tensor[0]
                        else:
                            deq_scale_temp = deq_scale_tensor[i]
                    elif is_two_scale:
                        if two_scale_tensor.shape[0] == 1:
                            two_scale_temp = two_scale_tensor[0]
                        else:
                            two_scale_temp = two_scale_tensor[i]
                    else:
                        if deq_scale_tensor.shape[0] == 1:
                            deq_scale_temp = deq_scale_tensor[0]
                        else:
                            deq_scale_temp = deq_scale_tensor[i]
                    out = single_group_mm_cal(
                        x1_temp, x2_temp, bias_temp, pertoken_scale_temp,
                        deq_scale_temp, two_scale_temp,
                        out_dtype, bias_dtype,
                        is_pertoken, is_two_scale, is_mx_quant, is_compatible)
                    outs.append(out)
            elif group_type == 2:
                for i in range(group_num):
                    M = x.shape[-1]
                    N = weight.shape[-1]
                    pre = 0 if i == 0 else group_list[i - 1]
                    cur = group_list[i]
                    if cur - pre == 0:
                        out = np.zeros((M, N), dtype=out_dtype)
                        outs.append(out)
                        continue
                    x1_g = x[pre:cur, :]
                    if trans_a:
                        x1_g = np.swapaxes(x1_g, -1, -2)
                    x2_g = weight[pre:cur, :]

                    if bias is not None and bias.size > 0:
                        if bias.shape[0] == 1:
                            bias_temp = bias[0]
                        else:
                            bias_temp = bias[i]
                    if is_pertoken:
                        if per_token_scale.shape[0] == 1:
                            pertoken_scale_temp = per_token_scale[0].flatten()
                        else:
                            pertoken_scale_temp = per_token_scale[i].flatten()
                        pertoken_scale_temp = torch.unsqueeze(
                            torch.from_numpy(pertoken_scale_temp), dim=1
                        ).to(torch.float32)
                        if deq_scale_tensor.shape[0] == 1:
                            deq_scale_temp = deq_scale_tensor[0]
                        else:
                            deq_scale_temp = deq_scale_tensor[i]
                    elif is_two_scale:
                        if two_scale_tensor.shape[0] == 1:
                            two_scale_temp = two_scale_tensor[0]
                        else:
                            two_scale_temp = two_scale_tensor[i]
                    else:
                        if deq_scale_tensor.shape[0] == 1:
                            deq_scale_temp = deq_scale_tensor[0]
                        else:
                            deq_scale_temp = deq_scale_tensor[i]
                    out = single_group_mm_cal(
                        x1_g, x2_g, bias_temp, pertoken_scale_temp,
                        deq_scale_temp, two_scale_temp,
                        out_dtype, bias_dtype,
                        is_pertoken, is_two_scale, is_mx_quant, is_compatible)
                    outs.append(out)
    elif is_weight_quant:
        if weight_quant_s8s4_flag:
            assert not trans_a and not trans_b

            if x2_format == "FRACTAL_NZ":
                weight = trans_nz2nd(weight)

            outs = cal_weight_quant_s8s4(x, weight, bias, scale, antiquant_scale, per_token_scale, group_list,
                                         group_list_type, _unwrap_dtype(kwargs['output_dtypes'][0]))
        elif weight_quant_mxfp4_flag:
            assert x2_format == "FRACTAL_NZ"
            x2_nd = trans_nz2nd(weight)
            # Convert FP4 to float before passing to torch
            if x2_dtype in ("float4_e2m1", "float4_e1m2"):
                x2_nd = x2_nd.astype(np.float32)
            x2_nd = torch.from_numpy(x2_nd).to(torch.float16) if x1_dtype == "float16" else torch.from_numpy(x2_nd)
            x1_t = torch.from_numpy(x.astype(np.float32))

            shape_k = x2_nd.shape[-2] if not trans_b else x2_nd.shape[-1]
            shape_n = x2_nd.shape[-1] if not trans_b else x2_nd.shape[-2]

            if (bias is not None) and (len(bias) != 0):
                bias_t = torch.from_numpy(bias.astype(np.float32))
            else:
                bias_t = torch.from_numpy(np.zeros([x2_nd.shape[0], shape_n], dtype=np.float32))

            if antiquant_scale.ndim == 4:
                s = antiquant_scale.shape
                antiquant_scale = antiquant_scale.reshape(s[0], s[1], -1)
            if per_token_scale is not None and per_token_scale.ndim == 3:
                s = per_token_scale.shape
                per_token_scale = per_token_scale.reshape(s[0], -1)

            group_size = get_per_group_size(antiquant_scale, shape_k, trans_b)

            antiquant_scale_broadcast, pertoken_scale_broadcast = get_mx_scale(antiquant_scale, group_size,
                                                                               per_token_scale, trans_b, x1_dtype)
            for i in range(len(group_list)):
                start_m, end_m = get_group_m(group_list, group_list_type, i)
                x1_group = x1_t[start_m:end_m, :]
                weight_group = x2_nd[i]
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
                group_out += bias_t[i, :].reshape(1, -1).to(torch.float32)
                if kwargs['output_dtypes'][0] == "bfloat16":
                    outs.append(group_out.numpy().astype(ml_dtypes.bfloat16))
                else:
                    outs.append(group_out.numpy().astype(np.float16))
        else:
            outs = cal_weight_quant_a16w8(x, weight, bias, antiquant_scale, antiquant_offset, group_list, group_list_type,
                                          x1_dtype, x2_dtype, _unwrap_dtype(kwargs['output_dtypes'][0]), trans_b)
    else:
        if group_list_type == 1:
            group_list = np.cumsum(group_list)
        if group_type == 0:
            _, w_outer, w_inner = weight.shape
            if x2_dtype == "float16" or x2_dtype == "bfloat16":
                w_outer_align = (w_outer + 16) // w_outer * w_outer
                w_inner_align = (w_inner + 16) // w_inner * w_inner

            if trans_a:
                x = np.swapaxes(x, -1, -2)
            x = torch.from_numpy(x.astype(np.float32))
            for i in range(group_num):
                x2_g = weight[i]
                if x2_format == "FRACTAL_NZ":
                    x2_g = x2_g.transpose((1, 2, 0, 3)).reshape(w_outer_align, w_inner_align)
                    x2_g = x2_g[:w_outer, :w_inner]

                if trans_b:
                    x2_g = np.swapaxes(x2_g, -1, -2)
                x2_g = torch.from_numpy(x2_g.astype(np.float32))
                if i == 0:
                    x1_temp = x[:group_list[i], :]
                else:
                    x1_temp = x[group_list[i - 1]:group_list[i], :]
                bias_temp = None
                if bias is not None and bias.size > 0:
                    bias_temp = bias[i]

                out = torch.matmul(x1_temp, x2_g)
                if bias_temp is not None:
                    bias_temp = torch.from_numpy(bias_temp.astype(np.float32))
                    out += bias_temp
                outs.append(out.numpy().astype(out_dtype))
        elif group_type == 2:
            x = torch.from_numpy(x.astype(np.float32))
            weight = torch.from_numpy(weight.astype(np.float32))
            for i in range(group_num):
                M = x.shape[-1]
                N = weight.shape[-1]
                pre = 0 if i == 0 else group_list[i - 1]
                cur = group_list[i]
                if cur - pre == 0:
                    out = np.zeros((M, N), dtype=out_dtype)
                    outs.append(out)
                    continue
                x1_g = x[pre:cur, :]
                if trans_a:
                    x1_g = x1_g.transpose(-1, -2)
                x2_g = weight[pre:cur, :]
                out = torch.matmul(x1_g, x2_g).numpy().astype(out_dtype)
                outs.append(out)

    for suffix in ["_deq_scale.npy", "_offset.npy"]:
        path = testcase_name + suffix
        if os.path.exists(path):
            os.remove(path)

    real_out = outs if not outs else np.concatenate(outs, axis=0)
    return real_out
