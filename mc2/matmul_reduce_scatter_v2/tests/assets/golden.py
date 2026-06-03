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
        "matmul_reduce_scatter_v2": "matmul_reduce_scatter_v2_golden"
    }
}

import numpy as np
import torch


def matmul_reduce_scatter_v2_golden(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    quant_scale=None,
    group="",
    reduce_op="sum",
    is_trans_a=False,
    is_trans_b=False,
    comm_turn=0,
    rank_size=0,
    block_size=0,
    group_size=0,
    is_amax_out=False,
    y_dtype=0,
    comm_mode="aicpu",
    **kwargs
):
    x1_dtype = kwargs.get('x1_dtype', 'bf16')
    x2_dtype = kwargs.get('x2_dtype', 'bf16')
    is_quant = x1_dtype not in ['fp16', 'bf16']
    is_mxFp = kwargs.get('x1_scale_dtype', '') == 'fp8_e8m0'
    per_block_flag = kwargs.get('per_block_flag', False)
    
    x1 = torch.from_numpy(x1.astype(np.float32))
    x2 = torch.from_numpy(x2.astype(np.float32))
    if bias is not None:
        bias = torch.from_numpy(bias.astype(np.float32))
    
    if is_trans_b:
        x2 = x2.transpose()
        if x2_scale is not None and per_block_flag:
            x2_scale = x2_scale.transpose()
    
    if not is_quant:
        output = torch.matmul(x1, x2)
        if bias is not None:
            output += bias
    else:
        if x1_scale is not None:
            x1_scale = torch.from_numpy(x1_scale.astype(np.float32))
        if x2_scale is not None:
            x2_scale = torch.from_numpy(x2_scale.astype(np.float32))
        
        if per_block_flag:
            output = per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale)
        elif is_mxFp:
            x1_np = x1.numpy()
            x2_np = x2.numpy()
            x1scale_np = x1_scale.numpy() if x1_scale is not None else None
            x2scale_np = x2_scale.numpy() if x2_scale is not None else None
            output = mxfp_cpu_compute(x1_np, x2_np, x1scale_np, x2scale_np)
            if bias is not None:
                output += bias
        else:
            output = torch.matmul(x1, x2)
            if bias is not None:
                output += bias
            if x1_scale is not None and x2_scale is not None:
                x1scale_np = x1_scale.numpy()
                x2scale_np = x2_scale.numpy()
                double_scale = scale_generate(x1scale_np * x2scale_np)
                double_scale_tensor = torch.unsqueeze(torch.from_numpy(double_scale), dim=1).to(torch.float32)
                output *= double_scale_tensor
    
    output = reduce_scatter_compute(output, rank_size)
    return output.numpy()


def reduce_scatter_compute(output, rank_size):
    output_all = output.repeat(rank_size, 1)
    scatter_shape_m = output.shape[0] // rank_size
    scatter_output = output_all.narrow(0, 0, scatter_shape_m)
    return scatter_output


def per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale):
    if x1.dim() != x1_scale.dim():
        raise ValueError(f"x1.dim() != x1_scale.dim(), x1.dim()={x1.dim()}, x1_scale.dim()={x1_scale.dim()}")
    if x2.dim() != x2_scale.dim():
        raise ValueError(f"x2.dim() != x2_scale.dim(), x2.dim()={x2.dim()}, x2_scale.dim()={x2_scale.dim()}")
    
    batch_x1 = np.array(x1.shape[:-2]).astype(int).tolist()
    batch_x2 = np.array(x2.shape[:-2]).astype(int).tolist()
    batch_out = fetch_batch_broadcast(batch_x1, batch_x2)
    
    if batch_x1 != batch_out:
        x1 = value_batch_broadcast(x1, batch_out)
        x1_scale = value_batch_broadcast(x1_scale, batch_out)
    if batch_x2 != batch_out:
        x2 = value_batch_broadcast(x2, batch_out)
        x2_scale = value_batch_broadcast(x2_scale, batch_out)
    
    batch_all = 1
    is2dim = True
    if batch_out != []:
        is2dim = False
        batch_all = np.prod(batch_out)
        x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
        x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
        x1_scale = x1_scale.reshape([batch_all] + list(x1_scale.shape[-2:]))
        x2_scale = x2_scale.reshape([batch_all] + list(x2_scale.shape[-2:]))
    
    m = x1.shape[-2]
    k = x1.shape[-1]
    n = x2.shape[-1]
    out = torch.zeros(m, n)
    if x2_scale.dim() > 2 or x1_scale.dim() > 2:
        out = torch.zeros(batch_all, m, n)
    
    group_size_m, group_size_n, group_size_k = unpack_group_size(group_size)
    for i in range(batch_all):
        for m_idx in range((m + group_size_m - 1) // group_size_m):
            m_start = m_idx * group_size_m
            m_end = min((m_idx + 1) * group_size_m, m)
            for n_idx in range((n + group_size_n - 1) // group_size_n):
                n_start = n_idx * group_size_n
                n_end = min((n_idx + 1) * group_size_n, n)
                for k_idx in range((k + group_size_k - 1) // group_size_k):
                    k_start = k_idx * group_size_k
                    k_end = min((k_idx + 1) * group_size_k, k)
                    if batch_all == 1 and is2dim:
                        block_output = torch.matmul(x1[m_start:m_end, k_start:k_end],
                                                    x2[k_start:k_end, n_start:n_end]) * x1_scale[m_idx, k_idx] * x2_scale[k_idx, n_idx]
                        out[m_start:m_end, n_start:n_end] += block_output
                    else:
                        out[i, m_start:m_end, n_start:n_end] += torch.matmul(x1[i, m_start:m_end, k_start:k_end],
                                                                              x2[i, k_start:k_end, n_start:n_end]) * x1_scale[i, m_idx, k_idx] * x2_scale[i, k_idx, n_idx]
    
    if x2_scale.dim() > 2 or x1_scale.dim() > 2:
        out_shape = batch_out
        out_shape.append(m)
        out_shape.append(n)
        out = torch.reshape(out, out_shape)
    else:
        out = torch.reshape(out, [m, n])
    return out


def mxfp_cpu_compute(x1, x2, x1scale, x2scale):
    for array in [x1, x2, x1scale, x2scale]:
        if array.ndim != 2:
            raise ValueError("[ERROR] array.ndim must be 2")
    if x1.shape[0] != x1scale.shape[0]:
        raise ValueError(f"x1.shape[0] != x1scale.shape[0], x1.shape[0]={x1.shape[0]}, x1scale.shape[0]={x1scale.shape[0]}")
    if x2.shape[1] != x2scale.shape[1]:
        raise ValueError(f"x2.shape[1] != x2scale.shape[1], x2.shape[1]={x2.shape[1]}, x2scale.shape[1]={x2scale.shape[1]}")
    if x1.shape[1] != x2.shape[0]:
        raise ValueError(f"x1.shape[1] != x2.shape[0], x1.shape[1]={x1.shape[1]}, x2.shape[0]={x2.shape[0]}")
    if x1scale.shape[1] != x2scale.shape[0]:
        raise ValueError(f"x1scale.shape[1] != x2scale.shape[0], x1scale.shape[1]={x1scale.shape[1]}, x2scale.shape[0]={x2scale.shape[0]}")
    
    repeated_x1scale = np.repeat(x1scale, 32, axis=-1)
    repeated_x2scale = np.repeat(x2scale, 32, axis=-2)
    x1_pad_length = repeated_x1scale.shape[-1] - x1.shape[-1]
    x2_pad_len = repeated_x2scale.shape[-2] - x2.shape[-2]
    x1_pad_tuple = [(0, 0)] * (len(x1.shape) - 1) + [(0, x1_pad_length)]
    x2_pad_tuple = [(0, 0)] * (len(x2.shape) - 2) + [(0, x2_pad_len)] + [(0, 0)]
    padded_x1 = np.pad(x1, x1_pad_tuple, mode='constant', constant_values=0)
    padded_x2 = np.pad(x2, x2_pad_tuple, mode='constant', constant_values=0)
    
    padded_x1 = torch.from_numpy(padded_x1)
    padded_x2 = torch.from_numpy(padded_x2)
    repeated_x1scale = torch.from_numpy(repeated_x1scale)
    repeated_x2scale = torch.from_numpy(repeated_x2scale)
    output = torch.matmul(padded_x1 * repeated_x1scale, padded_x2 * repeated_x2scale)
    return output


def scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32_deq_scale &= 0XFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
    return fp32_deq_scale


def unpack_group_size(group_size):
    if group_size == -1:
        return 0, 0, 0
    group_size_m = (group_size >> 32) & 0xFFFF
    group_size_n = (group_size >> 16) & 0xFFFF
    group_size_k = group_size & 0xFFFF
    return group_size_m, group_size_n, group_size_k


def fetch_batch_broadcast(batch_x1, batch_x2):
    import copy
    batch_out = copy.deepcopy(batch_x1) if len(batch_x1) > len(batch_x2) else copy.deepcopy(batch_x2)
    
    min_len, max_len = 0, 0
    if batch_x2 != batch_x1 and batch_x1 and batch_x2:
        min_len = min(len(batch_x1), len(batch_x2))
        max_len = max(len(batch_x1), len(batch_x2))
    
    for idx in range(min_len):
        batch_out[-(idx + 1)] = max(batch_x1[-(idx + 1)], batch_x2[-(idx + 1)])
    
    if len(batch_x1) > len(batch_x2):
        for idx in range(min_len, max_len):
            batch_out[-(idx + 1)] = batch_x1[-(idx + 1)]
    else:
        for idx in range(min_len, max_len):
            batch_out[-(idx + 1)] = batch_x2[-(idx + 1)]
    return batch_out


def value_batch_broadcast(x, batch):
    new_x_shape = batch + list(x.shape[-2:])
    x = torch.broadcast_to(x, new_x_shape)
    return x