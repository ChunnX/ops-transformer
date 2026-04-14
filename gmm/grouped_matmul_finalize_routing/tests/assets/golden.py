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
            "grouped_matmul_finalize_routing": "grouped_matmul_finalize_routing_golden"
        }
}

from typing import List
import numpy as np
import torch


def grouped_matmul_finalize_routing_golden(x, w, scale=None, bias=None, pertoken_scale=None, group_list=None,
                                           shared_input=None, logit=None, row_index=None, offset=None,
                                           dtype: int = 0, shared_input_weight: float = 1.0,
                                           shared_input_offset: int = 0, transpose_x: bool = False,
                                           transpose_w: bool = False, output_bs: int = 0, group_list_type: int = 1,
                                           tuning_config: List[int] = [0], **kwargs):
    x1_dtype = x.dtype.name
    x2_dtype = w.dtype.name
    is_mxa8w4 = (x1_dtype == 'float8_e4m3fn') and (x2_dtype in ('float4_e2m1', 'float4_e2m1fn'))

    if is_mxa8w4:
        return _mxa8w4_golden(x, w, scale, bias, pertoken_scale, group_list,
                              shared_input, logit, row_index,
                              transpose_w, output_bs, shared_input_weight, shared_input_offset,
                              group_list_type, kwargs['output_dtypes'][0])
    else:
        return _original_golden(x, w, scale, bias, pertoken_scale, group_list,
                           shared_input, logit, row_index, offset,
                           dtype, shared_input_weight, shared_input_offset,
                           transpose_x, transpose_w, output_bs, group_list_type,
                           tuning_config, **kwargs)


def _mxa8w4_golden(x1, x2_all, scale, bias, pertoken_scale, group_list,
                   shared_input, logit, row_index,
                   transpose_w, output_bs, shared_input_weight, shared_input_offset,
                   group_list_type, out_dtype):
    if group_list_type == 1 and group_list is not None:
        group_list = np.cumsum(group_list)
    group_num = len(group_list)

    # 激活反量化: pertoken_scale shape (m, ceil(k/64), 2)
    # reshape -> (m, ceil(k/32)), 每个scale值对应32个激活值
    m_ps, k0, k1 = pertoken_scale.shape
    pertoken_scale_flat = pertoken_scale.reshape(m_ps, k0 * k1)
    pertoken_scale_broadcast = np.repeat(pertoken_scale_flat, 32, axis=-1)

    x1 = x1.astype(np.float32)
    pad_len = pertoken_scale_broadcast.shape[-1] - x1.shape[-1]
    if pad_len > 0:
        x1 = np.pad(x1, [(0, 0), (0, pad_len)], mode='constant', constant_values=0)
    x1 = x1 * pertoken_scale_broadcast
    x1 = torch.from_numpy(x1)

    # 分组矩阵乘法
    outs = []
    for i in range(group_num):
        scale_g = scale[i]  # (n, ceil(k/64), 2)
        scale_g = scale_g.reshape(scale_g.shape[0], -1)  # (n, ceil(k/32))

        x2 = x2_all[i].astype(np.float32)  # (n, k) from viewShape (e, n, k)

        # transpose: (n, k) -> (k, n), scale (n, ceil(k/32)) -> (ceil(k/32), n)
        x2 = np.swapaxes(x2, -1, -2)
        scale_g = scale_g.transpose()

        # 权重反量化: 每个scale值对应32个权重值
        scale_broadcast = np.repeat(scale_g, 32, axis=-2)
        pad_len = scale_broadcast.shape[-2] - x2.shape[-2]
        if pad_len > 0:
            x2 = np.pad(x2, [(0, pad_len), (0, 0)], mode='constant', constant_values=0)
        x2 = x2 * scale_broadcast
        x2 = torch.from_numpy(x2)

        # 获取当前专家的激活
        start = 0 if i == 0 else group_list[i - 1]
        x1_temp = x1[start:group_list[i], :]

        out = torch.matmul(x1_temp, x2).numpy().astype(out_dtype)

        if bias is not None:
            out += bias[i, :].reshape(1, -1).astype(np.float32)
        outs.append(out)

    gmm_out = np.concatenate(outs, axis=0) if outs else np.array([])
    return combine_func(gmm_out, logit, shared_input, shared_input_weight,
                        row_index, output_bs, shared_input_offset)


def _original_golden(x, w, scale=None, bias=None, pertoken_scale=None, group_list=None,
                                           shared_input=None, logit=None, row_index=None, offset=None,
                                           dtype: int = 0, shared_input_weight: float = 1.0,
                                           shared_input_offset: int = 0, transpose_x: bool = False,
                                           transpose_w: bool = False, output_bs: int = 0, group_list_type: int = 1,
                                           tuning_config: List[int] = [0], **kwargs):
    """
    GMMFR Golden函数
    
    参数说明：
    - x: 激活输入，shape=(m, k)，支持FP8/FP4/INT8等
    - w: 权重，shape=(e, n, k)或(e, k, n)取决于transpose_w，支持NZ格式
    - scale: 权重反量化scale，shape=(e, n, ceil(k/64), 2)或(e, ceil(k/64), n, 2)
    - pertoken_scale: 激活反量化scale，shape=(m, ceil(k/64), 2)
    - group_list: 专家token分布列表
    - shared_input: 共享专家输入，shape=(bsdp, n)
    - logit: 专家logit权重，shape=(m,)
    - row_index: scatter-add行索引，shape=(m,)
    - transpose_w: 权重是否转置，MxA8W4场景固定为True
    """
    x1, x2_all = x, w
    x1_dtype = x1.dtype.name
    x2_dtype = w.dtype.name
    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    trans_b = transpose_w
    
    # 获取专家数量
    outs = []
    if group_list is not None:
        group_num = len(group_list)
 
    # group_list_type: 0=cumsum模式(前缀和), 1=count模式(直接计数)
    if group_list_type == 1 and group_list is not None:
        group_list = np.cumsum(group_list)
    
    M = x1.shape[0]
    N = x2_all.shape[-1]
    
    # =========================================================================
    # 步骤1: 激活反量化 (x1 * pertoken_scale)
    # pertoken_scale shape: (m, ceil(k/64), 2)
    # 每个scale值对应32个激活值，broadcast后shape: (m, ceil(k/64)*2)
    # =========================================================================
    if pertoken_scale is not None:
        pertoken_scale_mx = pertoken_scale
        m, k0, k1 = pertoken_scale_mx.shape
        # reshape: (m, ceil(k/64), 2) -> (m, ceil(k/64)*2)
        pertoken_scale_mx = pertoken_scale_mx.reshape(m, k0 * k1)
        # broadcast: 每个scale值重复32次
        pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
    
    x1_dims = len(x1.shape)
 
    # FP4类型转换为FP32
    if x1_dtype in ('float4_e2m1', 'float4_e2m1fn', 'float4_e1m2'):
        x1 = x1.astype(np.float32)

    # 激活反量化：pad + scale
    if pertoken_scale is not None:
        x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]
        x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x1 = x1 * pertoken_scale_mx_broadcast
    
    # 升精度到FP32（使用torch处理特殊dtype）
    x1 = convert_to_high_precision(x1, x1_dtype)
 
    # =========================================================================
    # 步骤2: 分组矩阵乘法 (x1_i @ x2_i * scale_i + bias_i)
    # =========================================================================
    for i in range(group_num):
        # ---------------------------------------------------------------------
        # 2.1 处理权重scale反量化
        # scale[i] shape: (n, ceil(k/64), 2) 当transpose_w=True
        #              或 (ceil(k/64), n, 2) 当transpose_w=False
        # ---------------------------------------------------------------------
        if scale is not None:
            scale_g = scale[i]
            if trans_b is False:
                # transpose_w=False: 转换scale形状
                scale_g = transform_tensor(scale_g)
            else:
                # transpose_w=True: MxA8W4场景走此分支
                # scale_g shape: (n, ceil(k/64), 2) -> (n, ceil(k/64)*2)
                n, k0, k1 = scale_g.shape
                scale_g = scale_g.reshape(n, k0 * k1)
 
        x2 = x2_all[i]
  
        # FP4类型转换
        if x2_dtype in ('float4_e2m1', 'float4_e2m1fn', 'float4_e1m2'):
            x2 = x2.astype(np.float32)
        
        # ---------------------------------------------------------------------
        # 2.2 处理权重transpose
        # MxA8W4场景：transpose_w=True，x2 viewShape=(e, n, k) -> (e, k, n)
        # ---------------------------------------------------------------------
        if trans_b:
            x2 = np.swapaxes(x2, -1, -2)
            if scale is not None:
                scale_g = scale_g.transpose()
        
        # ---------------------------------------------------------------------
        # 2.3 权重反量化：broadcast + scale
        # 每个scale值对应32个权重值
        # ---------------------------------------------------------------------
        if scale is not None:
            deq_scale_mx_broadcast = np.repeat(scale_g, 32, axis=-2)
        
        x2_dims = len(x2.shape)
        if scale is not None:
            x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2.shape[-2]
            x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
                        mode='constant', constant_values=0)
            x2 = x2 * deq_scale_mx_broadcast
        
        # ---------------------------------------------------------------------
        # 2.4 获取当前专家的激活并执行矩阵乘法
        # ---------------------------------------------------------------------
        if i == 0:
            x1_temp = x1[:group_list[i], :]
        else:
            x1_temp = x1[group_list[i-1]:group_list[i], :]
        
        # 权重升精度
        x2 = convert_to_high_precision(x2, x2_dtype)
        
        # 矩阵乘法
        out = single_group_mm_cal(x1_temp, x2, out_dtype)
        
        # 添加bias
        if bias is not None:
            out += bias[i, :].reshape(1, -1)
        outs.append(out)
 
    # 拼接所有专家输出
    gmm_out = outs if not outs else np.concatenate(outs, axis=0)
    
    # =========================================================================
    # 步骤3: 路由专家分配 + 共享专家融合
    # =========================================================================
    final_out = combine_func(gmm_out, logit, shared_input, shared_input_weight,
                             row_index, output_bs, shared_input_offset)
    return final_out


def convert_to_high_precision(input_tensor, input_type):
    if input_type in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e2m1fn", "float4_e1m2", "hifloat8"):
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))
    elif input_type in ("int4",):
        input_tensor = torch.from_numpy(input_tensor.astype(np.int32)).to(torch.int32)
    else:
        input_tensor = torch.from_numpy(input_tensor).to(torch.int32)
    return input_tensor


def transform_tensor(input_tensor):
    transposed = np.transpose(input_tensor, axes=(0, 2, 1))
    batch_size, height, width = transposed.shape
    result = transposed.reshape(batch_size * height, width)
    return result


def single_group_mm_cal(x1, x2, out_dtype):
    out = torch.matmul(x1, x2)
    torch.set_printoptions(threshold=torch.inf)
    # 检查是否有inf
    has_inf = torch.isinf(out).any()
    # 检查是否有nan
    has_nan = torch.isnan(out).any()
 
    out = out.numpy().astype(out_dtype)
    return out


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
    out = out[index].reshape(output_bs, top_k, x.shape[-1]).sum(axis=1) # m / total_token = topk
    if residual is not None:
        out[offset:offset + residual.shape[0], :] += resid_scale * residual.astype(np.float32)
    return out