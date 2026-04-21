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
        "allto_all_all_gather_batch_mat_mul": "allto_all_all_gather_batch_mat_mul_golden"
    }
}

import torch
import torch.distributed as dist

def activate(act_type, input):
    output = input
    if not act_type or act_type == 0:
        output = input
    elif act_type == 1:  # GeLU
        gelu = torch.nn.GELU(approximate="tanh")
        output = gelu(input)
    elif act_type == 2:  # SiLu
        output = input / (1 + torch.exp(-input))
    elif act_type == 3:  # ReLU
        relu = torch.nn.ReLU()
        output = relu(input)
    elif act_type == 4:  # FastGeLU
        output = input / (1 + torch.exp(-1.702 * input))
    return output

def allto_all_all_gather_batch_mat_mul_golden(
    x,
    weight,
    bias=None,
    group_ep="",
    group_tp="",
    ep_world_size: int = 1,
    tp_world_size: int = 1,
    x_shard_type: int = 0,
    act_type: int = 0,
    transpose_weight: bool = False,
    output_y2_flag: bool = False,
    output_y3_flag: bool = False,
    **kwargs
):
    """
    AllToAllAllGatherBatchMatMul golden implementation
    
    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Bias tensor (optional)
        group_ep: Communication group for EP
        group_tp: Communication group for TP
        ep_world_size: World size for EP
        tp_world_size: World size for TP
        x_shard_type: Shard type (0 or 1)
        act_type: Activation type (0: none, 1: GeLU, 2: SiLu, 3: ReLU, 4: FastGeLU)
        transpose_weight: Whether to transpose weight
        output_y2_flag: Whether to output y2
        output_y3_flag: Whether to output y3
        **kwargs: Additional arguments
    
    Returns:
        y1: Output tensor
        y2: Gather output (if output_y2_flag is True)
        y3: BMM output (if output_y3_flag is True)
    """
    x_float = x.to(torch.float32)
    weight_float = weight.to(torch.float32)
    
    if transpose_weight:
        weight_float = weight_float.permute(0, 2, 1)
    
    E, C, H = x_float.shape[0], x_float.shape[1], x_float.shape[2]
    M = weight_float.shape[2]
    
    all_to_all_out = torch.zeros_like(x_float)
    try:
        dist.all_to_all_single(all_to_all_out, x_float)
    except:
        all_to_all_out = x_float
    
    reshape_1_shape = [ep_world_size, E // ep_world_size, C, H]
    all_to_all_out = all_to_all_out.reshape(reshape_1_shape).permute(1, 0, 2, 3).contiguous()
    
    tensor_allgather_shape = [tp_world_size * E // ep_world_size, ep_world_size, C, H]
    gather_out = torch.zeros(tensor_allgather_shape, dtype=x_float.dtype)
    try:
        dist._all_gather_base(gather_out, all_to_all_out)
    except:
        gather_out = torch.cat([all_to_all_out] * tp_world_size, dim=0)
    
    reshape_2_shape = [tp_world_size, E // ep_world_size, ep_world_size, C, H]
    gather_out = gather_out.reshape(reshape_2_shape)
    
    if x_shard_type == 0:
        gather_out = gather_out.permute(1, 2, 3, 0, 4)
    else:
        gather_out = gather_out.permute(1, 2, 0, 3, 4)
    
    reshape_3_shape = [E // ep_world_size, ep_world_size * C, H]
    gather_out = gather_out.reshape(reshape_3_shape)
    
    bmm_out = torch.bmm(gather_out, weight_float)
    
    if bias is not None:
        bias_float = bias.to(torch.float32)
        if len(bias_float.shape) == 2:
            bias_float = bias_float.reshape([bias_float.shape[0], 1, bias_float.shape[1]])
        bmm_out = bmm_out + bias_float
    
    y1 = activate(act_type, bmm_out)
    
    y2 = gather_out if output_y2_flag else None
    y3 = bmm_out if output_y3_flag else None
    
    return y1, y2, y3
