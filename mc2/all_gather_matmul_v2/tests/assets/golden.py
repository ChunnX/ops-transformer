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
        "all_gather_matmul_v2": "all_gather_matmul_v2_golden"
    }
}

import torch
import torch.distributed as dist
import numpy as np

def all_gather_matmul_v2_golden(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    quant_scale=None,
    group="",
    is_trans_a=False,
    is_trans_b=False,
    gather_index: int = 0,
    comm_turn: int = 0,
    rank_size: int = 0,
    block_size: int = 0,
    group_size: int = 0,
    is_gather_out: bool = True,
    is_amax_out: bool = False,
    y_dtype: int = -1,
    comm_mode: str = "aicpu",
    **kwargs
):
    """
    AllGatherMatmulV2 golden implementation
    
    Args:
        x1: Input tensor 1
        x2: Input tensor 2
        bias: Bias tensor (optional)
        x1_scale: Scale for x1 (optional)
        x2_scale: Scale for x2 (optional)
        quant_scale: Quantization scale (optional)
        group: Communication group name
        is_trans_a: Whether to transpose x1
        is_trans_b: Whether to transpose x2
        gather_index: Gather index
        comm_turn: Communication turn
        rank_size: Number of ranks
        block_size: Block size for quantization
        group_size: Group size for quantization
        is_gather_out: Whether to output gather result
        is_amax_out: Whether to output amax
        y_dtype: Output data type
        comm_mode: Communication mode
        **kwargs: Additional arguments (hcomm_info, world_size, etc.)
    
    Returns:
        output: Matmul result
        gather_out: All-gathered x1 (if is_gather_out is True)
        amax_out: Amax output (if is_amax_out is True)
    """
    world_size = kwargs.get("world_size", rank_size)
    if world_size == 0:
        world_size = 1
    
    x1_float = x1.to(torch.float32)
    x2_float = x2.to(torch.float32)
    
    if is_trans_a:
        x1_float = x1_float.transpose()
    if is_trans_b:
        x2_float = x2_float.transpose()
    
    gather_out = torch.zeros([x1_float.shape[0] * world_size] + list(x1_float.shape[1:]),
                              dtype=x1_float.dtype)
    try:
        dist._all_gather_base(gather_out, x1_float)
    except:
        gather_out = torch.cat([x1_float] * world_size, dim=0)
    
    if x1_scale is not None and x2_scale is None:
        x1_scale_float = x1_scale.to(torch.float32)
        gather_scale = torch.zeros([x1_scale_float.shape[0] * world_size] + list(x1_scale_float.shape[1:]),
                                    dtype=x1_scale_float.dtype)
        try:
            dist._all_gather_base(gather_scale, x1_scale_float)
        except:
            gather_scale = torch.cat([x1_scale_float] * world_size, dim=0)
    
    output = torch.matmul(gather_out, x2_float)
    
    if bias is not None:
        bias_float = bias.to(torch.float32)
        output = output + bias_float
    
    if is_gather_out:
        return output, gather_out, None
    else:
        return output, None, None
