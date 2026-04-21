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
        "quant_all_reduce": "quant_all_reduce_golden"
    }
}

from typing import List
import numpy as np
import torch

def quant_all_reduce_golden(
    x,
    scales,
    group: str = "",
    reduce_op: str = "sum",
    output_dtype: int = 1,
    world_size: int = 1,
    **kwargs
):
    """
    QuantAllReduce golden function
    
    Args:
        x: Input tensor (quantized)
        scales: Scale tensor for dequantization
        group: HCCL group name (communication domain)
        reduce_op: Reduce operation type ("sum", "max", "min")
        output_dtype: Output data type enum
        world_size: Number of ranks in the communication group
        **kwargs: Additional parameters
        
    Returns:
        Dequantized and reduced output tensor
    """
    is_mxfp = kwargs.get('is_mxfp', False)
    group_size = 32 if is_mxfp else 128
    
    x_fp32 = x.astype(np.float32)
    scale_fp32 = scales.astype(np.float32)
    
    output = cpu_dequant(x_fp32, scale_fp32, group_size)
    
    if world_size > 1:
        output = torch.from_numpy(output)
        reduce_op_map = {
            "sum": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }
        op = reduce_op_map.get(reduce_op, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(output, op)
        output = output.numpy()
    
    return output

def cpu_dequant(x, scale, group_size):
    """
    Dequantize input tensor with scale
    
    Args:
        x: Quantized input tensor
        scale: Scale tensor for dequantization
        group_size: Group size for scale broadcasting
        
    Returns:
        Dequantized tensor
    """
    repeated_scale = np.repeat(scale, group_size, axis=-1)
    x_tensor = torch.from_numpy(x)
    repeated_scale_tensor = torch.from_numpy(repeated_scale)
    return (x_tensor * repeated_scale_tensor).numpy()