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
        "quant_reduce_scatter": "quant_reduce_scatter_golden"
    }
}

import numpy as np
import torch

def quant_reduce_scatter_golden(
    x,
    scales,
    group: str,
    reduce_op: str = "sum",
    output_dtype: int = 1,
    world_size: int = 1,
    **kwargs
):
    """
    QuantReduceScatter golden function
    
    Args:
        x: quantized input tensor
        scales: dequantization scales
        group: communication group (HCCL group)
        reduce_op: reduce operation type ("sum", "max", "min")
        output_dtype: output data type (0: fp16, 1: bf16, 2: fp32)
        world_size: number of ranks in the group
        **kwargs: additional parameters (rank, is_mxFp, etc.)
    
    Returns:
        Dequantized and reduced output tensor
    """
    rank = kwargs.get('rank', 0)
    is_mxFp = kwargs.get('is_mxFp', False)
    
    x_tensor = torch.from_numpy(x.astype(np.float32))
    scale_tensor = torch.from_numpy(scales.astype(np.float32))
    
    group_size = 32 if is_mxFp else 128
    
    repeated_scale = torch.repeat_interleave(scale_tensor, group_size, dim=-1)
    
    dequant_output = x_tensor * repeated_scale
    
    dequant_output = dequant_output.reshape(world_size, -1, dequant_output.shape[-1])
    
    if reduce_op == "sum":
        reduced_output = torch.sum(dequant_output, dim=0)
    elif reduce_op == "max":
        reduced_output = torch.amax(dequant_output, dim=0)
    elif reduce_op == "min":
        reduced_output = torch.amin(dequant_output, dim=0)
    else:
        reduced_output = torch.sum(dequant_output, dim=0)
    
    output_dtype_map = {
        0: torch.float16,
        1: torch.bfloat16,
        2: torch.float32
    }
    target_dtype = output_dtype_map.get(output_dtype, torch.bfloat16)
    reduced_output = reduced_output.to(target_dtype)
    
    scatter_shape_m = x.shape[0] // world_size
    scatter_output = reduced_output[rank * scatter_shape_m:(rank + 1) * scatter_shape_m, :]
    
    return scatter_output.numpy()