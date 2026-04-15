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
        "grouped_mat_mul_all_reduce": "grouped_mat_mul_all_reduce_golden"
    }
}

from typing import List
import numpy as np
import torch

def grouped_mat_mul_all_reduce_golden(
    x,
    weight,
    bias=None,
    group_list=None,
    splitItem: int = 0,
    group: str = "",
    reduceOp: str = "sum",
    commTurn: int = 0,
    **kwargs
):
    """
    GroupedMatMulAllReduce golden function
    
    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        group_list: Optional group list for grouping
        splitItem: Split item flag (default: 0)
        group: HCCL group name (required)
        reduceOp: Reduce operation type (default: "sum")
        commTurn: Communication turn parameter (default: 0)
        **kwargs: Additional arguments including world_size, rank for distributed
    """
    world_size = kwargs.get('world_size', 1)
    
    res_list = []
    
    x_dtype = x.dtype if isinstance(x, torch.Tensor) else torch.from_numpy(x).dtype
    weight_dtype = weight.dtype if isinstance(weight, torch.Tensor) else torch.from_numpy(weight).dtype
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)
    
    x = x.to(torch.float32)
    
    if group_list is None:
        x_list = [x]
        weight_list = [weight]
        bias_list = [bias] if bias is not None else None
    else:
        if not isinstance(group_list, np.ndarray):
            group_list = np.array(group_list)
        if splitItem in [1, 3]:
            x_list = []
            offset = 0
            for i in group_list:
                x_list.append(x[offset:i, :])
                offset = i
        else:
            x_list = [x]
        
        weight_list = [weight]
        
        if bias is not None:
            if isinstance(bias, np.ndarray):
                bias = torch.from_numpy(bias)
            bias_list = list(bias) if len(bias.shape) > 1 else [bias]
        else:
            bias_list = None
    
    for i in range(len(x_list)):
        x_tmp = x_list[i].to(torch.float32)
        w_tmp = weight_list[i].to(torch.float32)
        
        mm_tmp = torch.matmul(x_tmp, w_tmp)
        
        if bias_list is not None and i < len(bias_list) and bias_list[i] is not None:
            bias_tmp = bias_list[i]
            if isinstance(bias_tmp, np.ndarray):
                bias_tmp = torch.from_numpy(bias_tmp)
            bias_tmp = bias_tmp.to(torch.float32)
            gmm_output = torch.add(mm_tmp, bias_tmp)
        else:
            gmm_output = mm_tmp
        
        gmm_output = gmm_output * world_size
        
        res_list.append(gmm_output)
    
    output = res_list if len(res_list) > 1 else res_list[0]
    
    return output