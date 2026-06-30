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
        "matmul_reduce_scatter": "matmul_reduce_scatter_golden"
    }
}

import numpy as np
import torch
from torch.distributed import ReduceOp

def matmul_reduce_scatter_golden(
    x1,
    x2,
    bias=None,
    group: str = "",
    reduce_op: str = "sum",
    is_trans_a: bool = False,
    is_trans_b: bool = False,
    comm_turn: int = 0,
    rank_size: int = 0,
    **kwargs
):
    """
    Golden 计算函数 for matmul_reduce_scatter
    
    参数名称与 matmul_reduce_scatter_def.cpp 完全一致
    
    计算逻辑:
    1. 处理转置
    2. 执行 matmul
    3. 执行 reduce_scatter
    """
    if is_trans_a:
        x1 = x1.t()
    if is_trans_b:
        x2 = x2.t()
    
    if bias is not None:
        output = torch.matmul(x1, x2) + bias
    else:
        output = torch.matmul(x1, x2)
    
    tensor_scatter = kwargs.get('tensor_scatter', None)
    if tensor_scatter is None:
        output_shape = [x1.shape[0] // rank_size, x2.shape[0] if is_trans_b else x2.shape[1]]
        tensor_scatter = torch.zeros(output_shape, dtype=x1.dtype)
    
    dist = kwargs.get('dist', None)
    if dist is not None:
        dist._reduce_scatter_base(tensor_scatter, output, op=ReduceOp.SUM)
        output = tensor_scatter
    
    return output