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
        "allto_allv_grouped_mat_mul": "allto_allv_grouped_mat_mul_golden"
    }
}

import torch
import torch.distributed as dist
import numpy as np

def allto_allv_grouped_mat_mul_golden(
    gmm_x,
    gmm_weight,
    send_counts_tensor=None,
    recv_counts_tensor=None,
    mm_x=None,
    mm_weight=None,
    group="",
    ep_world_size: int = 1,
    send_counts=None,
    recv_counts=None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    permute_out_flag: bool = False,
    **kwargs
):
    """
    AlltoAllvGroupedMatMul golden implementation
    
    Args:
        gmm_x: Input tensor for grouped matmul
        gmm_weight: Weight tensor for grouped matmul
        send_counts_tensor: Send counts tensor (optional)
        recv_counts_tensor: Receive counts tensor (optional)
        mm_x: Input tensor for matmul (optional)
        mm_weight: Weight tensor for matmul (optional)
        group: Communication group name
        ep_world_size: World size for EP
        send_counts: Send counts list
        recv_counts: Receive counts list
        trans_gmm_weight: Whether to transpose gmm_weight
        trans_mm_weight: Whether to transpose mm_weight
        permute_out_flag: Whether to output permute result
        **kwargs: Additional arguments
    
    Returns:
        gmm_y: Grouped matmul output
        mm_y: Matmul output (if mm_x and mm_weight are provided)
        permute_out: Permute output (if permute_out_flag is True)
    """
    gmm_x_float = gmm_x.to(torch.float32)
    gmm_weight_float = gmm_weight.to(torch.float32)
    
    if trans_gmm_weight:
        gmm_weight_float = gmm_weight_float.permute(0, 2, 1) if len(gmm_weight_float.shape) == 3 else gmm_weight_float.transpose()
    
    send_counts_list = send_counts if send_counts else []
    recv_counts_list = recv_counts if recv_counts else []
    
    input_splits = []
    for i in range(ep_world_size):
        input_splits.append(sum(send_counts_list[i*len(send_counts_list)//ep_world_size:(i+1)*len(send_counts_list)//ep_world_size]))
    
    output_splits = []
    for i in range(ep_world_size):
        output_splits.append(sum(recv_counts_list[i*len(recv_counts_list)//ep_world_size:(i+1)*len(recv_counts_list)//ep_world_size]))
    
    alltoallv_out = torch.zeros(sum(output_splits), gmm_x_float.shape[1], dtype=gmm_x_float.dtype)
    
    try:
        dist.all_to_all_single(alltoallv_out, gmm_x_float, output_splits=output_splits, input_splits=input_splits)
    except:
        alltoallv_out = gmm_x_float
    
    gmm_y = torch.matmul(alltoallv_out, gmm_weight_float)
    
    mm_y = None
    if mm_x is not None and mm_weight is not None:
        mm_x_float = mm_x.to(torch.float32)
        mm_weight_float = mm_weight.to(torch.float32)
        if trans_mm_weight:
            mm_weight_float = mm_weight_float.transpose()
        mm_y = torch.matmul(mm_x_float, mm_weight_float)
    
    permute_out = alltoallv_out if permute_out_flag else None
    
    return gmm_y, mm_y, permute_out
