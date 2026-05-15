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
        "grouped_mat_mul_allto_allv": "grouped_mat_mul_allto_allv_golden"
    }
}

from typing import List
import numpy as np
import torch
import torch.distributed as dist


def grouped_mat_mul_allto_allv_golden(
    gmm_x,
    gmm_weight,
    send_counts_tensor=None,
    recv_counts_tensor=None,
    mm_x=None,
    mm_weight=None,
    group: str = "",
    ep_world_size: int = 1,
    send_counts: List[int] = None,
    recv_counts: List[int] = None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    **kwargs
):
    """
    Golden calculation for GroupedMatMulAlltoAllv operator
    
    Parameters must match _def.cpp exactly:
    - Inputs: gmm_x, gmm_weight, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight
    - Attrs: group, ep_world_size, send_counts, recv_counts, trans_gmm_weight, trans_mm_weight
    """
    input_splits = kwargs.get('input_splits', None)
    output_splits = kwargs.get('output_splits', None)
    expert_token_num = kwargs.get('expert_token_num', None)
    exp_per_card = kwargs.get('exp_per_card', None)
    
    input_dtype = kwargs.get('input_dtype', 'fp16')
    
    if send_counts is None:
        send_counts = []
    if recv_counts is None:
        recv_counts = []
    if input_splits is None:
        input_splits = []
    if output_splits is None:
        output_splits = []
    
    tmp_dtype = torch.float16 if input_dtype == "fp16" else torch.bfloat16
    
    if trans_gmm_weight:
        gmm_weight = torch.transpose(gmm_weight, 1, 2)
    if trans_mm_weight and mm_weight is not None:
        mm_weight = torch.transpose(mm_weight, 0, 1)
    
    group_list = _get_group_list_from_expert_token(expert_token_num, ep_world_size, exp_per_card) if expert_token_num is not None else None
    
    if group_list is not None:
        gmm_out = _grouped_matmul_cpu(gmm_x, gmm_weight, group_list.tolist() if isinstance(group_list, torch.Tensor) else group_list)
    else:
        gmm_out = torch.matmul(gmm_x, gmm_weight)
    
    unpermute_out = _unpermute(gmm_out, ep_world_size, exp_per_card, expert_token_num)
    
    output = torch.empty((sum(output_splits), gmm_out.shape[-1]), dtype=tmp_dtype)
    dist.all_to_all_single(output, input=unpermute_out.to(tmp_dtype), output_split_sizes=output_splits, input_split_sizes=input_splits)
    
    if input_dtype == "fp16":
        output = output.to(torch.float16)
    elif input_dtype == "bf16":
        output = output.to(torch.bfloat16)
    
    mm_out = None
    if mm_x is not None and mm_weight is not None:
        mm_out = torch.mm(mm_x.to(torch.float), mm_weight.to(torch.float))
        if input_dtype == "fp16":
            mm_out = mm_out.to(torch.float16)
        elif input_dtype == "bf16":
            mm_out = mm_out.to(torch.bfloat16)
    
    return output, mm_out


def _grouped_matmul_cpu(gmm_x, gmm_weight, group_list):
    """
    Grouped matrix multiplication on CPU
    """
    B_list = list(torch.unbind(gmm_weight, dim=0))
    A_groups = torch.split(gmm_x, group_list, dim=0)
    results = []
    for i in range(len(group_list)):
        a = A_groups[i].numpy()
        b = B_list[i].numpy()
        results.append(torch.from_numpy(np.matmul(a, b)))
    return torch.cat(results, dim=0)


def _unpermute(tokens, ep_world_size, exp_per_card, exp_token_nums, device="cpu"):
    """
    Unpermute function for alltoallv
    """
    if exp_token_nums is None or ep_world_size is None or exp_per_card is None:
        return tokens
    
    empty_arr = np.zeros((ep_world_size, exp_per_card), dtype=np.int64)
    for i in range(ep_world_size):
        for j in range(exp_per_card):
            empty_arr[i][j] = int(exp_token_nums[i][j])
    
    tmp1 = empty_arr.transpose()
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2 = np.cumsum(sum_list1, axis=0)
    offsets = [0]
    for s in sum_list2[:-1]:
        offsets.append(int(s))
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for i in range(exp_per_card):
        tmp = []
        for j in range(ep_world_size):
            if j == 0:
                tmp.append(list(map(lambda x: x + offsets[i], list(range(0, sum_list[i][j])))))
            else:
                tmp.append(list(map(lambda x: x + offsets[i], list(range(sum_list[i][j - 1], sum_list[i][j])))))
        indices_list.append(tmp)
    
    selected = []
    for i in range(ep_world_size):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
            selected_rows = tokens.to(device).index_select(dim=0, index=indices)
            selected.append(selected_rows)
    
    return torch.cat(selected, dim=0).to(tokens.dtype)


def _get_group_list_from_expert_token(expert_token_num, ep_world_size, exp_per_card):
    """
    Generate group_list from expert_token_num
    """
    recv = torch.zeros(ep_world_size, exp_per_card).to(torch.int64)
    for i in range(ep_world_size):
        tmp = expert_token_num[i][:exp_per_card]
        recv[i:] = torch.tensor(tmp)
    return torch.sum(recv, dim=0)