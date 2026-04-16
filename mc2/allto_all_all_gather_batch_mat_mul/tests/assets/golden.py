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


def _activate_by_type(act_type: int, x: torch.Tensor) -> torch.Tensor:
    if act_type == 0:
        return x
    if act_type == 1:
        return torch.nn.functional.gelu(x, approximate="tanh")
    if act_type == 2:
        return torch.nn.functional.silu(x)
    if act_type == 3:
        return torch.nn.functional.relu(x)
    if act_type == 4:
        # 与 MC2_test CPU 路径一致的 FastGeLu 近似
        return x / (1 + torch.exp(-1.702 * x))
    raise ValueError("act_type only supports 0/1/2/3/4.")


def allto_all_all_gather_batch_mat_mul_golden(x, weight, bias = None, group_ep = None, group_tp = None, ep_world_size = None, tp_world_size = None, x_shard_type = 0, act_type = 0, transpose_weight = False, output_y2_flag = False, output_y3_flag = False, **kwargs):
    if x_shard_type not in (0, 1):
        raise ValueError("x_shard_type only supports 0(H shard) or 1(C shard).")
    if ep_world_size <= 0 or tp_world_size <= 0:
        raise ValueError("ep_world_size and tp_world_size must be positive.")

    x_t = torch.as_tensor(x).to(torch.float32)
    w_t = torch.as_tensor(weight).to(torch.float32)
    b_t = None if bias is None else torch.as_tensor(bias).to(torch.float32)

    if x_t.dim() != 3 or w_t.dim() != 3:
        raise ValueError("x and weight must be 3-D tensors.")
    if x_t.shape[0] % ep_world_size != 0:
        raise ValueError("x.shape[0] must be divisible by ep_world_size.")

    if transpose_weight:
        w_t = w_t.permute(0, 2, 1).contiguous()

    e, c_or_ctp, h_or_htp = x_t.shape
    e_div_ep = e // ep_world_size

    # 1) all_to_all
    all_to_all_out = torch.zeros_like(x_t)
    if dist.is_available() and dist.is_initialized():
        dist.all_to_all_single(all_to_all_out, x_t, group=group_ep)
    else:
        all_to_all_out = x_t

    # 2) reshape + transpose, 与 MC2_test/op_class/aclnnAlltoAllAllGatherBatchMatMul.py 对齐
    reshape_1_shape = [ep_world_size, e_div_ep, c_or_ctp, h_or_htp]
    all_to_all_out = all_to_all_out.reshape(reshape_1_shape).permute(1, 0, 2, 3).contiguous()

    # 3) all_gather over tp group
    all_gather_list = [torch.zeros_like(all_to_all_out) for _ in range(tp_world_size)]
    if dist.is_available() and dist.is_initialized():
        dist.all_gather(all_gather_list, all_to_all_out, group=group_tp)
    else:
        all_gather_list = [all_to_all_out for _ in range(tp_world_size)]
    all_gather_out = torch.cat(all_gather_list, dim=0)

    # 4) reshape/permute/rehsape to bmm-left
    reshape_2_shape = [tp_world_size, e_div_ep, ep_world_size, c_or_ctp, h_or_htp]
    all_gather_out = all_gather_out.reshape(reshape_2_shape)
    if x_shard_type == 0:
        # (E/ep, ep, C, tp, H/tp)
        all_gather_out = all_gather_out.permute(1, 2, 3, 0, 4)
    else:
        # (E/ep, ep, tp, C/tp, H)
        all_gather_out = all_gather_out.permute(1, 2, 0, 3, 4)
    gather_out = all_gather_out.reshape(e_div_ep, -1, all_gather_out.shape[-1])

    # 5) bmm + bias + act
    y3 = torch.bmm(gather_out, w_t)
    if b_t is not None:
        if b_t.dim() == 2:
            b_t = b_t.reshape(b_t.shape[0], 1, b_t.shape[1])
        y3 = y3 + b_t
    y1 = _activate_by_type(act_type, y3)

    return y1, (gather_out if output_y2_flag else None), (y3 if output_y3_flag else None)