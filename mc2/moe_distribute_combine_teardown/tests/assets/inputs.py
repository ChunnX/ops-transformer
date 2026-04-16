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
__input__ = {
        "kernel": {
            "moe_distribute_combine_teardown": "moe_distribute_combine_teardown_inputs"
        }
}

import torch

def moe_distribute_combine_teardown_inputs(expand_x, quant_expand_x, expert_ids, expand_idx, expert_scales, comm_cmd_info, x_active_mask = None, shared_expert_x = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, global_bs = 0, comm_quant_mode = 0, comm_type = 0, comm_alg = '', **kwargs):
    if not isinstance(group_ep, str):
        raise TypeError("group_ep must be a string.")
    if not (0 < len(group_ep) < 128):
        raise ValueError("group_ep length must be in [1, 128).")
    if not isinstance(comm_alg, str):
        raise TypeError("comm_alg must be a string.")
    if comm_alg != "":
        raise ValueError("comm_alg is reserved and must be empty string.")

    int_fields = {
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "expert_shard_type": expert_shard_type,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "global_bs": global_bs,
        "comm_quant_mode": comm_quant_mode,
        "comm_type": comm_type,
    }
    for name, value in int_fields.items():
        if int(value) != value:
            raise ValueError(f"{name} must be an integer.")

    ep_ws = int(ep_world_size)
    ep_rank = int(ep_rank_id)
    moe_n = int(moe_expert_num)
    shard_type = int(expert_shard_type)
    shared_n = int(shared_expert_num)
    shared_rank_n = int(shared_expert_rank_num)
    gbs = int(global_bs)
    qmode = int(comm_quant_mode)
    ctype = int(comm_type)

    if ep_ws <= 0:
        raise ValueError("ep_world_size must be positive.")
    if not (0 <= ep_rank < ep_ws):
        raise ValueError("ep_rank_id must be in [0, ep_world_size).")
    if not (0 < moe_n <= 512):
        raise ValueError("moe_expert_num must be in (0, 512].")
    if shard_type != 0:
        raise ValueError("expert_shard_type currently only supports 0.")
    if shared_n <= 0:
        raise ValueError("shared_expert_num must be positive.")
    if not (0 <= shared_rank_n < ep_ws):
        raise ValueError("shared_expert_rank_num must be in [0, ep_world_size).")
    if shared_rank_n != 0 and ep_ws % shared_rank_n != 0:
        raise ValueError("ep_world_size must be divisible by shared_expert_rank_num when non-zero.")
    if gbs < 0:
        raise ValueError("global_bs must be >= 0.")
    if qmode not in (0, 2):
        raise ValueError("comm_quant_mode currently only supports 0 or 2.")
    if ctype != 0:
        raise ValueError("comm_type currently only supports 0.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")

    x_t = torch.as_tensor(expand_x)
    ids_t = torch.as_tensor(expert_ids)
    idx_t = torch.as_tensor(expand_idx)
    scales_t = torch.as_tensor(expert_scales)
    cmd_t = torch.as_tensor(comm_cmd_info)

    if x_t.dim() != 2:
        raise ValueError("expand_x must be 2-D.")
    if ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D.")
    if idx_t.dim() != 1:
        raise ValueError("expand_idx must be 1-D.")
    if scales_t.dim() != 2:
        raise ValueError("expert_scales must be 2-D.")
    if cmd_t.dim() != 1:
        raise ValueError("comm_cmd_info must be 1-D.")
    if tuple(scales_t.shape) != tuple(ids_t.shape):
        raise ValueError("expert_scales shape must equal expert_ids shape.")

    bs, k = int(ids_t.shape[0]), int(ids_t.shape[1])
    if not (1 <= bs <= 512):
        raise ValueError("Bs must be in [1, 512].")
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > moe_n:
        raise ValueError("K must be <= moe_expert_num.")

    if x_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expand_x dtype must be float16 or bfloat16.")
    if ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")
    if idx_t.dtype != torch.int32:
        raise ValueError("expand_idx dtype must be int32.")
    if scales_t.dtype != torch.float32:
        raise ValueError("expert_scales dtype must be float32.")
    if cmd_t.dtype != torch.int32:
        raise ValueError("comm_cmd_info dtype must be int32.")

    if quant_expand_x is not None and torch.as_tensor(quant_expand_x).numel() > 0:
        qx_t = torch.as_tensor(quant_expand_x)
        if qmode == 0:
            raise ValueError("quant_expand_x must be nullptr/empty when comm_quant_mode=0.")
        if qx_t.dim() != 2 or tuple(qx_t.shape) != tuple(x_t.shape):
            raise ValueError("quant_expand_x must be 2-D and have the same shape as expand_x when provided.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        mask_t = torch.as_tensor(x_active_mask)
        if mask_t.dtype != torch.bool:
            raise ValueError("x_active_mask must be bool when provided.")
        if mask_t.dim() == 1:
            if int(mask_t.shape[0]) != bs:
                raise ValueError("1-D x_active_mask shape must be (Bs,).")
        elif mask_t.dim() == 2:
            if tuple(mask_t.shape) != (bs, k):
                raise ValueError("2-D x_active_mask shape must be (Bs, K).")
        else:
            raise ValueError("x_active_mask must be 1-D or 2-D when provided.")

    if shared_expert_x is not None and torch.as_tensor(shared_expert_x).numel() > 0:
        sx_t = torch.as_tensor(shared_expert_x)
        if sx_t.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("shared_expert_x dtype must be float16 or bfloat16 when provided.")
        h = int(x_t.shape[1])
        if sx_t.dim() == 2:
            if tuple(sx_t.shape) != (bs, h):
                raise ValueError("2-D shared_expert_x shape must be (Bs, H).")
        elif sx_t.dim() == 3:
            if tuple(sx_t.shape) != (bs, 1, h):
                raise ValueError("3-D shared_expert_x shape must be (Bs, 1, H).")
        else:
            raise ValueError("shared_expert_x must be 2-D or 3-D when provided.")

    return (expand_x,quant_expand_x,expert_ids,expand_idx,expert_scales,comm_cmd_info,
            x_active_mask,shared_expert_x,group_ep,ep_world_size,ep_rank_id,
            moe_expert_num,expert_shard_type,shared_expert_num,shared_expert_rank_num,
            global_bs,comm_quant_mode,comm_type,comm_alg,kwargs)