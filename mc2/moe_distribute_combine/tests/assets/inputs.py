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
            "moe_distribute_combine": "moe_distribute_combine_inputs"
        }
}

import torch

def moe_distribute_combine_inputs(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, tp_send_counts = None, x_active_mask = None, activation_scale = None, weight_scale = None, group_list = None, expand_scales = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, group_tp = '', tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, global_bs = 0, out_dtype = 0, comm_quant_mode = 0, group_list_type = 0, **kwargs):
    if not isinstance(group_ep, str):
        raise TypeError("group_ep must be a string.")
    if not (0 < len(group_ep) < 128):
        raise ValueError("group_ep length must be in [1, 128).")
    if not isinstance(group_tp, str):
        raise TypeError("group_tp must be a string.")
    if group_tp and not (0 < len(group_tp) < 128):
        raise ValueError("group_tp length must be in [1, 128) when provided.")
    if group_tp and group_tp == group_ep:
        raise ValueError("group_tp must be different from group_ep.")

    for name, value in {
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "tp_world_size": tp_world_size,
        "tp_rank_id": tp_rank_id,
        "expert_shard_type": expert_shard_type,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "global_bs": global_bs,
        "out_dtype": out_dtype,
        "comm_quant_mode": comm_quant_mode,
        "group_list_type": group_list_type,
    }.items():
        if int(value) != value:
            raise ValueError(f"{name} must be an integer.")

    ep_ws = int(ep_world_size)
    ep_rank = int(ep_rank_id)
    moe_n = int(moe_expert_num)
    tp_ws = int(tp_world_size)
    tp_rank = int(tp_rank_id)
    shard_type = int(expert_shard_type)
    shared_n = int(shared_expert_num)
    shared_rank_n = int(shared_expert_rank_num)
    gbs = int(global_bs)
    out_dt = int(out_dtype)
    qmode = int(comm_quant_mode)
    gl_type = int(group_list_type)

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
    if out_dt != 0:
        raise ValueError("out_dtype currently only supports 0.")
    if gl_type != 0:
        raise ValueError("group_list_type currently only supports 0.")
    if qmode not in (0, 2):
        raise ValueError("comm_quant_mode currently only supports 0 or 2.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2"}:
        valid_ep_ws = {16, 32, 64}
    elif soc in {"a3", "atlas_a3"}:
        valid_ep_ws = {8, 16, 32, 64, 128, 144, 256, 288}
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        valid_ep_ws = {2, 4, 8, 16, 32, 64, 128, 144, 256, 288}
    else:
        valid_ep_ws = {2, 4, 8, 16, 32, 64, 128, 144, 256, 288}
    if ep_ws not in valid_ep_ws:
        raise ValueError(f"ep_world_size={ep_ws} is not supported for soc={soc or 'default'}; expected {sorted(valid_ep_ws)}.")

    # TP 域约束：A2/950 不支持 TP；A3 仅支持 tp_world_size in [0,2] 且有 TP 时取 2。
    if soc in {"a2", "atlas_a2", "950", "ascend950", "950pr", "950dt"}:
        if group_tp != "" or tp_ws != 0 or tp_rank != 0:
            raise ValueError("current soc does not support TP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
    else:
        if not (0 <= tp_ws <= 2):
            raise ValueError("tp_world_size must be in [0, 2].")
        if tp_ws in (0, 1):
            if tp_rank != 0:
                raise ValueError("tp_rank_id must be 0 when tp_world_size is 0 or 1.")
            if group_tp != "":
                raise ValueError("group_tp must be empty when there is no TP communication.")
        else:
            if not (0 <= tp_rank < tp_ws):
                raise ValueError("tp_rank_id must be in [0, tp_world_size).")
            if group_tp == "":
                raise ValueError("group_tp must be non-empty when tp_world_size=2.")

    expand_x_t = torch.as_tensor(expand_x)
    expert_ids_t = torch.as_tensor(expert_ids)
    expert_scales_t = torch.as_tensor(expert_scales)
    expand_idx_t = torch.as_tensor(expand_idx)
    ep_send_counts_t = torch.as_tensor(ep_send_counts)

    if expand_x_t.dim() != 2:
        raise ValueError("expand_x must be 2-D.")
    if expert_ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D.")
    if expert_scales_t.dim() != 2:
        raise ValueError("expert_scales must be 2-D.")
    if tuple(expert_scales_t.shape) != tuple(expert_ids_t.shape):
        raise ValueError("expert_scales shape must equal expert_ids shape.")
    if int(expert_ids_t.shape[0]) != int(expand_x_t.shape[0]):
        raise ValueError("expert_ids first dim must equal expand_x first dim.")
    if expand_idx_t.dim() != 1:
        raise ValueError("expand_idx must be 1-D.")
    if ep_send_counts_t.dim() != 1:
        raise ValueError("ep_send_counts must be 1-D.")
    if int(ep_send_counts_t.shape[0]) != ep_ws:
        raise ValueError("ep_send_counts length must equal ep_world_size.")

    if expand_x_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expand_x dtype must be float16 or bfloat16.")
    if expert_ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")
    if expand_idx_t.dtype != torch.int32:
        raise ValueError("expand_idx dtype must be int32.")
    if ep_send_counts_t.dtype != torch.int32:
        raise ValueError("ep_send_counts dtype must be int32.")
    if expert_scales_t.dtype != torch.float32:
        raise ValueError("expert_scales dtype must be float32.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        raise ValueError("x_active_mask is reserved and must be nullptr/empty.")
    if activation_scale is not None and torch.as_tensor(activation_scale).numel() > 0:
        raise ValueError("activation_scale is reserved and must be nullptr/empty.")
    if weight_scale is not None and torch.as_tensor(weight_scale).numel() > 0:
        raise ValueError("weight_scale is reserved and must be nullptr/empty.")
    if group_list is not None and torch.as_tensor(group_list).numel() > 0:
        raise ValueError("group_list is reserved and must be nullptr/empty.")

    if soc in {"a3", "atlas_a3", "950", "ascend950", "950pr", "950dt"}:
        if expand_scales is not None and torch.as_tensor(expand_scales).numel() > 0:
            raise ValueError("expand_scales is not supported on A3/950.")

    if tp_send_counts is not None and torch.as_tensor(tp_send_counts).numel() > 0:
        tp_counts_t = torch.as_tensor(tp_send_counts)
        if tp_ws <= 1:
            raise ValueError("tp_send_counts must be empty when there is no TP communication.")
        if tp_counts_t.dim() != 1 or int(tp_counts_t.shape[0]) != tp_ws:
            raise ValueError("tp_send_counts must be 1-D with length tp_world_size.")
        if tp_counts_t.dtype != torch.int32:
            raise ValueError("tp_send_counts dtype must be int32 when provided.")

    return (expand_x,expert_ids,expand_idx,ep_send_counts,expert_scales,
            tp_send_counts,x_active_mask,activation_scale,weight_scale,
            group_list,expand_scales,group_ep,ep_world_size,ep_rank_id,
            moe_expert_num,group_tp,tp_world_size,tp_rank_id,
            expert_shard_type,shared_expert_num,shared_expert_rank_num,
            global_bs,out_dtype,comm_quant_mode,group_list_type,kwargs)