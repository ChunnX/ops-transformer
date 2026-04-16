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
            "moe_distribute_combine_add_rms_norm": "moe_distribute_combine_add_rms_norm_inputs"
        }
}

import torch

def moe_distribute_combine_add_rms_norm_inputs(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, residual_x, gamma, tp_send_counts = None, x_active_mask = None, activation_scale = None, weight_scale = None, group_list = None, expand_scales = None, shared_expert_x = None, elastic_info = None, ori_x = None, const_expert_alpha_1 = None, const_expert_alpha_2 = None, const_expert_v = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, group_tp = '', tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 0, shared_expert_rank_num = 0, global_bs = 0, out_dtype = 0, comm_quant_mode = 0, group_list_type = 0, comm_alg = '', norm_eps = 1e-6, zero_expert_num = 0, copy_expert_num = 0, const_expert_num = 0, **kwargs):

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
    if comm_alg != "":
        raise ValueError("comm_alg is reserved and must be empty string.")
    if float(norm_eps) != 1e-6:
        raise ValueError("norm_eps currently only supports 1e-6.")

    int_fields = {
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
        "zero_expert_num": zero_expert_num,
        "copy_expert_num": copy_expert_num,
        "const_expert_num": const_expert_num,
    }
    for name, value in int_fields.items():
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
    zero_n = int(zero_expert_num)
    copy_n = int(copy_expert_num)
    const_n = int(const_expert_num)

    if ep_ws <= 0:
        raise ValueError("ep_world_size must be positive.")
    if not (0 <= ep_rank < ep_ws):
        raise ValueError("ep_rank_id must be in [0, ep_world_size).")
    if moe_n <= 0:
        raise ValueError("moe_expert_num must be positive.")
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

    if shard_type != 0:
        raise ValueError("expert_shard_type currently only supports 0.")
    if shared_n != 0 or shared_rank_n != 0:
        raise ValueError("shared_expert_num/shared_expert_rank_num are currently unsupported and must be 0.")
    if gbs < 0:
        raise ValueError("global_bs must be >= 0.")
    if out_dt != 0:
        raise ValueError("out_dtype currently only supports 0.")
    if qmode != 0:
        raise ValueError("comm_quant_mode currently only supports 0.")
    if gl_type != 0:
        raise ValueError("group_list_type currently only supports 0.")
    if zero_n != 0 or copy_n != 0 or const_n != 0:
        raise ValueError("zero_expert_num/copy_expert_num/const_expert_num are currently unsupported and must be 0.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a3", "atlas_a3"}:
        valid_ep_ws = {8, 16, 32, 64, 128, 144, 256, 288}
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        valid_ep_ws = {2, 4, 8, 16, 32, 64, 128, 144, 256, 288}
    else:
        valid_ep_ws = {2, 4, 8, 16, 32, 64, 128, 144, 256, 288}
    if ep_ws not in valid_ep_ws:
        raise ValueError(f"ep_world_size={ep_ws} is not supported for soc={soc or 'default'}; expected {sorted(valid_ep_ws)}.")

    expand_x_t = torch.as_tensor(expand_x)
    expert_ids_t = torch.as_tensor(expert_ids)
    assist_t = torch.as_tensor(assist_info_for_combine)
    ep_send_t = torch.as_tensor(ep_send_counts)
    expert_scales_t = torch.as_tensor(expert_scales)
    residual_t = torch.as_tensor(residual_x)
    gamma_t = torch.as_tensor(gamma)

    if expand_x_t.dim() != 2:
        raise ValueError("expand_x must be 2-D.")
    if expert_ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D.")
    if assist_t.dim() != 1:
        raise ValueError("assist_info_for_combine must be 1-D.")
    if ep_send_t.dim() != 1:
        raise ValueError("ep_send_counts must be 1-D.")
    if expert_scales_t.dim() != 2:
        raise ValueError("expert_scales must be 2-D.")
    if residual_t.dim() != 3:
        raise ValueError("residual_x must be 3-D with shape (Bs, 1, H).")
    if gamma_t.dim() != 1:
        raise ValueError("gamma must be 1-D with shape (H,).")

    if tuple(expert_ids_t.shape) != tuple(expert_scales_t.shape):
        raise ValueError("expert_scales shape must equal expert_ids shape.")

    bs, k = int(expert_ids_t.shape[0]), int(expert_ids_t.shape[1])
    a_mul_tp, h = int(expand_x_t.shape[0]), int(expand_x_t.shape[1])
    if int(residual_t.shape[0]) != bs or int(residual_t.shape[1]) != 1 or int(residual_t.shape[2]) != h:
        raise ValueError("residual_x shape must be (Bs, 1, H).")
    if int(gamma_t.shape[0]) != h:
        raise ValueError("gamma shape must be (H,).")

    if not (1024 <= h <= 8192):
        raise ValueError("H must be in [1024, 8192].")
    if not (1 <= bs <= 512):
        raise ValueError("Bs must be in [1, 512].")
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > moe_n:
        raise ValueError("K must be <= moe_expert_num.")

    tp_mul = max(tp_ws, 1)
    if a_mul_tp % tp_mul != 0:
        raise ValueError("expand_x first dim must be divisible by max(tp_world_size, 1).")
    a = a_mul_tp // tp_mul
    if int(assist_t.shape[0]) != a * 128:
        raise ValueError("assist_info_for_combine shape must be (A * 128,).")

    local_expert_num = moe_n // local_expert_den
    expect_ep_cnt = ep_ws * tp_mul * local_expert_num
    if int(ep_send_t.shape[0]) != expect_ep_cnt:
        raise ValueError(f"ep_send_counts shape must be ({expect_ep_cnt},).")
    if local_expert_num > 1 and tp_ws > 1:
        raise ValueError("TP communication is not supported when localExpertNum > 1.")

    if expand_x_t.dtype != torch.bfloat16:
        raise ValueError("expand_x dtype must be bfloat16.")
    if expert_ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")
    if assist_t.dtype != torch.int32:
        raise ValueError("assist_info_for_combine dtype must be int32.")
    if ep_send_t.dtype != torch.int32:
        raise ValueError("ep_send_counts dtype must be int32.")
    if expert_scales_t.dtype != torch.float32:
        raise ValueError("expert_scales dtype must be float32.")
    if residual_t.dtype != torch.bfloat16:
        raise ValueError("residual_x dtype must be bfloat16.")
    if gamma_t.dtype != torch.bfloat16:
        raise ValueError("gamma dtype must be bfloat16.")

    if tp_send_counts is not None and torch.as_tensor(tp_send_counts).numel() > 0:
        tp_send_t = torch.as_tensor(tp_send_counts)
        if tp_ws <= 1:
            raise ValueError("tp_send_counts must be empty when there is no TP communication.")
        if tp_send_t.dim() != 1 or int(tp_send_t.shape[0]) != tp_ws:
            raise ValueError("tp_send_counts must be 1-D with length tp_world_size.")
        if tp_send_t.dtype != torch.int32:
            raise ValueError("tp_send_counts dtype must be int32.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        xmask_t = torch.as_tensor(x_active_mask)
        if xmask_t.dtype != torch.bool:
            raise ValueError("x_active_mask must be bool when provided.")
        if xmask_t.dim() == 1:
            if int(xmask_t.shape[0]) != bs:
                raise ValueError("1-D x_active_mask shape must be (Bs,).")
        elif xmask_t.dim() == 2:
            if tuple(xmask_t.shape) != (bs, k):
                raise ValueError("2-D x_active_mask shape must be (Bs, K).")
        else:
            raise ValueError("x_active_mask must be 1-D or 2-D when provided.")

    if activation_scale is not None and torch.as_tensor(activation_scale).numel() > 0:
        raise ValueError("activation_scale is reserved and must be nullptr/empty.")
    if weight_scale is not None and torch.as_tensor(weight_scale).numel() > 0:
        raise ValueError("weight_scale is reserved and must be nullptr/empty.")
    if group_list is not None and torch.as_tensor(group_list).numel() > 0:
        raise ValueError("group_list is reserved and must be nullptr/empty.")
    if expand_scales is not None and torch.as_tensor(expand_scales).numel() > 0:
        raise ValueError("expand_scales is reserved and must be nullptr/empty.")
    if elastic_info is not None and torch.as_tensor(elastic_info).numel() > 0:
        raise ValueError("elastic_info is reserved and must be nullptr/empty.")
    if ori_x is not None and torch.as_tensor(ori_x).numel() > 0:
        raise ValueError("ori_x is reserved and must be nullptr/empty.")
    if const_expert_alpha_1 is not None and torch.as_tensor(const_expert_alpha_1).numel() > 0:
        raise ValueError("const_expert_alpha_1 is reserved and must be nullptr/empty.")
    if const_expert_alpha_2 is not None and torch.as_tensor(const_expert_alpha_2).numel() > 0:
        raise ValueError("const_expert_alpha_2 is reserved and must be nullptr/empty.")
    if const_expert_v is not None and torch.as_tensor(const_expert_v).numel() > 0:
        raise ValueError("const_expert_v is reserved and must be nullptr/empty.")

    if shared_expert_x is not None and torch.as_tensor(shared_expert_x).numel() > 0:
        sx_t = torch.as_tensor(shared_expert_x)
        if sx_t.dtype != torch.bfloat16:
            raise ValueError("shared_expert_x dtype must be bfloat16 when provided.")
        if sx_t.dim() == 2:
            if tuple(sx_t.shape) != (bs, h):
                raise ValueError("2-D shared_expert_x shape must be (Bs, H).")
        elif sx_t.dim() == 3:
            if tuple(sx_t.shape) != (bs, 1, h):
                raise ValueError("3-D shared_expert_x shape must be (Bs, 1, H).")
        else:
            raise ValueError("shared_expert_x must be 2-D or 3-D when provided.")

    return (expand_x,expert_ids,assist_info_for_combine,ep_send_counts,expert_scales,
                                  residual_x,gamma,tp_send_counts,x_active_mask,activation_scale,weight_scale,
                                  group_list,expand_scales,shared_expert_x,elastic_info,ori_x,const_expert_alpha_1,
                                  const_expert_alpha_2,const_expert_v,group_ep,ep_world_size,ep_rank_id,
                                  moe_expert_num,group_tp,tp_world_size,tp_rank_id,expert_shard_type,
                                  shared_expert_num,shared_expert_rank_num,global_bs,out_dtype,
                                  comm_quant_mode,group_list_type,comm_alg,norm_eps,
                                  zero_expert_num,copy_expert_num,const_expert_num,kwargs)