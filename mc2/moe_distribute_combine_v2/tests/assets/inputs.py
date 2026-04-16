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
            "moe_distribute_combine_v2": "moe_distribute_combine_v2_inputs"
        }
}

import torch

def moe_distribute_combine_v2_inputs(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, tp_send_counts = None, x_active_mask = None, activation_scale = None, weight_scale = None, group_list = None, expand_scales = None, shared_expert_x = None, elastic_info = None, ori_x = None, const_expert_alpha_1 = None, const_expert_alpha_2 = None, const_expert_v = None, performance_info = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, group_tp = '', tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, global_bs = 0, out_dtype = 0, comm_quant_mode = 0, group_list_type = 0, comm_alg = '', zero_expert_num = 0, copy_expert_num = 0, const_expert_num = 0, **kwargs):
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
    if not isinstance(comm_alg, str):
        raise TypeError("comm_alg must be a string.")

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
    if shard_type != 0:
        raise ValueError("expert_shard_type currently only supports 0.")
    if not (0 <= shared_n <= 4):
        raise ValueError("shared_expert_num must be in [0, 4].")
    if not (0 <= shared_rank_n < ep_ws):
        raise ValueError("shared_expert_rank_num must be in [0, ep_world_size).")
    if shared_rank_n == 0:
        if shared_n not in (0, 1):
            raise ValueError("when shared_expert_rank_num=0, shared_expert_num must be 0 or 1.")
    else:
        if shared_n == 0 or shared_rank_n % shared_n != 0:
            raise ValueError("when shared_expert_rank_num>0, require shared_expert_num>0 and shared_expert_rank_num % shared_expert_num == 0.")
    if gbs < 0:
        raise ValueError("global_bs must be >= 0.")
    if out_dt != 0:
        raise ValueError("out_dtype currently only supports 0.")
    if qmode not in (0, 2):
        raise ValueError("comm_quant_mode currently only supports 0 or 2.")
    if gl_type != 0:
        raise ValueError("group_list_type currently only supports 0.")
    max_i32 = 2147483647
    if not (0 <= zero_n < max_i32 and 0 <= copy_n < max_i32 and 0 <= const_n < max_i32):
        raise ValueError("zero_expert_num/copy_expert_num/const_expert_num must be in [0, INT32_MAX).")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2"}:
        if comm_alg not in {"", "fullmesh", "hierarchy"}:
            raise ValueError("A2 comm_alg only supports '', 'fullmesh', or 'hierarchy'.")
        valid_ep_ws = {2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, 384} if comm_alg != "hierarchy" else {16, 32, 64}
        if ep_ws not in valid_ep_ws:
            raise ValueError(f"A2 ep_world_size={ep_ws} not supported for comm_alg={comm_alg or 'default'}.")
        if tp_ws != 0 or tp_rank != 0 or group_tp != "":
            raise ValueError("A2 does not support TP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
        if shared_n != 0 or shared_rank_n != 0:
            raise ValueError("A2 does not support shared expert; require shared_expert_num=0 and shared_expert_rank_num=0.")
        if shared_expert_x is not None and torch.as_tensor(shared_expert_x).numel() > 0:
            raise ValueError("A2 does not support shared_expert_x.")
        if elastic_info is not None and torch.as_tensor(elastic_info).numel() > 0:
            raise ValueError("A2 does not support elastic_info.")
        if const_n != 0:
            raise ValueError("A2 does not support const expert; const_expert_num must be 0.")
        if any(v is not None and torch.as_tensor(v).numel() > 0 for v in (const_expert_alpha_1, const_expert_alpha_2, const_expert_v)):
            raise ValueError("A2 does not support const_expert_alpha_1/2/const_expert_v.")
        if comm_alg == "hierarchy":
            if expand_scales is None or torch.as_tensor(expand_scales).numel() == 0:
                raise ValueError("A2 with comm_alg='hierarchy' requires expand_scales.")
            if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
                raise ValueError("A2 hierarchy does not support x_active_mask.")
            if ori_x is not None and torch.as_tensor(ori_x).numel() > 0:
                raise ValueError("A2 hierarchy does not support ori_x.")
            if zero_n != 0 or copy_n != 0:
                raise ValueError("A2 hierarchy does not support zero/copy experts.")
    elif soc in {"a3", "atlas_a3"}:
        if ep_ws not in {8, 16, 32, 64, 128, 144, 256, 288}:
            raise ValueError("A3 ep_world_size is not supported.")
        if comm_alg not in {"", "hierarchy"}:
            raise ValueError("A3 comm_alg only supports '' or 'hierarchy'.")
        if not (0 <= tp_ws <= 2):
            raise ValueError("A3 tp_world_size must be in [0, 2].")
        if tp_ws in (0, 1):
            if tp_rank != 0 or group_tp != "":
                raise ValueError("A3 no TP communication requires group_tp='', tp_world_size in {0,1}, tp_rank_id=0.")
        else:
            if group_tp == "" or tp_rank not in (0, 1):
                raise ValueError("A3 tp_world_size=2 requires non-empty group_tp and tp_rank_id in [0,1].")
        if qmode == 2 and tp_ws >= 2:
            raise ValueError("A3 commQuantMode=2 is only supported when tp_world_size < 2.")
        if performance_info is not None and torch.as_tensor(performance_info).numel() > 0:
            raise ValueError("A3 currently does not support performance_info.")
        if comm_alg == "hierarchy":
            if expand_scales is None or torch.as_tensor(expand_scales).numel() == 0:
                raise ValueError("A3 with comm_alg='hierarchy' requires expand_scales.")
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        if not (2 <= ep_ws <= 768):
            raise ValueError("950 ep_world_size must be in [2, 768].")
        if tp_ws != 0 or tp_rank != 0 or group_tp != "":
            raise ValueError("950 does not support TP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
        if comm_alg != "":
            raise ValueError("950 does not support comm_alg; must be ''.")
        if expand_scales is not None and torch.as_tensor(expand_scales).numel() > 0:
            raise ValueError("950 does not support expand_scales.")
        if performance_info is not None and torch.as_tensor(performance_info).numel() > 0:
            raise ValueError("950 currently does not support performance_info.")
    else:
        if not (0 <= tp_ws <= 2):
            raise ValueError("tp_world_size must be in [0, 2].")
        if tp_ws in (0, 1):
            if tp_rank != 0:
                raise ValueError("tp_rank_id must be 0 when tp_world_size is 0 or 1.")
        else:
            if group_tp == "" or tp_rank not in (0, 1):
                raise ValueError("tp_world_size=2 requires non-empty group_tp and tp_rank_id in [0,1].")

    x_t = torch.as_tensor(expand_x)
    ids_t = torch.as_tensor(expert_ids)
    assist_t = torch.as_tensor(assist_info_for_combine)
    ep_send_t = torch.as_tensor(ep_send_counts)
    scales_t = torch.as_tensor(expert_scales)

    if x_t.dim() != 2:
        raise ValueError("expand_x must be 2-D.")
    if ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D.")
    if assist_t.dim() != 1:
        raise ValueError("assist_info_for_combine must be 1-D.")
    if ep_send_t.dim() != 1:
        raise ValueError("ep_send_counts must be 1-D.")
    if scales_t.dim() != 2:
        raise ValueError("expert_scales must be 2-D.")
    if tuple(scales_t.shape) != tuple(ids_t.shape):
        raise ValueError("expert_scales shape must equal expert_ids shape.")

    bs, k = int(ids_t.shape[0]), int(ids_t.shape[1])
    h = int(x_t.shape[1])
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > (moe_n + zero_n + copy_n + const_n):
        raise ValueError("K must be <= moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num.")
    if bs <= 0:
        raise ValueError("Bs must be positive.")

    if x_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expand_x dtype must be float16 or bfloat16.")
    if ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")
    if assist_t.dtype != torch.int32:
        raise ValueError("assist_info_for_combine dtype must be int32.")
    if ep_send_t.dtype != torch.int32:
        raise ValueError("ep_send_counts dtype must be int32.")
    if scales_t.dtype != torch.float32:
        raise ValueError("expert_scales dtype must be float32.")

    local_expert_num = moe_n // local_expert_den
    expect_ep = ep_ws * max(tp_ws, 1) * local_expert_num
    if int(ep_send_t.shape[0]) != expect_ep:
        raise ValueError(f"ep_send_counts shape must be ({expect_ep},).")
    if local_expert_num > 1 and tp_ws > 1:
        raise ValueError("TP communication is not supported when localExpertNum > 1.")

    a_mul_tp = int(x_t.shape[0])
    tp_mul = max(tp_ws, 1)
    if a_mul_tp % tp_mul != 0:
        raise ValueError("expand_x first dim must be divisible by max(tp_world_size, 1).")
    a = a_mul_tp // tp_mul
    if int(assist_t.shape[0]) != a * 128:
        raise ValueError("assist_info_for_combine shape must be (A * 128,).")

    if tp_send_counts is not None and torch.as_tensor(tp_send_counts).numel() > 0:
        tp_send_t = torch.as_tensor(tp_send_counts)
        if tp_ws <= 1:
            raise ValueError("tp_send_counts must be empty when there is no TP communication.")
        if tp_send_t.dim() != 1 or int(tp_send_t.shape[0]) != tp_ws or tp_send_t.dtype != torch.int32:
            raise ValueError("tp_send_counts must be int32 1-D with length tp_world_size.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        mask_t = torch.as_tensor(x_active_mask)
        if mask_t.dtype != torch.bool:
            raise ValueError("x_active_mask must be bool when provided.")
        if mask_t.dim() == 1:
            if int(mask_t.shape[0]) != bs:
                raise ValueError("1-D x_active_mask shape must be (Bs,).")
            # 1D 时 true 需排在 false 前
            hit_false = False
            for v in mask_t.tolist():
                if not v:
                    hit_false = True
                elif hit_false:
                    raise ValueError("1-D x_active_mask requires all true values before false values.")
        elif mask_t.dim() == 2:
            if tuple(mask_t.shape) != (bs, k):
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
        exs_t = torch.as_tensor(expand_scales)
        if exs_t.dtype != torch.float32:
            raise ValueError("expand_scales dtype must be float32 when provided.")

    if shared_expert_x is not None and torch.as_tensor(shared_expert_x).numel() > 0:
        sx_t = torch.as_tensor(shared_expert_x)
        if sx_t.dtype != x_t.dtype:
            raise ValueError("shared_expert_x dtype must match expand_x dtype.")
        if sx_t.dim() == 2:
            if tuple(sx_t.shape) != (bs, h):
                raise ValueError("2-D shared_expert_x shape must be (Bs, H).")
        elif sx_t.dim() == 3:
            if tuple(sx_t.shape) != (bs, 1, h):
                raise ValueError("3-D shared_expert_x shape must be (Bs, 1, H).")
        else:
            raise ValueError("shared_expert_x must be 2-D or 3-D when provided.")

    if elastic_info is not None and torch.as_tensor(elastic_info).numel() > 0:
        ei_t = torch.as_tensor(elastic_info)
        if ei_t.dtype != torch.int32:
            raise ValueError("elastic_info dtype must be int32 when provided.")

    if performance_info is not None and torch.as_tensor(performance_info).numel() > 0:
        p_t = torch.as_tensor(performance_info)
        if p_t.dtype != torch.int64:
            raise ValueError("performance_info dtype must be int64 when provided.")
        if p_t.dim() != 1 or int(p_t.shape[0]) != ep_ws:
            raise ValueError("performance_info must be 1-D with shape (ep_world_size,).")

    if ori_x is not None and torch.as_tensor(ori_x).numel() > 0:
        ox_t = torch.as_tensor(ori_x)
        if ox_t.dtype != x_t.dtype or tuple(ox_t.shape) != (bs, h):
            raise ValueError("ori_x must have shape (Bs, H) and same dtype as expand_x.")
    elif copy_n != 0 or const_n != 0:
        raise ValueError("ori_x is required when copy_expert_num != 0 or const_expert_num != 0.")

    if const_n != 0:
        for name, tensor_like, expect_shape in (
            ("const_expert_alpha_1", const_expert_alpha_1, (const_n,)),
            ("const_expert_alpha_2", const_expert_alpha_2, (const_n,)),
            ("const_expert_v", const_expert_v, (const_n, h)),
        ):
            if tensor_like is None or torch.as_tensor(tensor_like).numel() == 0:
                raise ValueError(f"{name} is required when const_expert_num != 0.")
            t = torch.as_tensor(tensor_like)
            if t.dtype != x_t.dtype or tuple(t.shape) != expect_shape:
                raise ValueError(f"{name} must have shape {expect_shape} and same dtype as expand_x.")
    else:
        for name, tensor_like in (
            ("const_expert_alpha_1", const_expert_alpha_1),
            ("const_expert_alpha_2", const_expert_alpha_2),
            ("const_expert_v", const_expert_v),
        ):
            if tensor_like is not None and torch.as_tensor(tensor_like).numel() > 0:
                raise ValueError(f"{name} must be empty when const_expert_num == 0.")

    return (expand_x,expert_ids,assist_info_for_combine,ep_send_counts,expert_scales,
                                     tp_send_counts,x_active_mask,activation_scale,weight_scale,group_list,
                                     expand_scales,shared_expert_x,elastic_info,ori_x,const_expert_alpha_1,
                                     const_expert_alpha_2,const_expert_v,performance_info,group_ep,ep_world_size,
                                     ep_rank_id,moe_expert_num,group_tp,tp_world_size,tp_rank_id,
                                     expert_shard_type,shared_expert_num,shared_expert_rank_num,
                                     global_bs,out_dtype,comm_quant_mode,group_list_type,
                                     comm_alg,zero_expert_num,copy_expert_num,const_expert_num,kwargs)