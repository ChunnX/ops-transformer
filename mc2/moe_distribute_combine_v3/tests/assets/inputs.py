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
            "moe_distribute_combine_v3": "moe_distribute_combine_v3_inputs"
        }
}

import torch

def moe_distribute_combine_v3_inputs(context, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, tp_send_counts = None, x_active_mask = None, activation_scale = None, weight_scale = None, group_list = None, expand_scales = None, shared_expert_x = None, elastic_info = None, ori_x = None, const_expert_alpha_1 = None, const_expert_alpha_2 = None, const_expert_v = None, performance_info = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, ccl_buffer_size = None, tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, global_bs = 0, out_dtype = 0, comm_quant_mode = 0, group_list_type = 0, comm_alg = '', zero_expert_num = 0, copy_expert_num = 0, const_expert_num = 0, **kwargs):
    if not isinstance(comm_alg, str):
        raise TypeError("comm_alg must be a string.")

    int_fields = {
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "ccl_buffer_size": ccl_buffer_size,
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
    ccl_bs = int(ccl_buffer_size)
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
    if ep_ws not in {8, 16, 32, 64, 128, 144, 256, 288}:
        raise ValueError("ep_world_size is not supported for moe_distribute_combine_v3 (A3 only).")
    if not (0 <= ep_rank < ep_ws):
        raise ValueError("ep_rank_id must be in [0, ep_world_size).")
    if not (0 < moe_n <= 1024):
        raise ValueError("moe_expert_num must be in (0, 1024].")
    if ccl_bs <= 0:
        raise ValueError("ccl_buffer_size must be positive.")
    if not (0 <= tp_ws <= 2):
        raise ValueError("tp_world_size must be in [0, 2].")
    if tp_ws in (0, 1):
        if tp_rank != 0:
            raise ValueError("tp_rank_id must be 0 when tp_world_size is 0 or 1.")
    else:
        if tp_rank not in (0, 1):
            raise ValueError("tp_rank_id must be in [0, 1] when tp_world_size=2.")

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
    if qmode == 2 and tp_ws >= 2:
        raise ValueError("comm_quant_mode=2 is only supported when tp_world_size < 2.")
    if gl_type != 0:
        raise ValueError("group_list_type currently only supports 0.")
    if comm_alg != "":
        raise ValueError("comm_alg is not supported on v3 currently, must be ''.")
    max_i32 = 2147483647
    if not (0 <= zero_n < max_i32 and 0 <= copy_n < max_i32 and 0 <= const_n < max_i32):
        raise ValueError("zero_expert_num/copy_expert_num/const_expert_num must be in [0, INT32_MAX).")

    # 这里按算子原型定义校验（op_host: context 为 int32 必填）
    ctx_t = torch.as_tensor(context)
    x_t = torch.as_tensor(expand_x)
    ids_t = torch.as_tensor(expert_ids)
    assist_t = torch.as_tensor(assist_info_for_combine)
    ep_send_t = torch.as_tensor(ep_send_counts)
    scales_t = torch.as_tensor(expert_scales)

    if ctx_t.dtype != torch.int32:
        raise ValueError("context dtype must be int32 (by operator prototype).")
    if ctx_t.numel() == 0:
        raise ValueError("context must be non-empty.")

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

    if x_t.dtype not in (torch.float16, torch.bfloat16, torch.int32):
        raise ValueError("expand_x dtype must be float16/bfloat16/int32.")
    if ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")
    if assist_t.dtype != torch.int32:
        raise ValueError("assist_info_for_combine dtype must be int32.")
    if ep_send_t.dtype != torch.int32:
        raise ValueError("ep_send_counts dtype must be int32.")
    if scales_t.dtype != torch.float32:
        raise ValueError("expert_scales dtype must be float32.")

    bs, k = int(ids_t.shape[0]), int(ids_t.shape[1])
    h = int(x_t.shape[1])
    if not (1 <= bs <= 512):
        raise ValueError("Bs must be in [1, 512].")
    if not (1 <= h <= 8192):
        raise ValueError("H must be in [1, 8192].")
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > (moe_n + zero_n + copy_n + const_n):
        raise ValueError("K must be <= moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")
    local_expert_num = moe_n // local_expert_den
    if local_expert_num > 1 and tp_ws > 1:
        raise ValueError("TP communication is not supported when localExpertNum > 1.")
    expect_ep = ep_ws * max(tp_ws, 1) * local_expert_num
    if int(ep_send_t.shape[0]) != expect_ep:
        raise ValueError(f"ep_send_counts shape must be ({expect_ep},).")

    a_mul_tp = int(x_t.shape[0])
    tp_mul = max(tp_ws, 1)
    if a_mul_tp % tp_mul != 0:
        raise ValueError("expand_x first dim must be divisible by max(tp_world_size, 1).")
    a = a_mul_tp // tp_mul
    if int(assist_t.shape[0]) != a * 128:
        raise ValueError("assist_info_for_combine shape must be (A * 128,).")

    # 可选输入：默认可空；传入时做合法性检查
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
            seen_false = False
            for v in mask_t.tolist():
                if not v:
                    seen_false = True
                elif seen_false:
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
        raise ValueError("expand_scales is not supported on v3 currently.")

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
        raise ValueError("performance_info is reserved and must be nullptr/empty on v3 currently.")

    # 条件必填参数（README 明确）
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

    return (context,expand_x,expert_ids,assist_info_for_combine,ep_send_counts,expert_scales,
                                     tp_send_counts,x_active_mask,activation_scale,weight_scale,group_list,
                                     expand_scales,shared_expert_x,elastic_info,ori_x,const_expert_alpha_1,
                                     const_expert_alpha_2,const_expert_v,performance_info,ep_world_size,
                                     ep_rank_id,moe_expert_num,ccl_buffer_size,tp_world_size,tp_rank_id,
                                     expert_shard_type,shared_expert_num,shared_expert_rank_num,
                                     global_bs,out_dtype,comm_quant_mode,group_list_type,
                                     comm_alg,zero_expert_num,copy_expert_num,const_expert_num,kwargs)