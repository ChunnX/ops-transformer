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
            "moe_distribute_dispatch_v2": "moe_distribute_dispatch_v2_inputs"
        }
}

import torch

def moe_distribute_dispatch_v2_inputs(x, expert_ids, scales = None, x_active_mask = None, expert_scales = None, elastic_info = None, performance_info = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, group_tp = '', tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, quant_mode = 0, global_bs = 0, expert_token_nums_type = 1, comm_alg = '', zero_expert_num = 0, copy_expert_num = 0, const_expert_num = 0, y_dtype = 28, **kwargs):
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
        "quant_mode": quant_mode,
        "global_bs": global_bs,
        "expert_token_nums_type": expert_token_nums_type,
        "zero_expert_num": zero_expert_num,
        "copy_expert_num": copy_expert_num,
        "const_expert_num": const_expert_num,
        "y_dtype": y_dtype,
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
    qmode = int(quant_mode)
    gbs = int(global_bs)
    etn_type = int(expert_token_nums_type)
    zero_n = int(zero_expert_num)
    copy_n = int(copy_expert_num)
    const_n = int(const_expert_num)
    ydt = int(y_dtype)

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
    if qmode not in (0, 1, 2, 3, 4):
        raise ValueError("quant_mode must be one of [0, 1, 2, 3, 4].")
    if gbs < 0:
        raise ValueError("global_bs must be >= 0.")
    if etn_type not in (0, 1):
        raise ValueError("expert_token_nums_type must be 0 or 1.")
    max_i32 = 2147483647
    if not (0 <= zero_n < max_i32 and 0 <= copy_n < max_i32 and 0 <= const_n < max_i32):
        raise ValueError("zero_expert_num/copy_expert_num/const_expert_num must be in [0, INT32_MAX).")
    if ydt not in (-1, 28):
        raise ValueError("y_dtype currently only supports -1 or 28.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")
    local_expert_num = moe_n // local_expert_den

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2"}:
        if qmode not in (0, 2):
            raise ValueError("A2 quant_mode only supports 0 or 2.")
        if shared_n != 0 or shared_rank_n != 0:
            raise ValueError("A2 does not support shared expert; require shared_expert_num=0 and shared_expert_rank_num=0.")
        if group_tp != "" or tp_ws != 0 or tp_rank != 0:
            raise ValueError("A2 only supports EP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
        if elastic_info is not None and torch.as_tensor(elastic_info).numel() > 0:
            raise ValueError("A2 does not support elastic_info.")
        if comm_alg not in {"", "fullmesh", "hierarchy"}:
            raise ValueError("A2 comm_alg only supports '', 'fullmesh', or 'hierarchy'.")
        valid_ep = {2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, 384} if comm_alg != "hierarchy" else {16, 32, 64}
        if ep_ws not in valid_ep:
            raise ValueError(f"A2 ep_world_size={ep_ws} not supported for comm_alg={comm_alg or 'default'}.")
        if comm_alg == "hierarchy":
            if scales is not None and torch.as_tensor(scales).numel() > 0:
                raise ValueError("A2 hierarchy does not support scales.")
            if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
                raise ValueError("A2 hierarchy does not support x_active_mask.")
            if zero_n != 0 or copy_n != 0:
                raise ValueError("A2 hierarchy does not support zero/copy experts.")
    elif soc in {"a3", "atlas_a3"}:
        if not (2 <= ep_ws <= 768):
            raise ValueError("A3 ep_world_size must be in [2, 768].")
        if qmode not in (0, 2):
            raise ValueError("A3 quant_mode only supports 0 or 2.")
        if not (0 <= tp_ws <= 2):
            raise ValueError("A3 tp_world_size must be in [0, 2].")
        if tp_ws in (0, 1):
            if tp_rank != 0:
                raise ValueError("A3 tp_rank_id must be 0 when tp_world_size is 0 or 1.")
        else:
            if group_tp == "" or tp_rank not in (0, 1):
                raise ValueError("A3 tp_world_size=2 requires non-empty group_tp and tp_rank_id in [0,1].")
        if comm_alg not in {"", "fullmesh_v1", "fullmesh_v2", "hierarchy"}:
            raise ValueError("A3 comm_alg only supports '', 'fullmesh_v1', 'fullmesh_v2', 'hierarchy'.")
        if elastic_info is not None and torch.as_tensor(elastic_info).numel() > 0:
            raise ValueError("A3 currently does not support elastic_info.")
        if performance_info is not None and torch.as_tensor(performance_info).numel() > 0:
            raise ValueError("A3 currently does not support performance_info.")
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        if not (2 <= ep_ws <= 768):
            raise ValueError("950 ep_world_size must be in [2, 768].")
        if group_tp != "" or tp_ws != 0 or tp_rank != 0:
            raise ValueError("950 only supports EP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
        if comm_alg not in {"", "fullmesh_v1", "fullmesh_v2"}:
            raise ValueError("950 comm_alg only supports '', 'fullmesh_v1', or 'fullmesh_v2'.")
        if elastic_info is not None and torch.as_tensor(elastic_info).numel() > 0:
            raise ValueError("950 currently does not support elastic_info.")
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

    if local_expert_num > 1 and tp_ws > 1:
        raise ValueError("TP communication is not supported when localExpertNum > 1.")

    x_t = torch.as_tensor(x)
    ids_t = torch.as_tensor(expert_ids)
    if x_t.dim() != 2:
        raise ValueError("x must be 2-D.")
    if ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D.")
    if x_t.dtype not in (torch.float16, torch.bfloat16):
        # float8/float4 兼容通过元数据放行
        if str(kwargs.get("x_dtype", "")).lower() not in {"float8_e5m2", "float8_e4m3fn", "hifloat8", "float4_e2m1", "float4_e1m2"}:
            raise ValueError("x dtype must be float16/bfloat16 (or float8/float4 via metadata x_dtype).")
    if ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")

    bs, h = int(x_t.shape[0]), int(x_t.shape[1])
    if int(ids_t.shape[0]) != bs:
        raise ValueError("expert_ids first dim must equal BS.")
    k = int(ids_t.shape[1])
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > (moe_n + zero_n + copy_n + const_n):
        raise ValueError("K must be <= moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num.")

    if scales is not None and torch.as_tensor(scales).numel() > 0:
        s_t = torch.as_tensor(scales)
        # def 中允许 float / float8_e8m0; PyTorch 常用 float32，float8_e8m0 通过元数据兼容
        if s_t.dtype != torch.float32 and str(kwargs.get("scales_dtype", "")).lower() not in {"float8_e8m0"}:
            raise ValueError("scales dtype must be float32 (or float8_e8m0 via metadata scales_dtype).")
        scale_rows = moe_n + (shared_n if shared_rank_n > 0 else 0)
        if s_t.dim() != 2 or tuple(s_t.shape) != (scale_rows, h):
            raise ValueError(f"scales shape must be ({scale_rows}, {h}) when provided.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        mask_t = torch.as_tensor(x_active_mask)
        if mask_t.dtype != torch.bool:
            raise ValueError("x_active_mask dtype must be bool when provided.")
        if mask_t.dim() == 1:
            if int(mask_t.shape[0]) != bs:
                raise ValueError("1-D x_active_mask shape must be (BS,).")
            seen_false = False
            for v in mask_t.tolist():
                if not v:
                    seen_false = True
                elif seen_false:
                    raise ValueError("1-D x_active_mask requires all true values before false values.")
        elif mask_t.dim() == 2:
            if tuple(mask_t.shape) != (bs, k):
                raise ValueError("2-D x_active_mask shape must be (BS, K).")
        else:
            raise ValueError("x_active_mask must be 1-D or 2-D when provided.")

    if expert_scales is not None and torch.as_tensor(expert_scales).numel() > 0:
        es_t = torch.as_tensor(expert_scales)
        if es_t.dtype != torch.float32:
            raise ValueError("expert_scales dtype must be float32 when provided.")
        if es_t.dim() != 2 or tuple(es_t.shape) != (bs, k):
            raise ValueError("expert_scales shape must be (BS, K) when provided.")

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

    return (x,expert_ids,scales,x_active_mask,expert_scales,elastic_info,performance_info,
            group_ep,ep_world_size,ep_rank_id,moe_expert_num,group_tp,tp_world_size,tp_rank_id,
            expert_shard_type,shared_expert_num,shared_expert_rank_num,quant_mode,global_bs,expert_token_nums_type,
            comm_alg,zero_expert_num,copy_expert_num,const_expert_num,y_dtype,kwargs)