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
            "moe_distribute_dispatch": "moe_distribute_dispatch_inputs"
        }
}

import torch

def moe_distribute_dispatch_inputs(x, expert_ids, scales = None, x_active_mask = None, expert_scales = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, group_tp = '', tp_world_size = 0, tp_rank_id = 0, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, quant_mode = 0, global_bs = 0, expert_token_nums_type = 1, **kwargs):
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

    if ep_ws <= 0:
        raise ValueError("ep_world_size must be positive.")
    if not (0 <= ep_rank < ep_ws):
        raise ValueError("ep_rank_id must be in [0, ep_world_size).")
    if not (0 < moe_n <= 512):
        raise ValueError("moe_expert_num must be in (0, 512].")
    if shard_type != 0:
        raise ValueError("expert_shard_type currently only supports 0.")
    if qmode not in (0, 1, 2, 3, 4):
        raise ValueError("quant_mode must be one of [0, 1, 2, 3, 4].")
    if gbs < 0:
        raise ValueError("global_bs must be >= 0.")
    if etn_type not in (0, 1):
        raise ValueError("expert_token_nums_type must be 0 or 1.")

    # 按 README：sharedExpertNum 当前仅支持传 1；且 sharedExpertRankNum 需在范围内并满足可整除约束
    if shared_n != 1:
        raise ValueError("shared_expert_num currently only supports 1.")
    if not (0 <= shared_rank_n < ep_ws):
        raise ValueError("shared_expert_rank_num must be in [0, ep_world_size).")
    if shared_rank_n != 0 and ep_ws % shared_rank_n != 0:
        raise ValueError("ep_world_size must be divisible by shared_expert_rank_num when non-zero.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")
    local_expert_num = moe_n // local_expert_den

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2"}:
        if ep_ws not in {16, 32, 64}:
            raise ValueError("A2 ep_world_size only supports {16, 32, 64}.")
        if qmode not in (0, 2):
            raise ValueError("A2 quant_mode only supports 0 or 2.")
        if group_tp != "" or tp_ws != 0 or tp_rank != 0:
            raise ValueError("A2 only supports EP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
        if shared_rank_n != 0:
            raise ValueError("A2 does not support shared expert rank; shared_expert_rank_num must be 0.")
    elif soc in {"a3", "atlas_a3"}:
        if ep_ws not in {8, 16, 32, 64, 128, 144, 256, 288}:
            raise ValueError("A3 ep_world_size is not supported.")
        if qmode not in (0, 2):
            raise ValueError("A3 quant_mode only supports 0 or 2.")
        if not (0 <= tp_ws <= 2):
            raise ValueError("A3 tp_world_size must be in [0, 2].")
        if tp_ws in (0, 1):
            if tp_rank != 0 or group_tp != "":
                raise ValueError("A3 no TP communication requires group_tp='', tp_world_size in {0,1}, tp_rank_id=0.")
        else:
            if group_tp == "" or tp_rank not in (0, 1):
                raise ValueError("A3 tp_world_size=2 requires non-empty group_tp and tp_rank_id in [0,1].")
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        if ep_ws not in {2, 4, 8, 16, 32, 64, 128, 144, 256, 288}:
            raise ValueError("950 ep_world_size is not supported.")
        if group_tp != "" or tp_ws != 0 or tp_rank != 0:
            raise ValueError("950 currently does not support TP domain; require group_tp='', tp_world_size=0, tp_rank_id=0.")
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
        raise ValueError("x dtype must be float16 or bfloat16.")
    if ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")

    bs, h = int(x_t.shape[0]), int(x_t.shape[1])
    k = int(ids_t.shape[1])
    if int(ids_t.shape[0]) != bs:
        raise ValueError("expert_ids first dim must equal BS.")
    if bs <= 0:
        raise ValueError("BS must be positive.")
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > moe_n:
        raise ValueError("K must be <= moe_expert_num.")

    if scales is not None and torch.as_tensor(scales).numel() > 0:
        s_t = torch.as_tensor(scales)
        if s_t.dtype != torch.float32:
            raise ValueError("scales dtype must be float32 when provided.")
        # README: quant_mode=2 时有效输入 shape 为 (moeExpertNum(+sharedExpertNum), H)；这里按 sharedExpertRankNum 判断
        scale_rows = moe_n + (shared_n if shared_rank_n > 0 else 0)
        if s_t.dim() != 2 or tuple(s_t.shape) != (scale_rows, h):
            raise ValueError(f"scales shape must be ({scale_rows}, {h}) when provided.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        raise ValueError("x_active_mask is reserved and must be nullptr/empty.")

    if expert_scales is not None and torch.as_tensor(expert_scales).numel() > 0:
        es_t = torch.as_tensor(expert_scales)
        if es_t.dtype != torch.float32:
            raise ValueError("expert_scales dtype must be float32 when provided.")
        if es_t.dim() != 2 or tuple(es_t.shape) != (bs, k):
            raise ValueError("expert_scales shape must be (BS, K) when provided.")

    return (x,expert_ids,scales,x_active_mask,expert_scales,group_ep,ep_world_size,
                                   ep_rank_id,moe_expert_num,group_tp,tp_world_size,tp_rank_id,
                                   expert_shard_type,shared_expert_num,shared_expert_rank_num,
                                   quant_mode,global_bs,expert_token_nums_type,kwargs)