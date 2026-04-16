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
            "moe_distribute_combine_setup": "moe_distribute_combine_setup_inputs"
        }
}

import torch

def moe_distribute_combine_setup_inputs(expand_x, expert_ids, assist_info_for_combine, group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, global_bs = 0, comm_quant_mode = 0, comm_type = 0, comm_alg = '', **kwargs):
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
    assist_t = torch.as_tensor(assist_info_for_combine)

    if x_t.dim() != 2:
        raise ValueError("expand_x must be 2-D.")
    if ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D.")
    if assist_t.dim() != 1:
        raise ValueError("assist_info_for_combine must be 1-D.")

    if x_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expand_x dtype must be float16 or bfloat16.")
    if ids_t.dtype != torch.int32:
        raise ValueError("expert_ids dtype must be int32.")
    if assist_t.dtype != torch.int32:
        raise ValueError("assist_info_for_combine dtype must be int32.")

    bs, k = int(ids_t.shape[0]), int(ids_t.shape[1])
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > moe_n:
        raise ValueError("K must be <= moe_expert_num.")
    if not (1 <= bs <= 512):
        raise ValueError("Bs must be in [1, 512].")

    a = int(x_t.shape[0])
    if int(assist_t.shape[0]) != a * 128:
        raise ValueError("assist_info_for_combine shape must be (A * 128,).")

    return (expand_x,expert_ids,assist_info_for_combine,group_ep,ep_world_size,ep_rank_id,
            moe_expert_num,expert_shard_type,shared_expert_num,shared_expert_rank_num,
            global_bs,comm_quant_mode,comm_type,comm_alg,kwargs)