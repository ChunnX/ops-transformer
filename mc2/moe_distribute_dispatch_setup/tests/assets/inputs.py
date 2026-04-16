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
            "moe_distribute_dispatch_setup": "moe_distribute_dispatch_setup_inputs"
        }
}

import torch

def moe_distribute_dispatch_setup_inputs(x, expert_ids, scales = None, x_active_mask = None, group_ep = None, ep_world_size = None, ep_rank_id = None, moe_expert_num = None, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, quant_mode = 0, global_bs = 0, comm_type = 0, comm_alg = '', **kwargs):
    if not isinstance(group_ep, str):
        raise TypeError("group_ep must be a string.")
    if not (0 < len(group_ep) < 128):
        raise ValueError("group_ep length must be in [1, 128).")
    if not isinstance(comm_alg, str):
        raise TypeError("comm_alg must be a string.")

    int_fields = {
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "moe_expert_num": moe_expert_num,
        "expert_shard_type": expert_shard_type,
        "shared_expert_num": shared_expert_num,
        "shared_expert_rank_num": shared_expert_rank_num,
        "quant_mode": quant_mode,
        "global_bs": global_bs,
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
    qmode = int(quant_mode)
    gbs = int(global_bs)
    ctype = int(comm_type)

    if ep_ws <= 0:
        raise ValueError("ep_world_size must be positive.")
    if not (0 <= ep_rank < ep_ws):
        raise ValueError("ep_rank_id must be in [0, ep_world_size).")
    if not (0 < moe_n <= 512):
        raise ValueError("moe_expert_num must be in (0, 512].")
    if shard_type != 0:
        raise ValueError("expert_shard_type currently only supports 0.")
    if shared_n not in (0, 1):
        raise ValueError("shared_expert_num currently only supports 0 or 1.")
    if not (0 <= shared_rank_n < ep_ws):
        raise ValueError("shared_expert_rank_num must be in [0, ep_world_size).")
    if shared_rank_n != 0 and ep_ws % shared_rank_n != 0:
        raise ValueError("ep_world_size must be divisible by shared_expert_rank_num when non-zero.")
    if qmode not in (0, 1, 2, 3, 4):
        raise ValueError("quant_mode must be one of [0, 1, 2, 3, 4].")
    if gbs < 0:
        raise ValueError("global_bs must be >= 0.")
    if ctype != 0:
        raise ValueError("comm_type currently only supports 0.")

    local_expert_den = ep_ws - shared_rank_n
    if local_expert_den <= 0:
        raise ValueError("ep_world_size - shared_expert_rank_num must be positive.")
    if moe_n % local_expert_den != 0:
        raise ValueError("moe_expert_num must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0.")

    # 原型仅注册在 ascend950；非 950 场景给出友好提示（不硬阻塞）
    soc = str(kwargs.get("soc", "")).lower()
    if soc and soc not in {"950", "ascend950", "950pr", "950dt"}:
        raise ValueError("moe_distribute_dispatch_setup is currently only registered for ascend950.")

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
    if int(ids_t.shape[0]) != bs:
        raise ValueError("expert_ids first dim must equal BS.")
    k = int(ids_t.shape[1])
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if k > moe_n:
        raise ValueError("K must be <= moe_expert_num.")

    # 可选输入：可空；传入时校验
    if scales is not None and torch.as_tensor(scales).numel() > 0:
        s_t = torch.as_tensor(scales)
        if s_t.dtype != torch.float32:
            raise ValueError("scales dtype must be float32 when provided.")
        scale_rows = moe_n + (shared_n if shared_rank_n > 0 else 0)
        if s_t.dim() != 2 or tuple(s_t.shape) != (scale_rows, h):
            raise ValueError(f"scales shape must be ({scale_rows}, {h}) when provided.")

    if x_active_mask is not None and torch.as_tensor(x_active_mask).numel() > 0:
        mask_t = torch.as_tensor(x_active_mask)
        if mask_t.dtype != torch.bool:
            raise ValueError("x_active_mask dtype must be bool when provided.")
        if mask_t.dim() not in (1, 2):
            raise ValueError("x_active_mask must be 1-D or 2-D when provided.")
        if mask_t.dim() == 1 and int(mask_t.shape[0]) != bs:
            raise ValueError("1-D x_active_mask shape must be (BS,).")
        if mask_t.dim() == 2 and tuple(mask_t.shape) != (bs, k):
            raise ValueError("2-D x_active_mask shape must be (BS, K).")

    return (x,expert_ids,scales,x_active_mask,group_ep,ep_world_size,
            ep_rank_id,moe_expert_num,expert_shard_type,shared_expert_num,shared_expert_rank_num,
            quant_mode,global_bs,comm_type,comm_alg,kwargs)