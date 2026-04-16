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
            "moe_update_expert": "moe_update_expert_inputs"
        }
}

import torch

def moe_update_expert_inputs(expert_ids, eplb_table, expert_scales = None, pruning_threshold = None, active_mask = None, local_rank_id = -1, world_size = -1, balance_mode = 0, **kwargs):
    for name, value in {
        "local_rank_id": local_rank_id,
        "world_size": world_size,
        "balance_mode": balance_mode,
    }.items():
        if int(value) != value:
            raise ValueError(f"{name} must be an integer.")

    local_rank = int(local_rank_id)
    ws = int(world_size)
    bm = int(balance_mode)
    if bm not in (0, 1):
        raise ValueError("balance_mode must be 0 or 1.")
    if ws != -1 and not (2 <= ws <= 768):
        raise ValueError("world_size must be -1 or in [2, 768].")
    if local_rank != -1 and ws != -1 and not (0 <= local_rank < ws):
        raise ValueError("local_rank_id must be -1 or in [0, world_size).")

    ids_t = torch.as_tensor(expert_ids)
    tbl_t = torch.as_tensor(eplb_table)

    if ids_t.dim() != 2:
        raise ValueError("expert_ids must be 2-D with shape (BS, K).")
    if tbl_t.dim() != 2:
        raise ValueError("eplb_table must be 2-D with shape (moeExpertNum, F).")

    if ids_t.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_ids dtype must be int32 or int64.")
    if tbl_t.dtype != torch.int32:
        raise ValueError("eplb_table dtype must be int32.")

    bs, k = int(ids_t.shape[0]), int(ids_t.shape[1])
    moe_n, f = int(tbl_t.shape[0]), int(tbl_t.shape[1])

    if not (1 <= bs <= 512):
        raise ValueError("BS must be in [1, 512].")
    if not (1 <= k <= 16):
        raise ValueError("K must be in [1, 16].")
    if not (0 < moe_n <= 1024):
        raise ValueError("moeExpertNum must be in (0, 1024].")
    if k > moe_n:
        raise ValueError("K must be <= moeExpertNum.")
    if ws != -1 and not (2 <= f <= ws + 1):
        raise ValueError("eplb_table second dim F must satisfy [2, world_size + 1] when world_size is provided.")
    if ws != -1 and moe_n % ws != 0:
        # 文档：各卡部署实例数需相同，可通过 moe_n 被 world_size 整除做保守校验
        raise ValueError("moeExpertNum should be divisible by world_size.")

    # eplb_table 逐行基础合法性：count > 0；若已知 world_size，count <= world_size
    count_col = tbl_t[:, 0]
    if torch.any(count_col <= 0):
        raise ValueError("eplb_table first column (count) must be > 0.")
    if ws != -1 and torch.any(count_col > ws):
        raise ValueError("eplb_table first column (count) must be <= world_size.")

    has_scales = expert_scales is not None and torch.as_tensor(expert_scales).numel() > 0
    has_threshold = pruning_threshold is not None and torch.as_tensor(pruning_threshold).numel() > 0
    has_active = active_mask is not None and torch.as_tensor(active_mask).numel() > 0

    # README 联动约束：
    # 1) expert_scales 与 pruning_threshold 要么都传，要么都不传
    # 2) active_mask 传有效时，expert_scales 与 pruning_threshold 也必须有效
    if has_scales != has_threshold:
        raise ValueError("expert_scales and pruning_threshold must be both provided or both empty.")
    if has_active and (not has_scales or not has_threshold):
        raise ValueError("active_mask requires expert_scales and pruning_threshold to be provided.")

    if has_scales:
        es_t = torch.as_tensor(expert_scales)
        if es_t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("expert_scales dtype must be float16/bfloat16/float32.")
        if es_t.dim() != 2 or tuple(es_t.shape) != (bs, k):
            raise ValueError("expert_scales shape must be (BS, K).")

    if has_threshold:
        pt_t = torch.as_tensor(pruning_threshold)
        if pt_t.dtype != torch.float32:
            raise ValueError("pruning_threshold dtype must be float32.")
        valid_shape = (pt_t.dim() == 1 and int(pt_t.shape[0]) == k) or (pt_t.dim() == 2 and tuple(pt_t.shape) == (1, k))
        if not valid_shape:
            raise ValueError("pruning_threshold shape must be (K,) or (1, K).")

    if has_active:
        am_t = torch.as_tensor(active_mask)
        if am_t.dtype != torch.bool:
            raise ValueError("active_mask dtype must be bool.")
        if am_t.dim() != 1 or int(am_t.shape[0]) != bs:
            raise ValueError("active_mask shape must be (BS,).")
        # true 需排在 false 前
        seen_false = False
        for v in am_t.tolist():
            if not v:
                seen_false = True
            elif seen_false:
                raise ValueError("active_mask requires all true values before false values.")

    soc = str(kwargs.get("soc", "")).lower()
    if soc and soc not in {"a3", "atlas_a3", "950", "ascend950", "950pr", "950dt", "ascend910_93"}:
        raise ValueError("moe_update_expert currently supports A3 and 950 platforms.")

    return (expert_ids,eplb_table,expert_scales,pruning_threshold,active_mask,
            local_rank_id,world_size,balance_mode,kwargs)