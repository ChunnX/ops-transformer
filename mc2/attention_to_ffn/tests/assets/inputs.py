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
            "attention_to_ffn": "attention_to_ffn_inputs"
        }
}

import torch

def attention_to_ffn_inputs(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, scales = None, active_mask = None, group = None, world_size = None, ffn_token_info_table_shape = None, ffn_token_data_shape = None, attn_token_info_table_shape = None, moe_expert_num = None, quant_mode = 0, sync_flag = 0, ffn_start_rank_id = 0, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (1 <= len(group) < 128):
        raise ValueError("group length must be in [1, 128).")

    if int(world_size) != world_size or not (2 <= int(world_size) <= 768):
        raise ValueError("world_size must be an integer in [2, 768].")

    if int(moe_expert_num) != moe_expert_num or not (0 < int(moe_expert_num) <= 1024):
        raise ValueError("moe_expert_num must be an integer in (0, 1024].")

    if quant_mode not in (0, 2):
        raise ValueError("quant_mode only supports 0 (non-quant) or 2 (dynamic quant).")
    if sync_flag not in (0, 1):
        raise ValueError("sync_flag only supports 0 (sync) or 1 (async).")
    if int(ffn_start_rank_id) != ffn_start_rank_id:
        raise ValueError("ffn_start_rank_id must be an integer.")

    x_t = torch.as_tensor(x)
    if x_t.dim() != 3:
        raise ValueError("x must be a 3-D tensor with shape (X, BS, H).")
    x_dim, bs, h = map(int, x_t.shape)
    if x_dim != 1:
        raise ValueError("Current version only supports X == 1.")
    if not (0 < bs <= 512):
        raise ValueError("BS must satisfy 0 < BS <= 512.")
    if not (1024 <= h <= 8192):
        raise ValueError("H must satisfy 1024 <= H <= 8192.")

    for name, tensor_like in [
        ("session_id", session_id),
        ("micro_batch_id", micro_batch_id),
        ("layer_id", layer_id),
    ]:
        t = torch.as_tensor(tensor_like)
        if t.dim() != 1 or int(t.shape[0]) != x_dim:
            raise ValueError(f"{name} must be 1-D with shape (X,).")

    expert_ids_t = torch.as_tensor(expert_ids)
    if expert_ids_t.dim() != 3:
        raise ValueError("expert_ids must be a 3-D tensor with shape (X, BS, K).")
    if int(expert_ids_t.shape[0]) != x_dim or int(expert_ids_t.shape[1]) != bs:
        raise ValueError("expert_ids first two dims must match (X, BS).")
    k = int(expert_ids_t.shape[2])
    if not (0 < k <= 16):
        raise ValueError("K must satisfy 0 < K <= 16.")
    if k > int(moe_expert_num):
        raise ValueError("K must satisfy K <= moe_expert_num.")
    if expert_ids_t.numel() > 0:
        if torch.any(expert_ids_t < 0) or torch.any(expert_ids_t >= int(moe_expert_num)):
            raise ValueError("expert_ids values must be in [0, moe_expert_num).")

    rank_table_t = torch.as_tensor(expert_rank_table)
    if rank_table_t.dim() != 3:
        raise ValueError("expert_rank_table must be a 3-D tensor with shape (L, moeExpertNum+sharedExpertNum, M).")
    l, expert_axis, m = map(int, rank_table_t.shape)
    if l != 1:
        raise ValueError("Current version only supports L == 1.")
    if expert_axis < int(moe_expert_num):
        raise ValueError("expert_rank_table.shape[1] must be >= moe_expert_num.")
    if m <= 0:
        raise ValueError("expert_rank_table.shape[2] (M) must be > 0.")

    if scales is not None:
        scales_t = torch.as_tensor(scales)
        if scales_t.dim() != 3:
            raise ValueError("scales must be a 3-D tensor when provided.")
        if tuple(scales_t.shape) != (l, expert_axis, h):
            raise ValueError(
                f"scales shape must be ({l}, {expert_axis}, {h}) when provided, got {tuple(scales_t.shape)}."
            )

    if active_mask is not None:
        active_mask_t = torch.as_tensor(active_mask)
        if active_mask_t.dim() != 2 or tuple(active_mask_t.shape) != (x_dim, bs):
            raise ValueError("active_mask must have shape (X, BS) when provided.")

    if len(ffn_token_info_table_shape) != 3:
        raise ValueError("ffn_token_info_table_shape length must be 3.")
    if len(ffn_token_data_shape) != 5:
        raise ValueError("ffn_token_data_shape length must be 5.")
    if len(attn_token_info_table_shape) != 3:
        raise ValueError("attn_token_info_table_shape length must be 3.")

    return (
        x, session_id, micro_batch_id, layer_id, expert_ids,
        expert_rank_table, scales, active_mask, group, world_size,
        ffn_token_info_table_shape, ffn_token_data_shape,
        attn_token_info_table_shape, moe_expert_num, quant_mode,
        sync_flag, ffn_start_rank_id, kwargs
    )