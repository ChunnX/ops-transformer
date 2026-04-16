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
            "ffn_to_attention": "ffn_to_attention_inputs"
        }
}

import torch

def ffn_to_attention_inputs(x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table = None, group = None, world_size = None, token_info_table_shape = None, token_data_shape = None, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (1 <= len(group) < 128):
        raise ValueError("group length must be in [1, 128).")

    if int(world_size) != world_size or not (2 <= int(world_size) <= 768):
        raise ValueError("world_size must be an integer in [2, 768].")

    x_t = torch.as_tensor(x)
    if x_t.dim() != 2:
        raise ValueError("x must be a 2-D tensor with shape (Y, H).")
    y, h = map(int, x_t.shape)
    if not (1024 <= h <= 8192):
        raise ValueError("H must satisfy 1024 <= H <= 8192.")

    for name, tensor_like in [
        ("session_ids", session_ids),
        ("micro_batch_ids", micro_batch_ids),
        ("token_ids", token_ids),
        ("expert_offsets", expert_offsets),
    ]:
        t = torch.as_tensor(tensor_like)
        if t.dim() != 1 or int(t.shape[0]) != y:
            raise ValueError(f"{name} must be 1-D with shape (Y,).")

    actual_t = torch.as_tensor(actual_token_num)
    if actual_t.dim() != 1 or int(actual_t.numel()) != 1:
        raise ValueError("actual_token_num must be a 1-D tensor with shape (1,).")
    actual_v = int(actual_t.reshape(-1)[0].item())
    if not (0 <= actual_v <= y):
        raise ValueError("actual_token_num value must be in [0, Y].")

    # token_info_table_shape: [MicroBatchNum, Bs, ExpertNumPerToken]
    if len(token_info_table_shape) != 3:
        raise ValueError("token_info_table_shape length must be 3.")
    micro_batch_num, bs, expert_num_per_token = map(int, token_info_table_shape)
    if micro_batch_num <= 0:
        raise ValueError("MicroBatchNum must be > 0.")
    if not (0 < bs <= 512):
        raise ValueError("Bs must satisfy 0 < Bs <= 512.")
    if expert_num_per_token <= 0:
        raise ValueError("ExpertNumPerToken must be > 0.")

    # token_data_shape: [MicroBatchNum, Bs, ExpertNumPerToken, HS, ...]（README: 长度/scale维信息）
    if len(token_data_shape) < 4:
        raise ValueError("token_data_shape length must be at least 4.")
    if int(token_data_shape[0]) != micro_batch_num:
        raise ValueError("token_data_shape[0] must equal token_info_table_shape[0] (MicroBatchNum).")
    if int(token_data_shape[1]) != bs:
        raise ValueError("token_data_shape[1] must equal token_info_table_shape[1] (Bs).")
    if int(token_data_shape[2]) != expert_num_per_token:
        raise ValueError("token_data_shape[2] must equal token_info_table_shape[2] (ExpertNumPerToken).")
    hs = int(token_data_shape[3])
    if not (1152 <= hs <= 8320):
        raise ValueError("HS must satisfy 1152 <= HS <= 8320.")

    sid_t = torch.as_tensor(session_ids)
    mb_t = torch.as_tensor(micro_batch_ids)
    tok_t = torch.as_tensor(token_ids)
    off_t = torch.as_tensor(expert_offsets)

    if sid_t.numel() > 0 and torch.any(sid_t < 0):
        raise ValueError("session_ids must be non-negative.")
    if mb_t.numel() > 0:
        if torch.any(mb_t < 0) or torch.any(mb_t >= micro_batch_num):
            raise ValueError("micro_batch_ids values must be in [0, MicroBatchNum-1].")
    if tok_t.numel() > 0:
        if torch.any(tok_t < 0) or torch.any(tok_t >= bs):
            raise ValueError("token_ids values must be in [0, Bs-1].")
    if off_t.numel() > 0:
        if torch.any(off_t < 0) or torch.any(off_t >= expert_num_per_token):
            raise ValueError("expert_offsets values must be in [0, ExpertNumPerToken-1].")

    if attn_rank_table is not None:
        rt = torch.as_tensor(attn_rank_table)
        if rt.numel() > 0 and rt.dim() != 1:
            raise ValueError("attn_rank_table should be 1-D when provided as non-empty tensor.")

    return (
        x, session_ids, micro_batch_ids, token_ids, expert_offsets,
        actual_token_num, attn_rank_table, group, world_size,
        token_info_table_shape, token_data_shape, kwargs
    )