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
            "batch_mat_mul_reduce_scatter_allto_all": "batch_mat_mul_reduce_scatter_allto_all_inputs"
        }
}

import torch

def batch_mat_mul_reduce_scatter_allto_all_inputs(x, weight, bias = None, group_ep = None, group_tp = None, ep_world_size = None, tp_world_size = None, y_shard_type = 0, transpose_weight = False, **kwargs):
    if not isinstance(group_ep, str) or not isinstance(group_tp, str):
        raise TypeError("group_ep and group_tp must be strings.")
    if not (0 < len(group_ep) < 128) or not (0 < len(group_tp) < 128):
        raise ValueError("group_ep/group_tp length must be in (0, 128).")
    if group_ep == group_tp:
        raise ValueError("group_ep and group_tp must not be the same.")

    valid_world_sizes = {2, 4, 8, 16, 32}
    if int(ep_world_size) != ep_world_size or int(ep_world_size) not in valid_world_sizes:
        raise ValueError(f"ep_world_size must be one of {sorted(valid_world_sizes)}.")
    if int(tp_world_size) != tp_world_size or int(tp_world_size) not in valid_world_sizes:
        raise ValueError(f"tp_world_size must be one of {sorted(valid_world_sizes)}.")

    if int(y_shard_type) != y_shard_type or int(y_shard_type) not in (0, 1):
        raise ValueError("y_shard_type must be 0 (H shard) or 1 (C shard).")

    x_t = torch.as_tensor(x)
    w_t = torch.as_tensor(weight)
    if x_t.dim() != 3 or w_t.dim() != 3:
        raise ValueError("x and weight must be 3-D tensors.")

    ep = int(ep_world_size)
    tp = int(tp_world_size)
    yst = int(y_shard_type)

    e_div_ep = int(x_t.shape[0])
    if not (1 <= e_div_ep <= 32):
        raise ValueError("E/ep (=x.shape[0]) must be in [1, 32].")

    e_total = e_div_ep * ep
    if not (2 <= e_total <= 512):
        raise ValueError("E must be in [2, 512].")

    if int(x_t.shape[1]) <= 0:
        raise ValueError("C-related x.shape[1] must be > 0.")

    if not transpose_weight:
        m_div_tp = int(w_t.shape[1])
        h = int(w_t.shape[2])
    else:
        # 允许测试输入以转置形态提供
        m_div_tp = int(w_t.shape[2])
        h = int(w_t.shape[1])

    if not (1 <= m_div_tp <= 65535):
        raise ValueError("M/tp must be in [1, 65535].")
    if not (1 <= h <= 65535):
        raise ValueError("H must be in [1, 65535].")
    if yst == 0 and h % tp != 0:
        raise ValueError("For y_shard_type=0, H must be divisible by tp_world_size.")

    # x 与 weight 的 bmm 对齐：x[..., M/tp] @ weight[..., M/tp, ...]
    if int(x_t.shape[2]) != m_div_tp:
        raise ValueError(
            f"x.shape[2] must equal M/tp from weight, got x.shape[2]={x_t.shape[2]} and M/tp={m_div_tp}."
        )
    if int(w_t.shape[0]) != e_div_ep:
        raise ValueError("weight.shape[0] must equal x.shape[0] (=E/ep).")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        target_n = h // tp if yst == 0 else h
        if b_t.numel() > 0:
            if b_t.dim() == 2:
                if tuple(b_t.shape) != (e_div_ep, target_n):
                    raise ValueError(
                        f"2-D bias must be (E/ep, N) = ({e_div_ep}, {target_n}), got {tuple(b_t.shape)}."
                    )
            elif b_t.dim() == 3:
                if tuple(b_t.shape) != (e_div_ep, 1, target_n):
                    raise ValueError(
                        f"3-D bias must be (E/ep, 1, N) = ({e_div_ep}, 1, {target_n}), got {tuple(b_t.shape)}."
                    )
            else:
                raise ValueError("bias must be 2-D or 3-D when non-empty.")

    return (
        x, weight, bias, group_ep, group_tp,
        ep_world_size, tp_world_size, y_shard_type,
        transpose_weight, kwargs
    )