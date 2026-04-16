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
            "allto_all_all_gather_batch_mat_mul": "allto_all_all_gather_batch_mat_mul_inputs"
        }
}

import torch

_VALID_EP_TP = {2, 4, 8, 16, 32}


def allto_all_all_gather_batch_mat_mul_inputs(x, weight, bias = None, group_ep = None, group_tp = None, ep_world_size = None, tp_world_size = None, x_shard_type = 0, act_type = 0, transpose_weight = False, output_y2_flag = False, output_y3_flag = False, **kwargs):
    """
    依据 README「约束说明」对入参做静态校验（与 golden 中假设一致）。

    - group_ep / group_tp：非空字符串，长度 (0,128)，且互不相同；
    - ep_world_size / tp_world_size ∈ {2,4,8,16,32}；
    - x_shard_type ∈ {0,1}；act_type ∈ {0,1,2,3,4}；
    - x、weight 均为 3 维；E、C、H、M/tp 等关系见 README 中 xShardType 0/1 两表；
    - bias 非空时须为 2 维 (E/ep, M/tp) 或 3 维 (E/ep, 1, M/tp)。
    """
    def _check_group(name: str, g) -> None:
        if not isinstance(g, str):
            raise TypeError(f"{name} must be str (README).")
        if not (0 < len(g) < 128):
            raise ValueError(f"{name} string length must be in (0, 128) (README).")

    _check_group("group_ep", group_ep)
    _check_group("group_tp", group_tp)
    if group_ep == group_tp:
        raise ValueError("allto_all_all_gather_batch_mat_mul: group_ep and group_tp must not be equal (README).")

    ep = int(ep_world_size)
    tp = int(tp_world_size)
    if ep not in _VALID_EP_TP or tp not in _VALID_EP_TP:
        raise ValueError(
            f"ep_world_size/tp_world_size must be one of {sorted(_VALID_EP_TP)} (README), "
            f"got ep={ep}, tp={tp}."
        )

    if x_shard_type not in (0, 1):
        raise ValueError("x_shard_type must be 0 (H shard) or 1 (C shard) (README).")
    if act_type not in (0, 1, 2, 3, 4):
        raise ValueError("act_type must be in [0,4] (README).")

    x_t = torch.as_tensor(x)
    w_t = torch.as_tensor(weight)
    if x_t.dim() != 3 or w_t.dim() != 3:
        raise ValueError("x and weight must be 3-D (README).")

    e = int(x_t.shape[0])
    if not (2 <= e <= 512):
        raise ValueError(f"E=x.shape[0] must be in [2, 512] (README), got E={e}.")
    if e % ep != 0:
        raise ValueError(f"x.shape[0] must be divisible by ep_world_size (README), got {e} % {ep} != 0.")

    e_div_ep = e // ep
    if not (1 <= e_div_ep <= 32):
        raise ValueError(f"E/ep must be in [1, 32] (README), got {e_div_ep}.")

    if int(w_t.shape[0]) != e_div_ep:
        raise ValueError(
            f"weight.shape[0] must equal E/ep={e_div_ep} (README), got weight.shape[0]={w_t.shape[0]}."
        )

    if not transpose_weight:
        h_w = int(w_t.shape[1])
        m_div_tp = int(w_t.shape[2])
    else:
        # golden: permute(0,2,1) 前为 (E/ep, M/tp, H)
        m_div_tp = int(w_t.shape[1])
        h_w = int(w_t.shape[2])

    if not (1 <= m_div_tp <= 65535):
        raise ValueError(f"M/tp must be in [1, 65535] (README), got {m_div_tp}.")

    if x_shard_type == 0:
        # x: (E, C, H/tp), weight: (E/ep, H, M/tp)
        c = int(x_t.shape[1])
        h_div_tp = int(x_t.shape[2])
        if c <= 0:
            raise ValueError("C must be > 0 (README).")
        if h_w != h_div_tp * tp:
            raise ValueError(
                "x_shard_type=0: expect weight H == x.shape[2]*tp_world_size "
                f"(README x:(E,C,H/tp), weight:(E/ep,H,M/tp)), got H_weight={h_w}, "
                f"x.shape[2]*tp={h_div_tp}*{tp}={h_div_tp * tp}."
            )
        if not (1 <= h_w <= 65535):
            raise ValueError(f"H must be in [1, 65535] (README), got H={h_w}.")
    else:
        # x: (E, C/tp, H), weight: (E/ep, H, M/tp)
        c_div_tp = int(x_t.shape[1])
        h_x = int(x_t.shape[2])
        if c_div_tp <= 0:
            raise ValueError("C/tp (x.shape[1]) must be > 0 (README).")
        if h_w != h_x:
            raise ValueError(
                "x_shard_type=1: expect weight.shape[1] == x.shape[2] (H) "
                f"(README), got {h_w} vs {h_x}."
            )
        c_full = c_div_tp * tp
        if c_full <= 0:
            raise ValueError("C must be > 0 (README).")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() == 0:
            pass
        elif b_t.dim() == 2:
            if tuple(b_t.shape) != (e_div_ep, m_div_tp):
                raise ValueError(
                    "bias 2-D shape must be (E/ep, M/tp) (README), "
                    f"expected ({e_div_ep}, {m_div_tp}), got {tuple(b_t.shape)}."
                )
        elif b_t.dim() == 3:
            if tuple(b_t.shape) != (e_div_ep, 1, m_div_tp):
                raise ValueError(
                    "bias 3-D shape must be (E/ep, 1, M/tp) (README), "
                    f"expected ({e_div_ep}, 1, {m_div_tp}), got {tuple(b_t.shape)}."
                )
        else:
            raise ValueError("bias must be 2-D or 3-D when non-empty (README).")

    return (x, weight, bias, group_ep,
            group_tp, ep_world_size, tp_world_size,
            x_shard_type, act_type, transpose_weight, output_y2_flag, output_y3_flag, kwargs)
