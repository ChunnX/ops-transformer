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
            "matmul_reduce_scatter": "matmul_reduce_scatter_inputs"
        }
}

import torch

def matmul_reduce_scatter_inputs(x1, x2, bias = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, rank_size = 0, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group length must be in (0, 128).")

    if str(reduce_op).lower() != "sum":
        raise ValueError("reduce_op currently only supports 'sum'.")
    if int(comm_turn) != 0:
        raise ValueError("comm_turn currently only supports 0.")
    if is_trans_a:
        raise ValueError("x1 does not support transpose (is_trans_a must be False).")

    if int(rank_size) != rank_size or int(rank_size) <= 0:
        raise ValueError("rank_size must be a positive integer.")
    rs = int(rank_size)

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2"}:
        valid_rs = {2, 4, 8}
    elif soc in {"a3", "atlas_a3"}:
        valid_rs = {2, 4, 8, 16, 32}
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        valid_rs = {2, 4, 8, 16, 32, 64}
    else:
        valid_rs = {2, 4, 8, 16, 32, 64}
    if rs not in valid_rs:
        raise ValueError(f"rank_size={rs} is not supported for soc={soc or 'default'}; expected {sorted(valid_rs)}.")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)

    # 与 README 空tensor约束：允许 m/n 为0，不允许 k 为0
    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("x1 and x2 must be 2-D tensors.")

    m, k = int(x1_t.shape[0]), int(x1_t.shape[1])
    if is_trans_b:
        n, k2 = int(x2_t.shape[0]), int(x2_t.shape[1])
    else:
        k2, n = int(x2_t.shape[0]), int(x2_t.shape[1])

    if k == 0:
        raise ValueError("k must be non-zero.")
    if k != k2:
        raise ValueError(f"k mismatch: x1 k={k}, x2 k={k2}.")
    if not (256 <= k < 65535):
        raise ValueError("k must be in [256, 65535).")

    # m 必须为 rank_size 的整数倍（README）
    if m % rs != 0:
        raise ValueError("m (x1.shape[0]) must be divisible by rank_size.")

    # bias: 可空，非空时 1D 且长度为 n
    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("bias must be 1-D when non-empty.")
            if int(b_t.shape[0]) != n:
                raise ValueError("bias length must equal n.")

    # dtype 一致性（README：x1/x2/output一致；inputs侧可检查 x1/x2）
    if x1_t.dtype != x2_t.dtype:
        raise ValueError("x1 and x2 dtype must be the same.")
    if x1_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x1/x2 dtype must be float16 or bfloat16.")

    return (x1, x2, bias, group, reduce_op, is_trans_a, is_trans_b, comm_turn, rank_size, kwargs)