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
            "all_gather_matmul": "all_gather_matmul_inputs"
        }
}

import torch


def all_gather_matmul_inputs(x1, x2, bias = None, group = None, is_trans_a = False, is_trans_b = False, gather_index = 0, comm_turn = 0, rank_size = 0, is_gather_out = True, **kwargs):
    """
    依据 README「约束说明」与 aclnn 侧约定，对入参做静态校验。

    - gather_index / commTurn 当前仅支持 0；
    - x1 仅支持不转置（is_trans_a 须为 False）；x2 可转置；
    - 非空 tensor 时 x1/x2 为 2 维，k 轴一致且 k∈[256,65535)（空 tensor 场景按 README 另行满足 k 非空等条件）；
    - bias 非空时须为 1 维且长度等于 n；
    - rank_size 若指定须为正整数。
    """
    if gather_index != 0:
        raise ValueError("all_gather_matmul: gather_index 当前版本仅支持 0（README）。")
    if comm_turn != 0:
        raise ValueError("all_gather_matmul: commTurn 当前版本仅支持 0（README）。")
    if is_trans_a:
        raise ValueError(
            "all_gather_matmul: x1 仅支持不转置，is_trans_a 须为 False（README / aclnnAllGatherMatmul）。")

    if rank_size < 0:
        raise ValueError("all_gather_matmul: rank_size 不得为负数。")
    if rank_size != 0 and int(rank_size) != rank_size:
        raise ValueError("all_gather_matmul: rank_size 须为整数。")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)

    # 与 golden 一致：标量空 tensor 不做 shape/k 约束
    if x1_t.shape == torch.Size([]):
        return x1, x2, bias, group, is_trans_a, is_trans_b, gather_index, comm_turn, rank_size, is_gather_out, kwargs

    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError(
            "all_gather_matmul: 当前版本输入 x1、x2 须为 2 维 tensor（README：x1 为 (m,k)，x2 为 (k,n) 或转置相关形状）。")

    m, k = int(x1_t.shape[0]), int(x1_t.shape[1])
    if is_trans_b:
        n, k2 = int(x2_t.shape[0]), int(x2_t.shape[1])
    else:
        k2, n = int(x2_t.shape[0]), int(x2_t.shape[1])

    if k == 0:
        raise ValueError("all_gather_matmul: k 轴不可为空（长度为 0）（README）。")
    if k != k2:
        raise ValueError(
            f"all_gather_matmul: matmul 要求 k 轴一致，x1 的 k={k}，x2 侧 k={k2}。")

    # README：k 轴取值范围 [256, 65535)；空 tensor 组合场景下仍要求 k 落在该范围（除非后续文档单独放宽）
    if not (256 <= k < 65535):
        raise ValueError(
            f"all_gather_matmul: k 轴须在 [256, 65535) 内（README），当前 k={k}。")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("all_gather_matmul: bias 非空时须为 1 维（README）。")
            if int(b_t.shape[0]) != n:
                raise ValueError(
                    f"all_gather_matmul: bias 长度须等于 n={n}，当前 bias.shape[0]={b_t.shape[0]}。")

    return x1, x2, bias, group, is_trans_a, is_trans_b, gather_index, comm_turn, rank_size, is_gather_out, kwargs
