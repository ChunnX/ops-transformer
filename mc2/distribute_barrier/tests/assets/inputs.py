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
            "distribute_barrier": "distribute_barrier_inputs"
        }
}

import torch

def distribute_barrier_inputs(x_ref, time_out = None, elastic_info = None, group = None, world_size = None, **kwargs):
    # x_ref: 仅作依赖，不做业务校验；至少要求是 tensor-like
    torch.as_tensor(x_ref)

    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group string length must be in (0, 128).")

    if int(world_size) != world_size or int(world_size) <= 0:
        raise ValueError("world_size must be a positive integer (UINT64 semantics).")

    if time_out is not None:
        t = torch.as_tensor(time_out)
        if t.dtype not in (torch.int32, torch.int64):
            raise ValueError("time_out must be INT32-compatible when provided.")
        if t.numel() != 1:
            raise ValueError("time_out must be a scalar tensor/value when provided.")
        if int(t.reshape(-1)[0].item()) < 0:
            raise ValueError("time_out must be >= 0.")

    if elastic_info is not None:
        ei = torch.as_tensor(elastic_info)
        if ei.dtype not in (torch.int32, torch.int64):
            raise ValueError("elastic_info must be INT32-compatible when provided.")
        if ei.dim() != 1:
            raise ValueError("elastic_info must be a 1-D tensor when provided.")
        expect_len = 4 + 2 * int(world_size)
        if int(ei.numel()) != expect_len:
            raise ValueError(
                f"elastic_info shape must be (4 + 2 * world_size,) = ({expect_len},) when provided."
            )

    return x_ref,time_out,elastic_info,group,world_size,kwargs