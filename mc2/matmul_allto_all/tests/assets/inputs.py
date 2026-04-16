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
            "matmul_allto_all": "matmul_allto_all_inputs"
        }
}

import torch

def matmul_allto_all_inputs(x1, x2, bias = None, x1_scale = None, x2_scale = None, comm_scale = None, x1_offset = None, x2_offset = None, group = None, world_size = None, all2all_axes = {-1,-2}, y_dtype = 28, x1_quant_mode = 0, x2_quant_mode = 0, comm_quant_mode = 0, comm_quant_dtype = 28, transpose_x1 = False, transpose_x2 = False, group_size = 0, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group length must be in (0, 128).")

    if int(world_size) != world_size or int(world_size) <= 0:
        raise ValueError("world_size must be a positive integer.")
    ws = int(world_size)

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2"}:
        valid_ws = {2, 4, 8}
    elif soc in {"a3", "atlas_a3", "950", "ascend950", "950pr", "950dt"}:
        valid_ws = {2, 4, 8, 16}
    else:
        valid_ws = {2, 4, 8, 16}
    if ws not in valid_ws:
        raise ValueError(f"world_size={ws} is not supported for soc={soc or 'default'}; expected {sorted(valid_ws)}.")

    if transpose_x1:
        raise ValueError("transpose_x1 is not supported, must be False.")

    if int(comm_quant_mode) != 0:
        raise ValueError("comm_quant_mode currently only supports 0.")
    if int(comm_quant_dtype) not in (-1, 28):
        raise ValueError("comm_quant_dtype currently only supports -1/28 (UNDEFINED placeholder).")
    if int(group_size) < 0:
        raise ValueError("group_size must be >= 0.")

    if all2all_axes is None:
        all2all_axes = [-1, -2]
    axes = list(all2all_axes)
    if len(axes) != 2 or sorted(axes) != [-2, -1]:
        raise ValueError("all2all_axes must be empty/default or [-1, -2].")

    if int(x1_quant_mode) not in range(0, 8) or int(x2_quant_mode) not in range(0, 8):
        raise ValueError("x1_quant_mode and x2_quant_mode must be in [0, 7].")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("x1 and x2 must be 2-D.")

    bs, h1 = int(x1_t.shape[0]), int(x1_t.shape[1])
    if bs * ws > 2147483647:
        raise ValueError("BS * world_size must be <= INT32_MAX.")
    if not (1 <= h1 <= 65535):
        raise ValueError("H1 must be in [1, 65535].")

    if not transpose_x2:
        if int(x2_t.shape[0]) != h1:
            raise ValueError("transpose_x2=False expects x2.shape[0] == H1.")
        h2 = int(x2_t.shape[1])
    else:
        if int(x2_t.shape[1]) != h1:
            raise ValueError("transpose_x2=True expects x2.shape[1] == H1.")
        h2 = int(x2_t.shape[0])

    if h2 <= 0 or h2 > 2147483647:
        raise ValueError("H2 must be in [1, INT32_MAX].")
    if h2 % ws != 0:
        raise ValueError("H2 must be divisible by world_size.")

    # bias shape
    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("bias must be 1-D when non-empty.")
            if int(b_t.shape[0]) != h2:
                raise ValueError("bias length must equal H2.")

    # scale shapes
    if x1_scale is not None:
        s1 = torch.as_tensor(x1_scale)
        if int(x1_quant_mode) == 6:
            ce = (h1 + 63) // 64
            if tuple(s1.shape) != (bs, ce, 2):
                raise ValueError(f"mx mode: x1_scale shape must be ({bs}, {ce}, 2).")
        else:
            if s1.dim() != 1 or int(s1.shape[0]) != bs:
                raise ValueError(f"non-mx mode: x1_scale shape must be ({bs},).")

    if x2_scale is not None:
        s2 = torch.as_tensor(x2_scale)
        if int(x2_quant_mode) == 6:
            ce2 = (h1 + 63) // 64
            if tuple(s2.shape) != (h2, ce2, 2):
                raise ValueError(f"mx mode: x2_scale shape must be ({h2}, {ce2}, 2).")
        else:
            if s2.dim() != 1 or int(s2.shape[0]) != h2:
                raise ValueError(f"non-mx mode: x2_scale shape must be ({h2},).")

    # 量化模式组合（README）
    if soc in {"a2", "atlas_a2"}:
        if int(x1_quant_mode) != 0 or int(x2_quant_mode) != 0:
            if not (int(x1_quant_mode) == 3 and int(x2_quant_mode) == 2):
                raise ValueError("A2 quant mode only supports x1_quant_mode=3 and x2_quant_mode=2.")
    elif soc in {"a3", "atlas_a3"}:
        if int(x1_quant_mode) != 0 or int(x2_quant_mode) != 0:
            raise ValueError("A3 currently does not support quantization modes.")
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        if int(x1_quant_mode) != 0 or int(x2_quant_mode) != 0:
            ok = (int(x1_quant_mode), int(x2_quant_mode)) in {(3, 2), (6, 6)}
            if not ok:
                raise ValueError("950 quant mode only supports (3,2) or (6,6).")

    return (x1,x2,bias,x1_scale,x2_scale,comm_scale,x1_offset,x2_offset,
            group,world_size,all2all_axes,y_dtype,x1_quant_mode,x2_quant_mode,comm_quant_mode,
            comm_quant_dtype,transpose_x1,transpose_x2,group_size,kwargs)