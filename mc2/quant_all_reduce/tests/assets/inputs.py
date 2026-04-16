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
            "quant_all_reduce": "quant_all_reduce_inputs"
        }
}

import torch

def quant_all_reduce_inputs(x, scales, group, reduce_op = 'sum', output_dtype = 27, world_size = None, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group length must be in [1, 128).")

    if str(reduce_op).lower() != "sum":
        raise ValueError("reduce_op currently only supports 'sum'.")

    if world_size is None:
        raise ValueError("world_size is required.")
    if int(world_size) != world_size:
        raise ValueError("world_size must be an integer.")
    ws = int(world_size)
    if ws not in {2, 4, 8}:
        raise ValueError("world_size currently only supports {2, 4, 8}.")

    if int(output_dtype) != output_dtype:
        raise ValueError("output_dtype must be an integer.")
    odt = int(output_dtype)
    if odt not in (1, 27, 28):
        raise ValueError("output_dtype only supports FLOAT16(1), BFLOAT16(27), FLOAT(28).")

    x_t = torch.as_tensor(x)
    s_t = torch.as_tensor(scales)

    if x_t.numel() == 0 or s_t.numel() == 0:
        raise ValueError("x and scales do not support empty tensor.")
    if x_t.dim() not in (2, 3):
        raise ValueError("x must be 2-D or 3-D.")
    if s_t.dim() not in (2, 3, 4):
        raise ValueError("scales must be 2-D/3-D/4-D.")

    # 按 def.cpp + README：x 支持 int8 / hifloat8 / float8 / float4（float8/float4 通过 metadata 放行）
    x_dtype_ok = x_t.dtype == torch.int8 or str(kwargs.get("x_dtype", "")).lower() in {
        "hifloat8", "float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2"
    }
    if not x_dtype_ok:
        raise ValueError("x dtype must be int8 (or hifloat8/float8/float4 via x_dtype metadata).")

    # scales 支持 float 或 float8_e8m0（后者通过 metadata 放行）
    scales_dtype_ok = s_t.dtype == torch.float32 or str(kwargs.get("scales_dtype", "")).lower() == "float8_e8m0"
    if not scales_dtype_ok:
        raise ValueError("scales dtype must be float32 (or float8_e8m0 via scales_dtype metadata).")
    if str(kwargs.get("scales_dtype", "")).lower() == "float8_e8m0":
        x_meta = str(kwargs.get("x_dtype", "")).lower()
        if x_meta not in {"float8_e4m3fn", "float8_e5m2"}:
            raise ValueError("when scales is float8_e8m0, x must be float8_e4m3fn or float8_e5m2 (README).")

    if tuple(s_t.shape[:-1]) != tuple(x_t.shape[:-1]):
        raise ValueError("scales leading dimensions must match x leading dimensions.")

    h = int(x_t.shape[-1])
    if not (1024 <= h <= 8192) or (h % 128 != 0):
        raise ValueError("x last dim H must be in [1024, 8192] and aligned to 128.")

    if s_t.dtype == torch.float32:
        # pertoken-pergroup: scales last dim = H / 128
        if int(s_t.shape[-1]) != h // 128:
            raise ValueError("float scales requires scales.shape[-1] == H/128.")
    else:
        # mx: README 要求 scales 形状严格为 (..., H/64, 2)
        if not (s_t.dim() == x_t.dim() + 1 and int(s_t.shape[-2]) == h // 64 and int(s_t.shape[-1]) == 2):
            raise ValueError("float8_e8m0 scales requires shape (..., H/64, 2).")

    soc = str(kwargs.get("soc", "")).lower()
    if soc and soc not in {"950", "ascend950", "950pr", "950dt"}:
        raise ValueError("quant_all_reduce is only supported on Ascend 950 series.")

    return (x, scales, group, reduce_op, output_dtype, world_size, kwargs)
