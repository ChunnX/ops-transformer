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
            "matmul_reduce_scatter_v2": "matmul_reduce_scatter_v2_inputs"
        }
}

import torch

def _dtype_name(tensor, meta_name):
    if meta_name:
        return str(meta_name).lower()
    return str(torch.as_tensor(tensor).dtype).lower()


def matmul_reduce_scatter_v2_inputs(x1, x2, bias = None, x1_scale = None, x2_scale = None, quant_scale = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, rank_size = 0, block_size = 0, group_size = 0, is_amax_out = False, y_dtype = 28, comm_mode = 'aicpu', **kwargs):
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
    if int(block_size) != 0:
        raise ValueError("block_size currently only supports 0.")
    if int(y_dtype) != 28:
        raise ValueError("y_dtype currently only supports 28.")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
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

    if int(rank_size) != rank_size or int(rank_size) <= 0:
        raise ValueError("rank_size must be a positive integer.")
    rs = int(rank_size)
    if m % rs != 0:
        raise ValueError("m (x1.shape[0]) must be divisible by rank_size.")

    soc = str(kwargs.get("soc", "")).lower()
    if soc in {"a2", "atlas_a2", "a3", "atlas_a3"}:
        valid_rs = {2, 4, 8}
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        valid_rs = {2, 4, 8, 16, 32, 64}
    else:
        valid_rs = {2, 4, 8, 16, 32, 64}
    if rs not in valid_rs:
        raise ValueError(f"rank_size={rs} is not supported for soc={soc or 'default'}; expected {sorted(valid_rs)}.")
    if soc in {"a2", "atlas_a2", "a3", "atlas_a3"} and (m <= 0 or n <= 0):
        raise ValueError("A2/A3 does not support empty tensor on m/n axis.")

    cm = str(comm_mode).lower()
    if soc in {"a2", "atlas_a2", "a3", "atlas_a3"}:
        if cm != "aiv":
            raise ValueError("A2/A3 currently only supports comm_mode='aiv'.")
        if int(group_size) != 0:
            raise ValueError("A2/A3 currently only supports group_size=0.")
    elif soc in {"950", "ascend950", "950pr", "950dt"}:
        if cm != "ccu":
            raise ValueError("950 currently only supports comm_mode='ccu'.")
    else:
        if cm not in {"aicpu", "aiv", "ccu"}:
            raise ValueError("comm_mode must be one of {'aicpu', 'aiv', 'ccu'}.")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("bias must be 1-D when non-empty.")
            if int(b_t.shape[0]) != n:
                raise ValueError("bias length must equal n.")
            if soc in {"a2", "atlas_a2", "a3", "atlas_a3"} and cm == "aiv":
                raise ValueError("A2/A3 + aiv scene requires bias to be nullptr.")

    x1_name = _dtype_name(x1_t, kwargs.get("x1_dtype"))
    x2_name = _dtype_name(x2_t, kwargs.get("x2_dtype"))

    fp16_family = {"float16", "bfloat16"}
    is_fp16_path = any(x in x1_name for x in fp16_family) and any(x in x2_name for x in fp16_family)
    if is_fp16_path:
        if x1_name != x2_name:
            raise ValueError("float16/bfloat16 path requires x1/x2 dtype to be identical.")
        if x1_scale is not None and torch.as_tensor(x1_scale).numel() > 0:
            raise ValueError("x1_scale must be nullptr/empty when x1/x2 are float16/bfloat16.")
        if x2_scale is not None and torch.as_tensor(x2_scale).numel() > 0:
            raise ValueError("x2_scale must be nullptr/empty when x1/x2 are float16/bfloat16.")
    else:
        if soc in {"a2", "atlas_a2", "a3", "atlas_a3"}:
            if "int8" not in x1_name or "int8" not in x2_name:
                raise ValueError("A2/A3 non-fp16 path only supports int8.")
            if x1_name != x2_name:
                raise ValueError("A2/A3 requires x1/x2 dtype identical.")
        elif soc in {"950", "ascend950", "950pr", "950dt"}:
            lowbit = ("hifloat8", "float8_e4m3fn", "float8_e5m2")
            if not any(x in x1_name for x in lowbit) or not any(x in x2_name for x in lowbit):
                raise ValueError("950 non-fp16 path only supports hifloat8/float8.")
            # README: hifloat8 必须同型；e4m3/e5m2可混合
            if ("hifloat8" in x1_name or "hifloat8" in x2_name) and x1_name != x2_name:
                raise ValueError("950 hifloat8 path requires x1/x2 dtype identical.")
            if x1_scale is None or torch.as_tensor(x1_scale).numel() == 0:
                raise ValueError("950 float8/hifloat8 path requires non-empty x1_scale.")
            if x2_scale is None or torch.as_tensor(x2_scale).numel() == 0:
                raise ValueError("950 float8/hifloat8 path requires non-empty x2_scale.")
        else:
            if x1_t.dtype != x2_t.dtype:
                raise ValueError("x1 and x2 dtype must be the same.")
            if x1_t.dtype != torch.int8:
                raise ValueError("default non-fp16 path only supports int8.")

    if quant_scale is not None and torch.as_tensor(quant_scale).numel() > 0:
        raise ValueError("quant_scale currently only supports nullptr/empty.")

    if bool(is_amax_out):
        raise ValueError("is_amax_out currently only supports False.")

    return (x1, x2, bias, x1_scale, x2_scale, quant_scale, group,
                                    reduce_op, is_trans_a, is_trans_b, comm_turn, rank_size, block_size,
                                    group_size, is_amax_out, y_dtype, comm_mode, kwargs)