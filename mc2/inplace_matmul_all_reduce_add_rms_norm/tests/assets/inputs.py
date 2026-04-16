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
            "inplace_matmul_all_reduce_add_rms_norm": "inplace_matmul_all_reduce_add_rms_norm_inputs"
        }
}

import torch

def inplace_matmul_all_reduce_add_rms_norm_inputs(x1, x2, bias = None, residual = None, gamma = None, antiquant_scale = None, antiquant_offset = None, dequant_scale = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, antiquant_group_size = 0, epsilon = 1e-6, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if not (0 < len(group) < 128):
        raise ValueError("group length must be in (0, 128).")

    if str(reduce_op).lower() != "sum":
        raise ValueError("reduce_op currently only supports 'sum'.")
    if int(comm_turn) != 0:
        raise ValueError("comm_turn currently only supports 0.")
    if is_trans_a:
        raise ValueError("x1 transpose is not supported (is_trans_a must be False).")
    if not (0.0 < float(epsilon) < 1.0):
        raise ValueError("epsilon must be in (0, 1).")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    residual_t = torch.as_tensor(residual)
    gamma_t = torch.as_tensor(gamma)

    if x1_t.dim() not in (2, 3):
        raise ValueError("x1 must be 2-D (s,k) or 3-D (b,s,k).")
    if x2_t.dim() != 2:
        raise ValueError("x2 must be 2-D.")
    if residual_t.dim() != 3:
        raise ValueError("residual must be 3-D (b,s,n).")
    if gamma_t.dim() != 1:
        raise ValueError("gamma must be 1-D (n,).")

    # x2 as (k,n) or (n,k) depending on transpose flag
    if not is_trans_b:
        k2, n = int(x2_t.shape[0]), int(x2_t.shape[1])
    else:
        n, k2 = int(x2_t.shape[0]), int(x2_t.shape[1])

    k = int(x1_t.shape[-1])
    if k == 0:
        raise ValueError("k must be > 0 (k=0 empty tensor is not supported).")
    if k != k2:
        raise ValueError(f"k mismatch: x1 k={k}, x2 k={k2}.")
    if not (1 <= k <= 65535) or not (1 <= n <= 65535):
        raise ValueError("k and n must be in [1, 65535].")

    # residual / gamma / bias shape constraints
    if int(residual_t.shape[-1]) != n:
        raise ValueError("residual last dim must equal n.")
    if int(gamma_t.shape[0]) != n:
        raise ValueError("gamma shape must be (n,).")

    if x1_t.dim() == 2:
        s = int(x1_t.shape[0])
        if int(residual_t.shape[0] * residual_t.shape[1]) != s:
            raise ValueError("when x1 is 2-D, residual (b*s) must equal x1 first dim.")
    else:
        b, s = int(x1_t.shape[0]), int(x1_t.shape[1])
        if tuple(residual_t.shape[:2]) != (b, s):
            raise ValueError("when x1 is 3-D, residual shape[:2] must equal (b,s).")

    if bias is not None:
        b_t = torch.as_tensor(bias)
        if b_t.numel() > 0:
            if b_t.dim() != 1:
                raise ValueError("bias must be 1-D when non-empty.")
            if int(b_t.shape[0]) != n:
                raise ValueError("bias length must equal n.")

    # dtype constraints (README): x1/bias/residual/gamma/y/normOut一致；x2为int8或int4
    x1_dt = x1_t.dtype
    if x1_dt not in (torch.float16, torch.bfloat16):
        raise ValueError("x1 dtype must be float16 or bfloat16.")
    if residual_t.dtype != x1_dt or gamma_t.dtype != x1_dt:
        raise ValueError("residual and gamma dtype must match x1 dtype.")
    if bias is not None and torch.as_tensor(bias).numel() > 0 and torch.as_tensor(bias).dtype != x1_dt:
        raise ValueError("bias dtype must match x1 dtype when provided.")
    # PyTorch 无 int4 dtype，允许 int8 以及通过 kwargs 明示 int4 场景
    if x2_t.dtype != torch.int8 and str(kwargs.get("x2_dtype", "")).lower() not in {"int4", "fp4_e2m1", "fp4_e1m2"}:
        raise ValueError("x2 dtype must be int8 (or int4 via case metadata).")

    # antiquant/dequant shape checks
    if antiquant_scale is not None:
        as_t = torch.as_tensor(antiquant_scale)
        if int(antiquant_group_size) == 0:
            # pertensor/perchannel
            ok = (
                tuple(as_t.shape) == (1,) or
                tuple(as_t.shape) == (n,) or
                tuple(as_t.shape) == (1, n)
            )
            if not ok:
                raise ValueError("antiquant_scale shape must be (1)/(n)/(1,n) when antiquant_group_size=0.")
        else:
            if int(antiquant_group_size) % 32 != 0:
                raise ValueError("antiquant_group_size must be multiple of 32 when non-zero.")
            if not (32 <= int(antiquant_group_size) <= max(32, min(k - 1, 2147483647))):
                raise ValueError("antiquant_group_size out of valid range.")
            kg = (k + int(antiquant_group_size) - 1) // int(antiquant_group_size)
            if tuple(as_t.shape) != (kg, n):
                raise ValueError(f"pergroup antiquant_scale shape must be ({kg}, {n}).")
        if antiquant_offset is not None:
            ao_t = torch.as_tensor(antiquant_offset)
            if tuple(ao_t.shape) != tuple(as_t.shape):
                raise ValueError("antiquant_offset shape must equal antiquant_scale shape.")

    if dequant_scale is not None:
        ds_t = torch.as_tensor(dequant_scale)
        if tuple(ds_t.shape) not in {(1,), (n,), (1, n)}:
            raise ValueError("dequant_scale shape must be (1)/(n)/(1,n).")

    return (x1,x2,bias,residual,gamma,antiquant_scale,antiquant_offset,
            dequant_scale,group,reduce_op,is_trans_a,is_trans_b,comm_turn,
            antiquant_group_size,epsilon,kwargs)