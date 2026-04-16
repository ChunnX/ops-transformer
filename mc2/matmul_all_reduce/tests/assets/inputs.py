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
            "matmul_all_reduce": "matmul_all_reduce_inputs"
        }
}

import torch

def matmul_all_reduce_inputs(x1, x2, bias = None, x3 = None, antiquant_scale = None, antiquant_offset = None, dequant_scale = None, pertoken_scale = None, comm_quant_scale_1 = None, comm_quant_scale_2 = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, antiquant_group_size = 0, group_size = 0, y_dtype = 28, comm_quant_mode = 0, **kwargs):
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
    if int(comm_quant_mode) not in (0, 1):
        raise ValueError("comm_quant_mode only supports 0/1.")

    # group_size: perblock 仅支持固定值，其它场景仅支持0
    if int(group_size) not in (0, 549764202624):
        raise ValueError("group_size only supports 0 or 549764202624.")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    if x1_t.dim() not in (2, 3):
        raise ValueError("x1 must be 2-D or 3-D.")
    if x2_t.dim() != 2:
        raise ValueError("x2 must be 2-D for ND path.")

    # k/n shape constraints
    k = int(x1_t.shape[-1])
    if not is_trans_b:
        k2, n = int(x2_t.shape[0]), int(x2_t.shape[1])
    else:
        n, k2 = int(x2_t.shape[0]), int(x2_t.shape[1])
    if k == 0:
        raise ValueError("k=0 empty tensor is not supported.")
    if k != k2:
        raise ValueError(f"k mismatch: x1 k={k}, x2 k={k2}.")
    if not (1 <= k <= 65535 and 1 <= n <= 65535):
        raise ValueError("k and n must be in [1, 65535].")

    # m range: x1 leading element count in matmul semantic
    m = int(x1_t.shape[-2]) if x1_t.dim() == 2 else int(x1_t.shape[0] * x1_t.shape[1])
    if not (1 <= m <= 2147483647):
        raise ValueError("m must be in [1, 2147483647].")

    # x3 / pertoken 与 x1 形状关系
    if x3 is not None:
        x3_t = torch.as_tensor(x3)
        expect_shape = (int(x1_t.shape[0]), int(x1_t.shape[1]), n) if x1_t.dim() == 3 else (int(x1_t.shape[0]), n)
        if tuple(x3_t.shape) != expect_shape:
            raise ValueError(f"x3 shape must be {expect_shape}.")

    if pertoken_scale is not None:
        ps = torch.as_tensor(pertoken_scale)
        expect_len = int(x1_t.shape[0] * x1_t.shape[1]) if x1_t.dim() == 3 else int(x1_t.shape[0])
        if ps.dim() != 1 or int(ps.shape[0]) != expect_len:
            raise ValueError(f"pertoken_scale shape must be ({expect_len},).")

    if bias is not None:
        b = torch.as_tensor(bias)
        if b.numel() > 0:
            if b.dim() != 1:
                raise ValueError("bias must be 1-D when non-empty.")
            if int(b.shape[0]) != n:
                raise ValueError("bias length must equal n.")

    # antiquant / dequant / comm quant scale shape
    if antiquant_scale is not None:
        as_t = torch.as_tensor(antiquant_scale)
        if int(antiquant_group_size) == 0:
            if tuple(as_t.shape) not in {(1,), (n,), (1, n)}:
                raise ValueError("antiquant_scale shape must be (1)/(n)/(1,n) when antiquant_group_size=0.")
        else:
            gs = int(antiquant_group_size)
            if gs % 32 != 0:
                raise ValueError("antiquant_group_size must be multiple of 32.")
            if not (32 <= gs <= max(32, min(k - 1, 2147483647))):
                raise ValueError("antiquant_group_size out of valid range.")
            kg = (k + gs - 1) // gs
            if tuple(as_t.shape) != (kg, n):
                raise ValueError(f"pergroup antiquant_scale shape must be ({kg}, {n}).")
        if antiquant_offset is not None:
            ao_t = torch.as_tensor(antiquant_offset)
            if tuple(ao_t.shape) != tuple(as_t.shape):
                raise ValueError("antiquant_offset shape must equal antiquant_scale shape.")

    if dequant_scale is not None:
        ds = torch.as_tensor(dequant_scale)
        if tuple(ds.shape) not in {(1,), (n,), (1, n)}:
            raise ValueError("dequant_scale shape must be (1)/(n)/(1,n).")

    for name, s in (("comm_quant_scale_1", comm_quant_scale_1), ("comm_quant_scale_2", comm_quant_scale_2)):
        if s is not None:
            st = torch.as_tensor(s)
            if tuple(st.shape) not in {(n,), (1, n)}:
                raise ValueError(f"{name} shape must be (n) or (1,n).")

    # dtype (README 要求 x1/x3/bias/output 一致，x2 为低比特；这里做输入可校验部分)
    if x1_t.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x1 dtype must be float16 or bfloat16.")
    if x3 is not None and torch.as_tensor(x3).dtype != x1_t.dtype:
        raise ValueError("x3 dtype must match x1 dtype.")
    if bias is not None and torch.as_tensor(bias).numel() > 0 and torch.as_tensor(bias).dtype != x1_t.dtype:
        raise ValueError("bias dtype must match x1 dtype when non-empty.")
    if x2_t.dtype not in (torch.int8,):
        # PyTorch 无 int4 dtype，允许通过元数据描述 int4 场景
        if str(kwargs.get("x2_dtype", "")).lower() not in {"int4", "fp4_e2m1", "fp4_e1m2"}:
            raise ValueError("x2 dtype must be int8 (or int4 via metadata x2_dtype).")

    return (x1,x2,bias,x3,antiquant_scale,antiquant_offset,
            dequant_scale,pertoken_scale,comm_quant_scale_1,
            comm_quant_scale_2,group,reduce_op,is_trans_a,is_trans_b,comm_turn,
            antiquant_group_size,group_size,y_dtype,comm_quant_mode,kwargs)