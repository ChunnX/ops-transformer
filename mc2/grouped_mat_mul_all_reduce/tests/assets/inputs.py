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
            "grouped_mat_mul_all_reduce": "grouped_mat_mul_all_reduce_inputs"
        }
}

import torch

def grouped_mat_mul_all_reduce_inputs(x, weight, bias, group_list = None, splitItem = 0, group = None, reduceOp = 'sum', commTurn = 0, **kwargs):
    if not isinstance(group, str):
        raise TypeError("group must be a string.")
    if len(group) == 0:
        raise ValueError("group must be non-empty.")

    if str(reduceOp).lower() != "sum":
        raise ValueError("reduceOp currently only supports 'sum'.")
    if int(commTurn) != 0:
        raise ValueError("commTurn currently only supports 0.")

    split_item = int(splitItem)
    if split_item not in (0, 1, 2, 3):
        raise ValueError("splitItem must be one of {0,1,2,3}.")

    def _to_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [torch.as_tensor(t) for t in v]
        return [torch.as_tensor(v)]

    x_list = _to_list(x)
    w_list = _to_list(weight)
    b_list = _to_list(bias)

    if len(x_list) == 0 or len(w_list) == 0:
        raise ValueError("x and weight must be non-empty tensor list/value.")
    if len(x_list) > 64 or len(w_list) > 64 or (b_list and len(b_list) > 64):
        raise ValueError("x/weight/bias list length must be <= 64.")

    # splitItem=1/3: x 输入不切分（单tensor）+ 需要 group_list 指示分组
    if split_item in (1, 3):
        if len(x_list) != 1:
            raise ValueError("splitItem=1/3 expects x as a single tensor.")
        if group_list is None:
            raise ValueError("splitItem=1/3 requires non-empty group_list.")
        gl = torch.as_tensor(group_list, dtype=torch.int64).reshape(-1)
        if gl.numel() == 0:
            raise ValueError("group_list must be non-empty for splitItem=1/3.")
        if torch.any(gl <= 0):
            raise ValueError("group_list values must be positive cumulative endpoints.")
        if int(gl[-1].item()) != int(x_list[0].shape[0]):
            raise ValueError("group_list last value must equal x.shape[0] for splitItem=1/3.")
    else:
        if group_list is not None:
            gl = torch.as_tensor(group_list, dtype=torch.int64).reshape(-1)
            if gl.numel() > 64:
                raise ValueError("group_list length must be <= 64.")

    # splitItem=0/2: x 与 weight 分组数应匹配（golden同逻辑）
    if split_item in (0, 2):
        if len(x_list) != len(w_list):
            raise ValueError("For splitItem=0/2, x and weight group count must match.")

    # 维度约束：weight 必须2D；splitItem=1/2/3时 x 需2D；splitItem=0时 x 支持2-6D
    for i, w in enumerate(w_list):
        if w.dim() != 2:
            raise ValueError(f"weight[{i}] must be 2-D.")
        if int(w.shape[-1]) >= 65536:
            raise ValueError(f"weight[{i}] last dim must be < 65536.")

    for i, xv in enumerate(x_list):
        if split_item == 0:
            if not (2 <= xv.dim() <= 6):
                raise ValueError(f"x[{i}] must be 2-6D when splitItem=0.")
        else:
            if xv.dim() != 2:
                raise ValueError(f"x[{i}] must be 2D when splitItem in {{1,2,3}}.")
        if int(xv.shape[-1]) >= 65536:
            raise ValueError(f"x[{i}] last dim must be < 65536.")

    # bias 允许空；非空时长度为1或与组数一致，且每项1D/2D（广播友好）
    if b_list:
        if len(b_list) not in (1, len(w_list)):
            raise ValueError("bias must be empty, length 1, or same length as group count.")
        for i, b in enumerate(b_list):
            if b.dim() not in (1, 2):
                raise ValueError(f"bias[{i}] must be 1-D or 2-D.")

    # 数据类型组合约束（README）
    x_dtypes = {t.dtype for t in x_list}
    w_dtypes = {t.dtype for t in w_list}
    if len(x_dtypes) > 1 or len(w_dtypes) > 1:
        raise ValueError("All x tensors and all weight tensors should have consistent dtype.")
    x_dtype = next(iter(x_dtypes))
    w_dtype = next(iter(w_dtypes))
    if x_dtype != w_dtype:
        raise ValueError("x and weight dtype must be the same.")
    if x_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x/weight dtype must be float16 or bfloat16.")

    if b_list:
        b_dtypes = {t.dtype for t in b_list}
        if len(b_dtypes) > 1:
            raise ValueError("All bias tensors should have consistent dtype.")
        b_dtype = next(iter(b_dtypes))
        if x_dtype == torch.float16 and b_dtype != torch.float16:
            raise ValueError("When x/weight are float16, bias must be float16.")
        if x_dtype == torch.bfloat16 and b_dtype != torch.float32:
            raise ValueError("When x/weight are bfloat16, bias must be float32.")

    return (x, weight, bias, group_list, splitItem, group,
            reduceOp, commTurn, kwargs)