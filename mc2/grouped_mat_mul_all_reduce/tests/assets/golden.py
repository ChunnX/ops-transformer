# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "grouped_mat_mul_all_reduce": "grouped_mat_mul_all_reduce_golden"
        }
}

import torch
import torch.distributed as dist


def _to_tensor_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [torch.as_tensor(t) for t in v]
    return [torch.as_tensor(v)]


def _split_x_by_group_list(x, group_list):
    gl = [int(i) for i in group_list]
    out, start = [], 0
    for end in gl:
        out.append(x[start:end, :])
        start = end
    return out


def grouped_mat_mul_all_reduce_golden(x,weight,bias,group_list,splitItem,group,
                                      reduceOp,commTurn,**kwargs):
    del group
    del kwargs

    if reduceOp.lower() != "sum":
        raise ValueError("grouped_mat_mul_all_reduce currently only supports reduceOp='sum'.")
    if int(commTurn) != 0:
        raise ValueError("grouped_mat_mul_all_reduce currently only supports commTurn=0.")

    x_list = _to_tensor_list(x)
    w_list = _to_tensor_list(weight)
    b_list = _to_tensor_list(bias)
    split_item = int(splitItem)

    if split_item in (1, 3):
        # 对齐 MC2_test: splitItem=1/3 时按 group_list 对 x 做行切分
        if len(x_list) != 1:
            raise ValueError("splitItem=1/3 expects x as a single tensor.")
        if group_list is None:
            raise ValueError("splitItem=1/3 requires non-empty group_list.")
        x_list = _split_x_by_group_list(x_list[0], group_list)

    if len(x_list) != len(w_list):
        raise ValueError("x and weight group count must match after split.")
    if b_list and len(b_list) not in (1, len(x_list)):
        raise ValueError("bias must be empty, length 1, or same length as groups.")

    out_list = []
    for i, (x_i, w_i) in enumerate(zip(x_list, w_list)):
        y_i = torch.matmul(x_i.to(torch.float32), w_i.to(torch.float32))
        if b_list:
            b_i = b_list[0] if len(b_list) == 1 else b_list[i]
            y_i = y_i + b_i.to(torch.float32)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(y_i, op=dist.ReduceOp.SUM)

        # 输出 dtype 与 x 保持一致（README/def 约束）
        out_list.append(y_i.to(x_i.dtype))

    # splitItem=2/3 表示输出切分到单 tensor，按组拼接返回
    if split_item in (2, 3):
        return [torch.cat(out_list, dim=0)]
    return out_list