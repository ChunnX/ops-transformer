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
            "matmul_all_reduce": "matmul_all_reduce_golden"
        }
}

import torch
import torch.distributed as dist


def matmul_all_reduce_golden(x1, x2, bias = None, x3 = None, antiquant_scale = None, antiquant_offset = None, dequant_scale = None, pertoken_scale = None, comm_quant_scale_1 = None, comm_quant_scale_2 = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, antiquant_group_size = 0, group_size = 0, y_dtype = 28, comm_quant_mode = 0, **kwargs):       # 28 : static_cast<int>(ge::DT_UNDEFINED)
    del group
    del antiquant_group_size
    del group_size
    del y_dtype
    del comm_quant_mode
    del kwargs

    # README 约束：当前仅支持 sum / comm_turn=0 / x1不转置
    if reduce_op.lower() != "sum":
        raise ValueError("matmul_all_reduce currently only supports reduce_op='sum'.")
    if comm_turn != 0:
        raise ValueError("matmul_all_reduce currently only supports comm_turn=0.")
    if is_trans_a:
        raise ValueError("matmul_all_reduce does not support is_trans_a=True.")

    x1_t = torch.as_tensor(x1).to(torch.float32)
    x2_t = torch.as_tensor(x2).to(torch.float32)
    if is_trans_b:
        x2_t = x2_t.t()

    out = torch.matmul(x1_t, x2_t)

    if bias is not None:
        b = torch.as_tensor(bias).to(torch.float32)
        out = out + b
    if x3 is not None:
        out = out + torch.as_tensor(x3).to(torch.float32)

    # 量化相关输入给出可运行参考路径（主要用于框架侧 golden 兼容）
    if antiquant_scale is not None:
        out = out * torch.as_tensor(antiquant_scale).to(torch.float32)
    if antiquant_offset is not None:
        out = out + torch.as_tensor(antiquant_offset).to(torch.float32)
    if dequant_scale is not None:
        out = out * torch.as_tensor(dequant_scale).to(torch.float32)
    if pertoken_scale is not None:
        ps = torch.as_tensor(pertoken_scale).to(torch.float32)
        if ps.dim() == 1 and out.shape[0] == ps.shape[0]:
            view_shape = [ps.shape[0]] + [1] * (out.dim() - 1)
            out = out * ps.reshape(view_shape)
        else:
            out = out * ps
    if comm_quant_scale_1 is not None and comm_quant_scale_2 is not None:
        cq1 = torch.as_tensor(comm_quant_scale_1).to(torch.float32)
        cq2 = torch.as_tensor(comm_quant_scale_2).to(torch.float32)
        out = out * (cq1 / cq2) * cq2

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out