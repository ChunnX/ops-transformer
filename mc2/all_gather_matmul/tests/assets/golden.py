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
__golden__ = {
        "kernel": {
            "all_gather_matmul": "all_gather_matmul_golden"
        }
}

import numpy as np
import torch
import torch.distributed as dist


def all_gather_matmul_golden(x1, x2, bias = None, group = None, is_trans_a = False, is_trans_b = False, gather_index = 0, comm_turn = 0, rank_size = 0, is_gather_out = True, **kwargs):
    """
    AllGather + MatMul 标杆（CPU / float32），对齐 MC2_test AllGatherMatmul.get_cpu。

    与 op_host/all_gather_matmul_def.cpp 一致：
    - 输入 x1, x2, 可选 bias；属性 is_trans_a / is_trans_b, gather_index, comm_turn,
      rank_size, is_gather_out；group 仅通信域名，CPU 参考不直接使用。

    aclnn API 侧要求 x1 不做转置（参见 aclnn_all_gather_matmul CheckShape），故 is_trans_a 必须为 False。
    tiling 当前要求 gather_index == 0。
    """
    del group  # HCCL 组名，golden 使用已初始化的默认 process group
    del comm_turn  # 流式/多轮通信参数，CPU 数学参考不涉及

    if is_trans_a:
        raise ValueError(
            "aclnnAllGatherMatmul does not support transposed x1 (is_trans_a must be False).")
    if gather_index != 0:
        raise ValueError("gather_index must be 0 for the current AllGatherMatmul implementation.")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)

    # 与 executor 一致：标量空 tensor 直接返回
    if x1_t.shape == torch.Size([]):
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty

    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("all_gather_matmul_golden expects 2-D x1 and x2.")

    x1_f = x1_t.to(torch.float32)
    x2_f = x2_t.to(torch.float32)
    if is_trans_b:
        x2_f = x2_f.t()     # 只适用于2D

    if rank_size > 0:
        world_size = int(rank_size)
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("rank_size is 0 but torch.distributed is not initialized; pass rank_size explicitly.")
        world_size = dist.get_world_size()

    ag_list = [torch.zeros_like(x1_f) for _ in range(world_size)]
    dist.all_gather(ag_list, x1_f)
    gathered = torch.cat(ag_list, dim=0)

    out = torch.matmul(gathered, x2_f)

    if bias is not None:
        b = torch.as_tensor(bias, dtype=torch.float32)
        if b.numel() == 0:
            pass
        else:
            b = b.reshape(-1)
            out = out + b

    gather_out = gathered if is_gather_out else None

    od = kwargs.get("output_dtypes")
    if od is not None:
        y_dt = od[0] if len(od) > 0 else None
        g_dt = od[1] if len(od) > 1 else y_dt
        if y_dt is not None:
            out = _cast_like_numpy_or_torch(out, y_dt)
        if gather_out is not None and g_dt is not None:
            gather_out = _cast_like_numpy_or_torch(gather_out, g_dt)

    return out, gather_out


def _cast_like_numpy_or_torch(t: torch.Tensor, dtype_spec):
    """dtype_spec: numpy dtype, torch.dtype, or string like 'float16'."""
    if isinstance(dtype_spec, torch.dtype):
        return t.to(dtype_spec)
    if isinstance(dtype_spec, np.dtype):
        return torch.from_numpy(t.detach().cpu().numpy().astype(dtype_spec))
    if isinstance(dtype_spec, str):
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_spec in mapping:
            return t.to(mapping[dtype_spec])
    return torch.from_numpy(t.detach().cpu().numpy().astype(np.dtype(dtype_spec)))
