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
            "all_gather_matmul_v2": "all_gather_matmul_v2_golden"
        }
}


import torch
import torch.distributed as dist


def all_gather_matmul_v2_golden(x1,x2,bias,x1_scale,x2_scale,
                                quant_scale,group,
                                is_trans_a:bool = False,is_trans_b:bool=False,
                                gather_index:int = 0,comm_turn:int = 0,
                                rank_size:int = 0,block_size:int = 0,
                                group_size:int = 0,is_gather_out:bool = True,
                                is_amax_out:bool=False,y_dtype:int=28,comm_mode:str = 'aicpu',**kwargs):     # y_dtype:int=28 : static_cast<int>(ge::DT_UNDEFINED)
    del group
    del comm_turn
    del group_size
    del y_dtype
    del comm_mode

    if is_trans_a:
        raise ValueError("all_gather_matmul_v2 does not support is_trans_a=True in current implementation.")
    if gather_index != 0:
        raise ValueError("gather_index must be 0 for current all_gather_matmul_v2 implementation.")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)
    if x1_t.shape == torch.Size([]):
        empty = torch.tensor([], dtype=torch.float32)
        if is_amax_out:
            return empty, empty, torch.tensor(0.0, dtype=torch.float32)
        return empty, empty if is_gather_out else None

    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("all_gather_matmul_v2_golden expects 2-D x1 and x2.")

    x1_f = x1_t.to(torch.float32)
    x2_f = x2_t.to(torch.float32)
    if is_trans_b:
        x2_f = x2_f.t()

    if rank_size > 0:
        world_size = int(rank_size)
    else:
        if not dist.is_available() or not dist.is_initialized():
            world_size = 1
        else:
            world_size = dist.get_world_size()

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        all_gather_list = [torch.zeros_like(x1_f) for _ in range(world_size)]
        dist.all_gather(all_gather_list, x1_f)
        gather_out = torch.cat(all_gather_list, dim=0)
    else:
        gather_out = torch.cat([x1_f] * world_size, dim=0)

    out = torch.matmul(gather_out, x2_f)

    if bias is not None:
        b = torch.as_tensor(bias, dtype=torch.float32)
        if b.numel() > 0:
            out = out + b.reshape(-1)

    # 量化路径 CPU 参考（简化）：对齐 MC2_test get_cpu 的“先 matmul 再 scale”
    # per-block/mxfp 依赖专用核函数，此处只提供通用可运行标杆。
    if x1_scale is not None and x2_scale is not None:
        s1 = torch.as_tensor(x1_scale, dtype=torch.float32)
        s2 = torch.as_tensor(x2_scale, dtype=torch.float32)
        if block_size > 0:
            # block 量化场景下按最后一维近似广播，作为可运行 CPU 标杆。
            if s1.dim() > 1:
                s1 = s1.reshape(s1.shape[0], -1).mean(dim=1, keepdim=True)
            else:
                s1 = s1.reshape(-1, 1)
            s2 = s2.reshape(-1).mean()
            out = out * (s1 * s2)
        else:
            s = s1.reshape(-1) * s2.reshape(-1)
            out = out * s.reshape(-1, 1)
    if quant_scale is not None:
        qs = torch.as_tensor(quant_scale, dtype=torch.float32)
        if qs.numel() > 0:
            out = out * qs.reshape(-1)[0]

    if is_amax_out:
        amax_out = torch.max(torch.abs(out)).to(torch.float32)
        return out, (gather_out if is_gather_out else None), amax_out
    return out, (gather_out if is_gather_out else None)