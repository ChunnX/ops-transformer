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
            "matmul_reduce_scatter": "matmul_reduce_scatter_golden"
        }
}

import torch
import torch.distributed as dist


def matmul_reduce_scatter_golden(x1, x2, bias = None, group = None, reduce_op = 'sum', is_trans_a = False, is_trans_b = False, comm_turn = 0, rank_size = 0, **kwargs):
    """
    ReduceScatter(MatMul(x1, x2) + bias) 的 CPU float32 参考，对齐 MC2_test MatmulReduceScatter.get_cpu /
    get_hccl_mm：先 matmul（+bias），各 rank 结果做 SUM，再按行切成 world_size 份，本 rank 取第 rank 份。

    与 README / matmul_reduce_scatter_def.cpp 一致：group 为通信域名；reduce_op 当前仅 sum；comm_turn 仅 0；
    x1 在 aclnn 文档中要求不转置，故 is_trans_a=True 时抛错。
    """
    del group

    if reduce_op.lower() != "sum":
        raise ValueError("matmul_reduce_scatter currently only supports reduce_op='sum'.")
    if comm_turn != 0:
        raise ValueError("matmul_reduce_scatter currently only supports comm_turn=0.")
    if is_trans_a:
        raise ValueError("aclnnMatmulReduceScatter does not support transposed x1 (is_trans_a must be False).")

    x1_t = torch.as_tensor(x1)
    x2_t = torch.as_tensor(x2)

    if x1_t.shape == torch.Size([]):
        return torch.tensor([], dtype=torch.float32)

    if x1_t.dim() != 2 or x2_t.dim() != 2:
        raise ValueError("matmul_reduce_scatter_golden expects 2-D x1 and x2.")

    x1_f = x1_t.to(torch.float32)
    x2_f = x2_t.to(torch.float32)
    if is_trans_b:
        x2_f = x2_f.t()

    out = torch.matmul(x1_f, x2_f)

    if bias is not None:
        b = torch.as_tensor(bias, dtype=torch.float32)
        if b.numel() > 0:
            out = out + b.reshape(-1)

    if rank_size > 0:
        world_size = int(rank_size)
    else:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    if world_size < 1:
        raise ValueError("rank_size / world_size must be positive.")

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(kwargs.get("rank", 0))

    if out.shape[0] % world_size != 0:
        raise ValueError("output row count (m) must be divisible by rank_size/world_size.")

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

    scatter_m = out.shape[0] // world_size
    return out.narrow(0, rank * scatter_m, scatter_m)
