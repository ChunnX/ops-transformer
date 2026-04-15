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
            "quant_reduce_scatter": "quant_reduce_scatter_golden"
        }
}

import numpy as np
import torch
import torch.distributed as dist


def _cpu_dequant(x, scale, group_size: int):
    """对齐 MC2_test QuantReduceScatter.cpu_dequant。"""
    x_np = torch.as_tensor(x, dtype=torch.float32).numpy()
    scale_np = torch.as_tensor(scale, dtype=torch.float32).numpy()
    repeated_scale = np.repeat(scale_np, group_size, axis=-1)
    x_t = torch.from_numpy(x_np).to(torch.float32)
    rs_t = torch.from_numpy(repeated_scale).to(torch.float32)
    return x_t * rs_t


def _infer_repeat_group_size(x, scales, is_mxfp):
    if is_mxfp is not None:
        return 32 if is_mxfp else 128
    x_t = torch.as_tensor(x)
    s_t = torch.as_tensor(scales)
    if x_t.dim() < 2 or s_t.dim() < 2:
        raise ValueError("quant_reduce_scatter_golden expects x and scales with at least 2 dimensions.")
    h = x_t.shape[-1]
    s_last = s_t.shape[-1]
    if s_last == (h + 31) // 32:
        return 32
    if s_last == (h + 127) // 128:
        return 128
    raise ValueError(
        "Cannot infer mx vs pertoken scale layout from shapes; pass is_mxfp=True/False in kwargs."
    )


def _reduce_op_from_str(name: str):
    m = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    op = m.get(name.lower())
    if op is None:
        raise ValueError(f"Unsupported reduce_op={name!r}; expected one of {list(m.keys())}.")
    return op


def _cast_output_dtype(t: torch.Tensor, output_dtype: int) -> torch.Tensor:
    """与 def.cpp 默认 BF16 对齐；27 按工程约定为 ge::DT_BF16。"""
    if output_dtype == 27:
        return t.to(torch.bfloat16)
    return t


def quant_reduce_scatter_golden(x, scales, group, world_size, reduce_op: str = 'sum',
                                output_dtype: int = 27, **kwargs):  # output_dtype: ge::DT_BF16 等
    """
    CPU 参考：反量化后对全精度张量 all_reduce，再按行做 ReduceScatter 切片（与 get_cpu 一致）。

    README：等价于 AllToAll 后 Reduce；MC2_test get_cpu 采用先 dequant、all_reduce、再 narrow(0, ...)。
    """
    del group

    x_t = torch.as_tensor(x)
    if x_t.numel() == 0:
        raise ValueError("quant_reduce_scatter does not support empty tensor input.")

    if dist.is_available() and dist.is_initialized():
        if int(world_size) != dist.get_world_size():
            raise ValueError(
                f"world_size attribute ({world_size}) must match torch.distributed world size "
                f"({dist.get_world_size()})."
            )

    is_mxfp = kwargs.get("is_mxfp", None)
    gs = _infer_repeat_group_size(x, scales, is_mxfp)
    out = _cpu_dequant(x, scales, gs)

    op = _reduce_op_from_str(reduce_op)

    if dist.is_available() and dist.is_initialized() and int(world_size) > 1:
        dist.all_reduce(out, op=op)
    elif int(world_size) > 1:
        pass

    ws = int(world_size)
    if ws < 1:
        raise ValueError("world_size must be positive.")

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(kwargs.get("rank", 0))

    if out.shape[0] % ws != 0:
        raise ValueError("Leading dimension of dequantized tensor must be divisible by world_size.")

    scatter_m = out.shape[0] // ws
    out = out.narrow(0, rank * scatter_m, scatter_m)

    return _cast_output_dtype(out, output_dtype)
