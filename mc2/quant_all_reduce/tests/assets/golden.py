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
            "quant_all_reduce": "quant_all_reduce_golden"
        }
}

import numpy as np
import torch
import torch.distributed as dist


def _cpu_dequant(x, scale, group_size: int):
    """对齐 MC2_test QuantAllReduce.cpu_dequant：沿最后一维 repeat scale 后与 x 逐元素相乘。"""
    x_np = torch.as_tensor(x, dtype=torch.float32).numpy()
    scale_np = torch.as_tensor(scale, dtype=torch.float32).numpy()
    repeated_scale = np.repeat(scale_np, group_size, axis=-1)
    x_t = torch.from_numpy(x_np).to(torch.float32)
    rs_t = torch.from_numpy(repeated_scale).to(torch.float32)
    return x_t * rs_t


def _infer_repeat_group_size(x, scales, is_mxfp):
    """
    mx：README 中 scale 与 x 最后一维 H 的关系对应 repeat=32；
    pertoken-pergroup：repeat=128。
    也可通过 kwargs['is_mxfp'] 显式指定。
    """
    if is_mxfp is not None:
        return 32 if is_mxfp else 128
    x_t = torch.as_tensor(x)
    s_t = torch.as_tensor(scales)
    if x_t.dim() < 2 or s_t.dim() < 2:
        raise ValueError("quant_all_reduce_golden expects x and scales with at least 2 dimensions.")
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


def quant_all_reduce_golden(x, scales, group, reduce_op = 'sum', output_dtype = 27, world_size = None, **kwargs):  # output_dtype: ge::DT_BF16 等
    """
    CPU 参考：反量化（与 README / get_cpu 一致）后对全精度张量做 AllReduce。

    README 公式：先 AllGather 再 Reduce(scale*data)；数学上等价于各 rank 上先 dequant 再 all_reduce。

    output_dtype 与 def.cpp 中可选属性一致；MC2_test get_cpu 保存为 float32，本函数返回 float32。
    """
    del group

    x_t = torch.as_tensor(x)
    if x_t.numel() == 0:
        raise ValueError("quant_all_reduce does not support empty tensor input.")

    if world_size is None:
        raise ValueError("world_size is required.")

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

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=op)
    else:
        # 单机或未初始化进程组：不做跨 rank 规约（与 matmul_all_reduce golden 一致）
        pass

    return out
