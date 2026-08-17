#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MoeTutelCombineX 多通路 golden（TestSpec 范式）。

通路支持面（照抄 01_requirement.md §3.3 / §5，与下方 __spec__ 注册项逐条相等）：

| 通路   | 支持 | 依据                                                          |
|--------|------|---------------------------------------------------------------|
| kernel | 是   | op_kernel/arch35/ 有 Ascend950 regbase 实现                   |
| geir   | 是   | op_graph/moe_tutel_combine_x_proto.h 有 REG_OP(MoeTutelCombineX) |
| aclnn  | 否   | 无 op_api/、无 docs/aclnnMoeTutelCombineX.md，canndev 侧亦无   |
| e2e    | 否   | torch_npu 二进制无 aclnnMoeTutelCombineX 符号                 |

语义：
    valid(k, s) = indices[k, s] >= 0 and locations[k, s] < C
    x_grad[s, :] = sum_k valid(k, s) ? gates[k, s] * y_grad[indices[k, s], locations[k, s], :] : 0
累加从 0 起，K = 0 或全部路由无效时输出为 0。
"""

import numpy as np
import torch

__spec__ = {
    # kernel + geir 共用同一个注册键（算子蛇形名），geir 不另写。
    "moe_tutel_combine_x": "MoeTutelCombineXKernelSpec",
}

# ── 判据 tolerance ───────────────────────────────────────────────────────────
# 浮点输出全部配 cross_check：它是唯一会去拉三方数据判精度的标准，不配则
# third_party 写了也不会被取用，三方精度那一栏永远是空的。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(np.ascontiguousarray(value))


def _float_workspace(tensor):
    """浮点输入的工作精度：只向上兜底，绝不向下砍。

    tolerance 配 cross_check 时框架按 golden_mode=Promote 把输入抬一档喂进来
    （fp16/bf16 -> fp32、fp32 -> fp64），此处保持不动；只有 CPU 上残缺的
    half/bfloat16 才抬到 fp32。
    """
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


def _compute(y_grad, gates, indices, locations, **kwargs):
    """torch 竞品接口实现（R3：禁 numpy 纯公式）。

    y_grad   : [E, C, H]
    gates    : [K, S]
    indices  : [K, S] int32
    locations: [K, S] int32
    返回     : [list([S, H])]，顺序照 def.cpp 的输出序。
    """
    del kwargs
    y_grad = _float_workspace(_to_tensor(y_grad))
    gates = _float_workspace(_to_tensor(gates))
    indices = _to_tensor(indices).to(torch.int64)
    locations = _to_tensor(locations).to(torch.int64)

    capacity = y_grad.shape[1]
    hidden = y_grad.shape[2]
    samples = gates.shape[1]
    work_dtype = y_grad.dtype

    out = torch.zeros((samples, hidden), dtype=work_dtype)
    if gates.shape[0] == 0 or samples == 0 or hidden == 0:
        return [out]

    flat = y_grad.reshape(-1, hidden)
    valid = torch.logical_and(indices >= 0, locations < capacity)
    # 无效项的行号先夹到 0，避免 index_select 报越界；其贡献随后被 where 抹成 0。
    row = torch.where(valid, indices * capacity + locations, torch.zeros_like(indices))
    zero_row = torch.zeros((samples, hidden), dtype=work_dtype)

    for k in range(gates.shape[0]):
        gathered = torch.index_select(flat, 0, row[k])
        weighted = torch.mul(gathered, gates[k].reshape(-1, 1))
        # 乘法与加法分两步写，不用 addcmul 之类的融合算子（FMA 会改运算序列）。
        out = torch.add(out, torch.where(valid[k].reshape(-1, 1), weighted, zero_row))
    return [out]


class _Compose:
    """竞品标杆：用与 _compute 不同的 torch 路径拼出同一语义。

    _compute 走 index_select + 逐 k 循环累加；这里走高级索引一次取出 [K, S, H]
    再用 masked_fill + sum 一次归约，两条路径相互独立，避免"同错一起过"。
    输出必须 cast 回 NPU 的输出 dtype，否则竞品对 golden 的误差远小于 NPU，
    cross_check 的 ratio 会凭空爆表。
    """

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, y_grad, gates, indices, locations, **kwargs):
        del kwargs
        y_grad_t = _to_tensor(y_grad)
        out_dtype = y_grad_t.dtype
        work = _float_workspace(y_grad_t)
        gates_t = _float_workspace(_to_tensor(gates))
        indices_t = _to_tensor(indices).to(torch.int64)
        locations_t = _to_tensor(locations).to(torch.int64)

        capacity = work.shape[1]
        hidden = work.shape[2]
        samples = gates_t.shape[1]
        if gates_t.shape[0] == 0 or samples == 0 or hidden == 0:
            return [torch.zeros((samples, hidden), dtype=out_dtype)]

        valid = torch.logical_and(indices_t >= 0, locations_t < capacity)
        safe_expert = torch.where(valid, indices_t, torch.zeros_like(indices_t))
        safe_loc = torch.where(valid, locations_t, torch.zeros_like(locations_t))
        gathered = work[safe_expert, safe_loc]  # [K, S, H]
        scaled = gathered * gates_t.unsqueeze(-1)  # [K, S, H]
        masked = scaled.masked_fill(~valid.unsqueeze(-1), 0)
        return [masked.sum(dim=0).to(out_dtype)]


class MoeTutelCombineXKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。

    入参名取自 op_host/moe_tutel_combine_x_def.cpp（无属性，全部按位置传）。
    """

    def golden(*inputs, **kwargs):
        outs = _compute(*inputs, **kwargs)
        out_dtypes = kwargs.get("output_dtypes") or []
        out_dtypes = [
            d[0] if isinstance(d, (list, tuple)) else str(d) for d in out_dtypes
        ]
        result = []
        for i, o in enumerate(outs):
            arr = o.numpy() if isinstance(o, torch.Tensor) else np.asarray(o)
            result.append(arr.astype(out_dtypes[i]) if i < len(out_dtypes) else arr)
        return result

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路：本算子无 op_host/op_api/、无 docs/aclnnMoeTutelCombineX.md，
#   canndev 侧同样没有同名 aclnn，README「调用说明」只列图模式调用。
# 【不存在】e2e 通路：strings libtorch_npu.so | grep -c aclnnMoeTutelCombineX 结果为 0
#   （对照组 aclnnMoeFinalizeRouting/V2/V4 有命中，证明查法有效）。
