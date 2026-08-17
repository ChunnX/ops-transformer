#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MoeTutelDispatch 多通路 golden（TTK TestSpec 范式）。

通路支持面（与 01_requirement.md §3.3 / §5 逐条一致）：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/moe_tutel_dispatch.cpp + op_kernel/arch35/ 有 arch35 实现 |
| geir   | ✅   | REG_OP(MoeTutelDispatch) 在 common/include/op_graph/op_transformer_proto_extend.h； |
|        |      | op_host/moe_tutel_dispatch_infershape.cpp + op_graph/moe_tutel_dispatch_graph_plugin.cpp |
| aclnn  | ❌   | 无同名 public aclnn（canndev ops/built-in/op_api/ grep 无命中），CMake 显式 ACLNNTYPE aclnn_exclude |
| e2e    | ❌   | strings libtorch_npu.so | grep -ic moetutel → 0，torch_npu 无本算子绑定 |

语义：按 (indices, locations) 把 x 的整行 scatter 到 y[expert][slot]；gates 只作零/非零掩码，
不参与乘法；命中槽位是 bit 级拷贝；未命中槽位为 0。
"""

__spec__ = {
    # kernel 与 geir 共用同一个注册键（算子蛇形名）与同一个 TestSpec
    "moe_tutel_dispatch": "MoeTutelDispatchKernelSpec",
}

# 保留仓内既有的单函数注册入口（与 __spec__ 同一份计算），供只认 __golden__ 的调用方使用。
__golden__ = {
    "kernel": {"moe_tutel_dispatch": "moe_tutel_dispatch_golden"},
}

import numpy as np
import torch


# 纯 bit 搬运算子：判据是逐位相等，任何非零容差都会掩盖真缺陷。
# 也因此不配 cross_check —— 三方精度对 bit-exact 拷贝没有判别力（详见 02_design §9）。
_TOL = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
}


def _attr(kwargs, name, default):
    """CSV 里的 attributes 都是字符串，统一转成 default 的类型。"""
    value = kwargs.get(name, default)
    if isinstance(value, str):
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default
    return value


def _numpy_dtype(dtype):
    """解析 TTK 传来的 dtype 值，含 numpy 没有原生支持的 bfloat16。"""
    name = getattr(dtype, "name", str(dtype)).lower()
    if name in ("bf16", "bfloat16"):
        from ml_dtypes import bfloat16

        return bfloat16
    return np.dtype(name)


def _to_torch_bit_exact(array):
    """numpy -> torch，不改动任何 payload 位。"""
    array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _to_numpy_bit_exact(tensor):
    """torch -> numpy，保留 BF16 的 payload 位。"""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_numpy_dtype("bfloat16"))
    return tensor.numpy()


def _compute(x, gates, indices, locations, capacity):
    """torch.Tensor 进、list[torch.Tensor] 出，顺序照 moe_tutel_dispatch_def.cpp 的输出序。

    用 torch 的 index_put_ 竞品接口完成 scatter，不手写元素循环、不复刻内核的分块结构。
    """
    expert_num, _ = gates.shape
    hidden = x.shape[1]
    y = torch.zeros((expert_num, int(capacity), hidden), dtype=x.dtype)
    if int(capacity) == 0 or hidden == 0 or expert_num == 0:
        return [y]

    loc = locations.to(torch.int64)
    idx = indices.to(torch.int64)
    valid = (
        (loc >= 0)
        & (loc < int(capacity))
        & (idx >= 0)
        & (idx < expert_num)
        & (
            gates != 0
        )  # ±0.0 判零，NaN / Inf / denormal 判非零，与 kernel 的按位判定一致
    )
    b_sel, s_sel = torch.nonzero(valid, as_tuple=True)
    if b_sel.numel() == 0:
        return [y]

    y.index_put_((idx[b_sel, s_sel], loc[b_sel, s_sel]), x[s_sel], accumulate=False)
    return [y]


class _Compose:
    """竞品标杆：用 torch 的布尔掩码 + 高级索引赋值拼出等价语义。

    与 _compute 走不同的 torch 路径（__setitem__ 而非 index_put_），避免两边同错。
    参数名与 REG_OP(MoeTutelDispatch) 的注册名逐字一致。
    """

    def __init__(self, **kwargs):
        self.capacity = _attr(kwargs, "capacity", 0)

    def __call__(self, x, gates, indices, locations, **kwargs):
        capacity = int(self.capacity)
        expert_num = gates.shape[0]
        hidden = x.shape[1]
        out = torch.zeros(
            (expert_num, capacity, hidden), dtype=x.dtype, device=x.device
        )
        if capacity == 0 or hidden == 0 or expert_num == 0:
            return [out]

        loc = locations.to(torch.int64)
        idx = indices.to(torch.int64)
        keep = torch.logical_and(
            torch.logical_and(loc >= 0, loc < capacity),
            torch.logical_and(idx >= 0, idx < expert_num),
        )
        keep = torch.logical_and(keep, gates != 0)
        if not bool(keep.any()):
            return [out]

        flat_dst = (idx * capacity + loc)[keep]
        src_row = torch.arange(gates.shape[1], device=x.device).expand_as(loc)[keep]
        out.view(expert_num * capacity, hidden)[flat_dst] = x[src_row]
        return [out]


class MoeTutelDispatchKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。

    参数名取自 op_host/moe_tutel_dispatch_def.cpp：x / gates / indices / locations / capacity。
    """

    def golden(x, gates, indices, locations, **kwargs):
        capacity = _attr(kwargs, "capacity", 0)
        tensors = [_to_torch_bit_exact(a) for a in (x, gates, indices, locations)]
        outs = _compute(*tensors, capacity=capacity)

        out_dtypes = kwargs.get("output_dtypes") or []
        out_dtypes = [d[0] if isinstance(d, (list, tuple)) else d for d in out_dtypes]
        result = []
        for i, o in enumerate(outs):
            arr = _to_numpy_bit_exact(o)
            if i < len(out_dtypes):
                arr = arr.astype(_numpy_dtype(out_dtypes[i]), copy=False)
            result.append(arr)
        return result

    third_party = {"torch": _Compose}
    tolerance = _TOL


def moe_tutel_dispatch_golden(x, gates, indices, locations, **kwargs):
    """__golden__ 的单函数入口，与 MoeTutelDispatchKernelSpec.golden 同一份计算。"""
    return MoeTutelDispatchKernelSpec.golden(x, gates, indices, locations, **kwargs)[0]


# 【不存在】aclnn 通路：canndev 无同名 public aclnn，本算子 CMakeLists.txt 显式配置
#           add_modules_sources(... ACLNNTYPE aclnn_exclude)，不交付 aclnn 符号。
# 【不存在】e2e 通路：torch_npu 二进制无本算子 aclnn 符号
#           （strings .../libtorch_npu.so | grep -ic moetutel → 0），无 torch 侧入口。
