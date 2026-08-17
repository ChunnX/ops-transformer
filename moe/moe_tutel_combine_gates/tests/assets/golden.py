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

"""MoeTutelCombineGates multi-path golden in the TTK TestSpec format.

Contract (frozen from the canndev implementation plus its official UT data
generator; that module ships no InferShape, so the shapes below are the only
authoritative reading):

    x          : [S, H]      upstream gradient of the combine output
    y_grad     : [E, C, H]   the dispatched activations saved by the forward
    indices    : [E, S]      expert id per (route, token)
    locations  : [E, S]      slot id inside the expert capacity
    gates_grad : [E, S]

    valid[e,s]      = 0 <= indices[e,s] < E and 0 <= locations[e,s] < C
    gates_grad[e,s] = dot(x[s,:], y_grad[indices[e,s], locations[e,s], :]) if valid else 0

The routing axis length equals the expert count (K == E): canndev derives its
task count as ``samples * y_grad.shape[0]`` and its UT builds the routing
tensors as ``[batchSize, samples]``.

The reference math is composed from Torch competitor operators
(``index_select`` / ``mul`` / ``sum`` / ``where``), never a hand-rolled NumPy
formula.  As an H-axis reduction the CPU true-value core accumulates in FP64;
the separate third-party path replays the FP32 arithmetic the kernel actually
performs so that cross_check stays algorithm-equivalent.
"""

import numpy as np
import torch


__spec__ = {
    # kernel + GEIR share this snake-case key and this one Spec class.
    "moe_tutel_combine_gates": "MoeTutelCombineGatesKernelSpec",
}

__golden__ = {
    "kernel": {"moe_tutel_combine_gates": "moe_tutel_combine_gates_golden"},
}


_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _numpy_dtype(dtype):
    """Resolve TTK dtype values, including NumPy's optional bfloat16 dtype."""
    name = getattr(dtype, "name", str(dtype)).lower()
    if name in ("bf16", "bfloat16"):
        from ml_dtypes import bfloat16

        return bfloat16
    return np.dtype(name)


def _to_torch(array):
    """Convert NumPy to Torch without changing any payload bits."""
    if isinstance(array, torch.Tensor):
        return array
    array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _to_numpy(tensor, dtype_name=None):
    tensor = tensor.detach().cpu().contiguous()
    if dtype_name is not None:
        target = str(dtype_name).lower()
        if target in ("bf16", "bfloat16"):
            packed = tensor.to(torch.bfloat16).view(torch.int16).numpy()
            return np.ascontiguousarray(packed.view(_numpy_dtype("bfloat16")))
        return np.ascontiguousarray(
            tensor.numpy().astype(_numpy_dtype(target), copy=False)
        )
    if tensor.dtype == torch.bfloat16:
        packed = tensor.view(torch.int16).numpy()
        return np.ascontiguousarray(packed.view(_numpy_dtype("bfloat16")))
    return np.ascontiguousarray(tensor.numpy())


def _output_dtype_names(kwargs):
    output_dtypes = kwargs.get("output_dtypes") or []
    names = []
    for dtype in output_dtypes:
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0] if dtype else None
        if dtype is not None:
            names.append(str(dtype))
    return names


def _route_mask_and_rows(y_grad, indices, locations):
    """Return (valid mask, safe flat row index) built from Torch predicates."""
    experts, capacity = y_grad.shape[0], y_grad.shape[1]
    valid = torch.logical_and(
        torch.logical_and(torch.ge(indices, 0), torch.lt(indices, experts)),
        torch.logical_and(torch.ge(locations, 0), torch.lt(locations, capacity)),
    )
    # Out-of-range routes must never index the gathered tensor; clamp them to row
    # 0 and drop the value afterwards with torch.where.
    flat = torch.add(
        torch.mul(indices.to(torch.int64), capacity), locations.to(torch.int64)
    )
    safe_flat = torch.where(valid, flat, torch.zeros_like(flat))
    return valid, safe_flat


def _combine_gates(x, y_grad, indices, locations, compute_dtype):
    """One shared computation core; only the working precision differs."""
    valid, safe_flat = _route_mask_and_rows(y_grad, indices, locations)
    hidden = x.shape[1]

    x_work = x.to(compute_dtype)
    # Reshape with an explicit row count so that H == 0 (empty inner product)
    # stays legal instead of tripping the ambiguous -1 dimension.
    y_work = y_grad.to(compute_dtype).reshape(y_grad.shape[0] * y_grad.shape[1], hidden)

    rows = torch.index_select(y_work, 0, safe_flat.reshape(-1))  # [E * S, H]
    tokens = torch.index_select(
        x_work, 0, torch.arange(x.shape[0], dtype=torch.int64).repeat(indices.shape[0])
    )
    products = torch.mul(rows, tokens)
    dots = torch.sum(products, dim=-1).reshape(indices.shape)

    return torch.where(valid, dots, torch.zeros_like(dots))


def _compute(x, y_grad, indices, locations, **kwargs):
    """CPU true value: FP64 accumulation over the contracted H axis."""
    del kwargs  # MoeTutelCombineGates has no attributes.
    return [_combine_gates(x, y_grad, indices, locations, torch.float64)]


def _kernel_golden(x, y_grad, indices, locations, **kwargs):
    """Kernel/GEIR container adapter: NumPy in, NumPy list out."""
    outputs = _compute(
        _to_torch(x),
        _to_torch(y_grad),
        _to_torch(indices),
        _to_torch(locations),
        **kwargs,
    )
    output_dtypes = _output_dtype_names(kwargs)
    result = []
    for index, output in enumerate(outputs):
        name = output_dtypes[index] if index < len(output_dtypes) else None
        result.append(_to_numpy(output, name))
    return result


class _MoeTutelCombineGatesCompose:
    """Independent A100 composition replaying the kernel's FP32 arithmetic.

    The CPU golden above stays an FP64 true value.  This third-party path runs
    the same Torch operators in FP32 on the competitor GPU, matching what the
    arch35 kernel does (FP16/BF16 promoted to FP32, one cast back on store), so
    cross_check compares an algorithm-equivalent implementation rather than
    reusing NPU output data.
    """

    def __init__(self, **kwargs):
        del kwargs  # The proto declares no attributes.

    def __call__(self, x, y_grad, indices, locations, **kwargs):
        del kwargs
        out = _combine_gates(x, y_grad, indices, locations, torch.float32)
        # The NPU stores gates_grad in the input floating dtype; align the
        # competitor output the same way (this is the single output-side cast).
        return [out.to(dtype=x.dtype)]


class MoeTutelCombineGatesKernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow def.cpp input order."""

    @staticmethod
    def golden(x, y_grad, indices, locations, **kwargs):
        return _kernel_golden(x, y_grad, indices, locations, **kwargs)

    third_party = {"torch": _MoeTutelCombineGatesCompose}
    tolerance = _TOL


def moe_tutel_combine_gates_golden(x, y_grad, indices, locations, *args, **kwargs):
    """Compatibility entry for the historical ``__golden__`` kernel loader."""
    del args
    legacy_kwargs = dict(kwargs)
    if not legacy_kwargs.get("output_dtypes"):
        legacy_kwargs["output_dtypes"] = [[str(getattr(x, "dtype", "float32"))]]
    return _kernel_golden(x, y_grad, indices, locations, **legacy_kwargs)[0]


# 【不存在】ACLNN 通路：op_host/CMakeLists.txt 显式配置 ACLNNTYPE aclnn_exclude；
# 本算子无同名 public aclnn 接口，也没有 op_host/op_api 与 aclnn 接口文档。
# 【不存在】e2e 通路：不交付 ACLNN 符号，torch_npu 无对应绑定入口，支持面为 Kernel + GE 图模式。
