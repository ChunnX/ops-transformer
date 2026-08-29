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

Contract:

    x          : [S, H]      upstream gradient of the combine output
    y_grad     : [E, C, H]   the dispatched activations saved by the forward
    indices    : [K, S]      expert id per (route, token)
    locations  : [K, S]      slot id inside the expert capacity
    gates_grad : [K, S]

    valid[k,s]      = 0 <= indices[k,s] < E and 0 <= locations[k,s] < C
    gates_grad[k,s] = dot(x[s,:], y_grad[indices[k,s], locations[k,s], :]) if valid else 0

K is independent of E and satisfies 1 <= K <= E.

The reference math is composed from Torch competitor operators
(``index_select`` / ``mul`` / ``sum`` / ``where``), never a hand-rolled NumPy
formula. Inputs are promoted to FP32 for the multiply/reduce and cast once to
the output dtype, as required by the public numerical contract.
"""

import numpy as np
import torch


__spec__ = {
    # kernel + GEIR share this snake-case key and this one Spec class.
    "moe_tutel_combine_gates": "MoeTutelCombineGatesKernelSpec",
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

    rows = torch.index_select(y_work, 0, safe_flat.reshape(-1))  # [K * S, H]
    tokens = torch.index_select(
        x_work,
        0,
        torch.arange(x.shape[0], dtype=torch.int64, device=x_work.device).repeat(
            indices.shape[0]
        ),
    )
    products = torch.mul(rows, tokens)
    dots = torch.sum(products, dim=-1).reshape(indices.shape)

    return torch.where(valid, dots, torch.zeros_like(dots))


def _compute(x, y_grad, indices, locations, **kwargs):
    """Semantic reference with FP32 accumulation and one output cast."""
    del kwargs  # MoeTutelCombineGates has no attributes.
    out = _combine_gates(x, y_grad, indices, locations, torch.float32)
    return [out.to(dtype=x.dtype)]


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
    """Supplementary Torch competitor, independent from the CPU golden path."""

    def __init__(self, **kwargs):
        del kwargs  # The proto declares no attributes.

    def __call__(self, x, y_grad, indices, locations, **kwargs):
        del kwargs
        experts, capacity, hidden = y_grad.shape
        valid = torch.logical_and(
            torch.logical_and(torch.ge(indices, 0), torch.lt(indices, experts)),
            torch.logical_and(torch.ge(locations, 0), torch.lt(locations, capacity)),
        )

        # Keep this path structurally separate from the CPU golden's flattened
        # index_select/mul/sum implementation. Advanced indexing plus einsum
        # provides an independent native Torch expression
        # of the same public contract without mirroring the NPU reduction tree.
        safe_experts = torch.where(valid, indices, torch.zeros_like(indices)).to(
            torch.int64
        )
        safe_locations = torch.where(valid, locations, torch.zeros_like(locations)).to(
            torch.int64
        )
        if x.shape[0] == 0 or hidden == 0:
            dots = torch.zeros(indices.shape, dtype=torch.float32, device=x.device)
        else:
            rows = y_grad.to(torch.float32)[safe_experts, safe_locations, :]
            dots = torch.einsum("ksh,sh->ks", rows, x.to(torch.float32))
        out = torch.where(valid, dots, torch.zeros_like(dots))
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


# This operator has no ACLNN or torch_npu E2E entry. The supported paths are
# kernel direct invocation and GE graph mode, both covered by the spec above.
