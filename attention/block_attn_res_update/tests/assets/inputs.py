#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stable ops-test-kit inputs for BlockAttnResUpdate.

Kernel and ACLNN expose the same tensors in different orders.  The public
hooks below preserve their respective ABI orders and map both to the common
input generator.

The generated historical state is deliberately finite and internally
consistent: ``inter_exp_sum`` is positive and ``inter_numerator`` is a
weighted numerator whose magnitude is bounded by that denominator.  This
keeps the online-softmax merge away from overflow/underflow while still
exercising both the historical and current branches.
"""

import numpy as np


__input__ = {
    "e2e": {
        "block_attn_res_update_ttk_ops.block_attn_res_update_ttk": "block_attn_res_update_e2e_input",
        "cann_ops_transformer.block_attn_res_update": "block_attn_res_update_e2e_input",
    },
    "kernel": {
        "block_attn_res_update": "block_attn_res_update_input",
    },
    "aclnn": {
        "aclnnBlockAttnResUpdate": "aclnn_block_attn_res_update_input",
    },
}


def _to_numpy(tensor):
    """Return a host FP32 view/copy suitable for constructing input values."""
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "numpy"):
        try:
            return tensor.numpy()
        except (TypeError, RuntimeError):
            # torch.bfloat16 cannot be converted to NumPy directly.
            return tensor.float().numpy()
    return np.asarray(tensor)


def _write_tensor(tensor, value):
    """Write a NumPy value into either a NumPy array or a torch Tensor."""
    value = np.asarray(value)
    if isinstance(tensor, np.ndarray):
        tensor[...] = value.astype(tensor.dtype, copy=False)
        return

    import torch

    source = torch.as_tensor(value, device=tensor.device)
    tensor.copy_(source.reshape(tensor.shape).to(dtype=tensor.dtype))


def _make_stable_inputs(
    partial_block,
    partial_delta,
    effective_query,
    inter_max,
    inter_exp_sum,
    inter_numerator,
):
    shape = _to_numpy(partial_block).shape
    if len(shape) != 2:
        raise ValueError(f"partial_block must have shape [T, D], got {shape}")
    token_num, hidden_size = shape
    if _to_numpy(partial_delta).shape != shape:
        raise ValueError("partial_delta shape must equal partial_block shape")
    if _to_numpy(inter_numerator).shape != shape:
        raise ValueError("inter_numerator shape must equal partial_block shape")
    if _to_numpy(effective_query).shape != (hidden_size,):
        raise ValueError("effective_query shape must be [D]")
    if _to_numpy(inter_max).shape != (token_num,):
        raise ValueError("inter_max shape must be [T]")
    if _to_numpy(inter_exp_sum).shape != (token_num,):
        raise ValueError("inter_exp_sum shape must be [T]")

    # Broadcasted row/column patterns are deterministic and avoid allocating an
    # additional arange(T * D) buffer for large cases.
    row = ((np.arange(token_num, dtype=np.float32) % 17.0) - 8.0)[:, None] / 32.0
    col = ((np.arange(hidden_size, dtype=np.float32) % 29.0) - 14.0)[None, :] / 32.0
    partial_value = row + col
    delta_value = 0.25 * (row - col)

    # Unit-L2 query keeps score magnitude roughly independent of D.
    query_value = (np.arange(hidden_size, dtype=np.float32) % 31.0) - 15.0
    query_norm = float(np.linalg.norm(query_value.astype(np.float64)))
    if query_norm == 0.0:
        query_value[...] = 1.0
        query_norm = float(np.sqrt(hidden_size))
    query_value /= np.float32(query_norm)

    history_max = np.linspace(-0.5, 0.5, token_num, dtype=np.float32)
    history_denominator = np.linspace(1.25, 2.25, token_num, dtype=np.float32)
    history_value = np.tanh(row - 0.5 * col).astype(np.float32)
    history_numerator = history_value * history_denominator[:, None]

    _write_tensor(partial_block, partial_value)
    _write_tensor(partial_delta, delta_value)
    _write_tensor(effective_query, query_value)
    _write_tensor(inter_max, history_max)
    _write_tensor(inter_exp_sum, history_denominator)
    _write_tensor(inter_numerator, history_numerator)


def block_attn_res_update_input(
    partial_block,
    delta,
    inter_numerator,
    inter_max,
    inter_exp_sum,
    effective_query,
    eps=1e-6,
    **kwargs,
):
    """Kernel hook; parameters follow block_attn_res_update_def.cpp."""
    del kwargs
    if not np.isfinite(eps) or float(eps) <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    _make_stable_inputs(
        partial_block,
        delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
    )
    return (
        partial_block,
        delta,
        inter_numerator,
        inter_max,
        inter_exp_sum,
        effective_query,
    )


def block_attn_res_update_e2e_input(
    partial_block,
    partial_delta,
    effective_query,
    inter_max,
    inter_exp_sum,
    inter_numerator,
    epsilon,
    **kwargs,
):
    """E2E hook; parameters follow the public Torch wrapper."""
    del kwargs
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    _make_stable_inputs(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
    )


def aclnn_block_attn_res_update_input(
    partialBlock,
    delta,
    interNumerator,
    interMax,
    interExpSum,
    effectiveQuery,
    eps,
    h,
    **kwargs,
):
    """ACLNN input hook; ops-test-kit consumes the in-place mutations."""
    del h, kwargs
    if not np.isfinite(eps) or float(eps) <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    _make_stable_inputs(
        partialBlock,
        delta,
        effectiveQuery,
        interMax,
        interExpSum,
        interNumerator,
    )
    return (
        partialBlock,
        delta,
        interNumerator,
        interMax,
        interExpSum,
        effectiveQuery,
    )
