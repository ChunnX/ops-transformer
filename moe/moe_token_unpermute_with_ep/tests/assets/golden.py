#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent Torch-composition references for every delivered test route."""

import torch


__spec__ = {
    "moe_token_unpermute_with_ep": "MoeTokenUnpermuteWithEpKernelSpec",
    "aclnnMoeTokenUnpermuteWithEp": "MoeTokenUnpermuteWithEpAclnnSpec",
    "MoeTokenUnpermuteWithEp": "MoeTokenUnpermuteWithEpGeirSpec",
}


_CROSS_CHECK_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _is_ep_range(range_value):
    return (
        range_value is not None
        and len(range_value) == 2
        and not (int(range_value[0]) == 0 and int(range_value[1]) == 0)
    )


def _golden_compute(permuted_tokens, sorted_indices, probs, num_topk, range_value):
    """Golden chain: index_select/gather, mask, mul and sum, all in Torch."""
    output_dtype = permuted_tokens.dtype
    tokens_fp32 = permuted_tokens.to(torch.float32)
    indices = sorted_indices.to(torch.int64).reshape(-1)
    ep_range = _is_ep_range(range_value)

    if probs is not None:
        token_num = int(probs.shape[0])
        topk = int(probs.shape[1])
    elif ep_range:
        topk = int(num_topk)
    else:
        topk = 1

    if topk <= 0:
        raise ValueError("topk must be greater than 0")
    if indices.numel() % topk != 0:
        raise ValueError("sorted_indices length must be divisible by topk")
    inferred_token_num = indices.numel() // topk
    if probs is not None and token_num != inferred_token_num:
        raise ValueError(
            "sorted_indices length must equal probs.shape[0] * probs.shape[1]"
        )
    token_num = inferred_token_num

    if ep_range:
        start, end = int(range_value[0]), int(range_value[1])
        valid = (indices >= start) & (indices < end)
        local_indices = (indices - start).clamp(
            min=0, max=max(tokens_fp32.shape[0] - 1, 0)
        )
        gathered = torch.index_select(tokens_fp32, 0, local_indices)
    else:
        valid = None
        gathered = torch.index_select(tokens_fp32, 0, indices)

    gathered = gathered.reshape(token_num, topk, tokens_fp32.shape[1])
    if probs is not None:
        gathered = torch.mul(gathered, probs.to(torch.float32).unsqueeze(-1))
    if valid is not None:
        valid = valid.reshape(token_num, topk, 1)
        gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
    return torch.sum(gathered, dim=1).to(output_dtype)


def _third_party_compute(permuted_tokens, sorted_indices, probs, num_topk, range_value):
    """Independent composition used only by the third-party timing/check path."""
    output_dtype = permuted_tokens.dtype
    tokens_fp32 = permuted_tokens.float()
    indices = sorted_indices.long().flatten()
    ep_range = _is_ep_range(range_value)

    if probs is None:
        topk = int(num_topk) if ep_range else 1
    else:
        token_num, topk = (int(probs.shape[0]), int(probs.shape[1]))

    if topk <= 0:
        raise ValueError("topk must be greater than 0")
    if indices.numel() % topk != 0:
        raise ValueError("sorted_indices length must be divisible by topk")
    inferred_token_num = indices.numel() // topk
    if probs is not None and token_num != inferred_token_num:
        raise ValueError(
            "sorted_indices length must equal probs.shape[0] * probs.shape[1]"
        )
    token_num = inferred_token_num

    if ep_range:
        start, end = int(range_value[0]), int(range_value[1])
        valid = (indices >= start) & (indices < end)
        slots = torch.zeros(
            (indices.numel(), tokens_fp32.shape[1]),
            dtype=torch.float32,
            device=tokens_fp32.device,
        )
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        if positions.numel() != 0:
            local = torch.index_select(indices, 0, positions) - start
            selected = torch.index_select(tokens_fp32, 0, local)
            slots.index_copy_(0, positions, selected)
    else:
        valid = None
        slots = torch.index_select(tokens_fp32, 0, indices)

    slots = slots.view(token_num, topk, tokens_fp32.shape[1])
    if probs is not None:
        slots = slots * probs.float().unsqueeze(2)
    if valid is not None:
        valid = valid.view(token_num, topk, 1)
        slots = torch.where(valid, slots, torch.zeros_like(slots))
    return slots.sum(1).to(output_dtype)


def _numpy_route(permuted_tokens, sorted_indices, probs, num_topk, range_value):
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    tokens_t = numpy_to_torch_tensor(permuted_tokens)
    indices_t = numpy_to_torch_tensor(sorted_indices)
    probs_t = numpy_to_torch_tensor(probs) if probs is not None else None
    result = _golden_compute(tokens_t, indices_t, probs_t, num_topk, range_value)
    return torch_to_numpy_tensor(result.detach().cpu())


def moe_token_unpermute_with_ep_kernel_golden(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range=None,
    padded_mode=False,
    restore_shape=None,
    unpermuted_tokens=None,
    **kwargs,
):
    del padded_mode, restore_shape, unpermuted_tokens, kwargs
    return [_numpy_route(permuted_tokens, sorted_indices, probs, num_topk, range)]


def moe_token_unpermute_with_ep_aclnn_golden(
    permutedTokens,
    sortedIndices,
    probsOptional=None,
    numTopk=1,
    rangeOptional=None,
    paddedMode=False,
    restoreShapeOptional=None,
    out=None,
    **kwargs,
):
    del paddedMode, restoreShapeOptional, kwargs
    result = _golden_compute(
        permutedTokens, sortedIndices, probsOptional, numTopk, rangeOptional
    )
    if out is not None:
        result = result.reshape(out.shape).to(out.dtype)
    return [result]


def moe_token_unpermute_with_ep_geir_golden(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range=None,
    padded_mode=False,
    restore_shape=None,
    unpermuted_tokens=None,
    **kwargs,
):
    del padded_mode, restore_shape, unpermuted_tokens, kwargs
    return [_numpy_route(permuted_tokens, sorted_indices, probs, num_topk, range)]


def _customize_inputs(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range_value=None,
    testcase_name="",
):
    del num_topk
    name = testcase_name.lower()
    total = sorted_indices.numel()
    token_rows = int(permuted_tokens.shape[0])

    if "l2_indices_negative" in name:
        indices = torch.full(
            (total,), -1, dtype=sorted_indices.dtype, device=sorted_indices.device
        )
    elif "l2_indices_upper_bound" in name:
        indices = torch.full(
            (total,),
            2134372523,
            dtype=sorted_indices.dtype,
            device=sorted_indices.device,
        )
    elif _is_ep_range(range_value):
        start, end = int(range_value[0]), int(range_value[1])
        candidates = [start, max(start, end - 1)]
        if start > 0:
            candidates.append(start - 1)
        if end < total:
            candidates.append(end)
        values = torch.tensor(
            candidates, dtype=sorted_indices.dtype, device=sorted_indices.device
        )
        indices = values.repeat((total + values.numel() - 1) // values.numel())[:total]
    else:
        indices = torch.arange(
            total - 1, -1, -1, dtype=sorted_indices.dtype, device=sorted_indices.device
        )
        indices.remainder_(max(token_rows, 1))
    sorted_indices.reshape(-1).copy_(indices)

    if "deterministic" in name:
        token_values = torch.arange(
            permuted_tokens.numel(), dtype=torch.float32, device=permuted_tokens.device
        ).reshape(permuted_tokens.shape)
        permuted_tokens.copy_(((token_values % 29) - 14).to(permuted_tokens.dtype) / 8)
        if probs is not None:
            prob_values = torch.arange(
                probs.numel(), dtype=torch.float32, device=probs.device
            ).reshape(probs.shape)
            probs.copy_(((prob_values % 11) + 1).to(probs.dtype) / 16)

    token_flat = permuted_tokens.reshape(-1)
    prob_flat = probs.reshape(-1) if probs is not None else None
    if "tokens_poszero" in name:
        token_flat.fill_(0.0)
    elif "tokens_negzero" in name:
        token_flat.fill_(-0.0)
    elif "tokens_posinf" in name:
        token_flat[0] = float("inf")
    elif "tokens_neginf" in name:
        token_flat[0] = float("-inf")
    elif "tokens_nan" in name:
        token_flat[0] = float("nan")
    elif "tokens_large" in name:
        token_flat.fill_(torch.finfo(permuted_tokens.dtype).max / 8)

    if prob_flat is not None:
        if "probs_poszero" in name:
            prob_flat.fill_(0.0)
        elif "probs_negzero" in name:
            prob_flat.fill_(-0.0)
        elif "probs_posinf" in name:
            prob_flat[0] = float("inf")
        elif "probs_neginf" in name:
            prob_flat[0] = float("-inf")
        elif "probs_nan" in name:
            prob_flat[0] = float("nan")
        elif "probs_large" in name:
            prob_flat.fill_(torch.finfo(probs.dtype).max / 8)


def _customize_numpy_inputs(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range_value=None,
    testcase_name="",
):
    """Customize raw/GEIR arrays through TTK's official Torch boundary."""
    from ttk.utilities.dtypes import numpy_to_torch_tensor

    tokens_t = numpy_to_torch_tensor(permuted_tokens)
    indices_t = numpy_to_torch_tensor(sorted_indices)
    probs_t = numpy_to_torch_tensor(probs) if probs is not None else None
    _customize_inputs(
        tokens_t, indices_t, probs_t, num_topk, range_value, testcase_name
    )


def moe_token_unpermute_with_ep_kernel_inputs(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range=None,
    padded_mode=False,
    restore_shape=None,
    unpermuted_tokens=None,
    testcase_name="",
    **kwargs,
):
    del padded_mode, restore_shape, unpermuted_tokens, kwargs
    _customize_numpy_inputs(
        permuted_tokens, sorted_indices, probs, num_topk, range, testcase_name
    )
    return permuted_tokens, sorted_indices, probs


def moe_token_unpermute_with_ep_aclnn_inputs(
    permutedTokens,
    sortedIndices,
    probsOptional=None,
    numTopk=1,
    rangeOptional=None,
    paddedMode=False,
    restoreShapeOptional=None,
    out=None,
    testcase_name="",
    **kwargs,
):
    del paddedMode, restoreShapeOptional, out, kwargs
    _customize_inputs(
        permutedTokens,
        sortedIndices,
        probsOptional,
        numTopk,
        rangeOptional,
        testcase_name,
    )


def moe_token_unpermute_with_ep_geir_inputs(
    permuted_tokens,
    sorted_indices,
    probs=None,
    num_topk=1,
    range=None,
    padded_mode=False,
    restore_shape=None,
    unpermuted_tokens=None,
    testcase_name="",
    **kwargs,
):
    del padded_mode, restore_shape, unpermuted_tokens, kwargs
    _customize_numpy_inputs(
        permuted_tokens, sorted_indices, probs, num_topk, range, testcase_name
    )
    return permuted_tokens, sorted_indices, probs


class _TorchKernelReference:
    def __call__(
        self,
        permuted_tokens,
        sorted_indices,
        probs=None,
        num_topk=1,
        range=None,
        padded_mode=False,
        restore_shape=None,
        **kwargs,
    ):
        del padded_mode, restore_shape, kwargs
        return [
            _third_party_compute(
                permuted_tokens, sorted_indices, probs, num_topk, range
            )
        ]


class _TorchAclnnReference:
    def __call__(
        self,
        permutedTokens,
        sortedIndices,
        probsOptional=None,
        numTopk=1,
        rangeOptional=None,
        paddedMode=False,
        restoreShapeOptional=None,
        out=None,
        **kwargs,
    ):
        del paddedMode, restoreShapeOptional, out, kwargs
        return [
            _third_party_compute(
                permutedTokens, sortedIndices, probsOptional, numTopk, rangeOptional
            )
        ]


class _TorchGeirReference:
    def __call__(
        self,
        permuted_tokens,
        sorted_indices,
        probs=None,
        num_topk=1,
        range=None,
        padded_mode=False,
        restore_shape=None,
        **kwargs,
    ):
        del padded_mode, restore_shape, kwargs
        return [
            _third_party_compute(
                permuted_tokens, sorted_indices, probs, num_topk, range
            )
        ]


class MoeTokenUnpermuteWithEpKernelSpec:
    golden = staticmethod(moe_token_unpermute_with_ep_kernel_golden)
    customize_inputs = staticmethod(moe_token_unpermute_with_ep_kernel_inputs)
    third_party = {"torch": _TorchKernelReference}
    tolerance = _CROSS_CHECK_L0


class MoeTokenUnpermuteWithEpAclnnSpec:
    golden = staticmethod(moe_token_unpermute_with_ep_aclnn_golden)
    customize_inputs = staticmethod(moe_token_unpermute_with_ep_aclnn_inputs)
    third_party = {"torch": _TorchAclnnReference}
    tolerance = _CROSS_CHECK_L0


class MoeTokenUnpermuteWithEpGeirSpec:
    golden = staticmethod(moe_token_unpermute_with_ep_geir_golden)
    customize_inputs = staticmethod(moe_token_unpermute_with_ep_geir_inputs)
    third_party = {"torch": _TorchGeirReference}
    tolerance = _CROSS_CHECK_L0
