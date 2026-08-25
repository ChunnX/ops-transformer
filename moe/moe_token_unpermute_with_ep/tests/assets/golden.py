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

import torch

__spec__ = {
    "aclnnMoeTokenUnpermuteWithEp": "AclnnMoeTokenUnpermuteWithEpTestSpec",
}


class AclnnMoeTokenUnpermuteWithEpTestSpec:
    @staticmethod
    def golden(
        permutedTokens,
        sortedIndices,
        probsOptional=None,
        numTopk=1,
        rangeOptional=None,
        paddedMode=False,
        restoreShapeOptional=None,
        out=None,
        *args,
        **kwargs,
    ):
        del paddedMode, restoreShapeOptional, out, args, kwargs

        output_dtype = permutedTokens.dtype
        tokens = permutedTokens.to(torch.float32)
        indices = sortedIndices.to(torch.int64)
        has_range = rangeOptional is not None and len(rangeOptional) == 2

        if probsOptional is not None:
            token_num, topk = probsOptional.shape
        elif has_range:
            topk = numTopk
            token_num = sortedIndices.numel() // topk
        else:
            # rangeOptional=None follows aclnnMoeTokenUnpermute: without probs, K is fixed to one.
            topk = 1
            token_num = sortedIndices.numel()

        hidden_size = tokens.shape[1]
        if has_range:
            start, end = int(rangeOptional[0]), int(rangeOptional[1])
            gathered = torch.zeros(
                (indices.numel(), hidden_size), dtype=torch.float32, device=tokens.device
            )
            valid_mask = (indices >= start) & (indices < end)
            valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if valid_positions.numel() != 0:
                local_indices = indices.index_select(0, valid_positions) - start
                valid_tokens = tokens.index_select(0, local_indices)
                gathered.index_copy_(0, valid_positions, valid_tokens)
        else:
            # No EP range: sortedIndices is a direct gather index and no range filtering is applied.
            gathered = tokens.index_select(0, indices)

        gathered = gathered.reshape(token_num, topk, hidden_size)
        if probsOptional is not None:
            probs = probsOptional.to(torch.float32).unsqueeze(-1)
            gathered = gathered * probs

        return [gathered.sum(dim=1).to(output_dtype)]

    @staticmethod
    def customize_inputs(permutedTokens, sortedIndices, probsOptional=None, *args, **kwargs):
        del args, kwargs
        total_length = sortedIndices.numel()
        # A deterministic reverse permutation covers both in-range and out-of-range slots for partial EP ranges.
        indices = torch.arange(
            total_length - 1,
            -1,
            -1,
            dtype=sortedIndices.dtype,
            device=sortedIndices.device,
        )
        sortedIndices.copy_(indices)
        return permutedTokens, sortedIndices, probsOptional

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
