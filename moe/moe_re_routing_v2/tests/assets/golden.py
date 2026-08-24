#!/usr/bin/env python3
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

import numpy

__spec__ = {
    "moe_re_routing_v2": "MoeReRoutingV2TestSpec",
}


def moe_re_routing_v2_numpy(
    tokens,
    expert_token_num_per_rank,
    idx_type,
    per_token_scales=None,
    expert_topk_weight=None,
):
    token_num, token_length = tokens.shape
    rank_num, expert_num = expert_token_num_per_rank.shape

    permute_tokens = numpy.zeros((token_num, token_length), dtype=tokens.dtype)
    permute_token_idx = numpy.zeros(token_num, dtype=numpy.int32)

    if per_token_scales is not None:
        if per_token_scales.ndim == 1:
            permute_per_token_scales = numpy.zeros(
                token_num, dtype=per_token_scales.dtype
            )
        else:
            scalar_length = per_token_scales.shape[1]
            permute_per_token_scales = numpy.zeros(
                (token_num, scalar_length), dtype=per_token_scales.dtype
            )
    else:
        permute_per_token_scales = None

    if expert_topk_weight is not None:
        permute_topk_weight = numpy.zeros_like(expert_topk_weight)
    else:
        permute_topk_weight = None

    expert_token_num = numpy.sum(expert_token_num_per_rank, axis=0)
    per_expert_offset = numpy.zeros(expert_num, dtype=expert_token_num_per_rank.dtype)
    per_expert_offset[1:] = numpy.cumsum(expert_token_num)[:-1]

    src_offset = 0
    for cur_rank in range(rank_num):
        for expert in range(expert_num):
            num_tokens = int(expert_token_num_per_rank[cur_rank, expert])
            if num_tokens == 0:
                continue
            dst_start = int(per_expert_offset[expert])
            dst_end = dst_start + num_tokens
            src_end = src_offset + num_tokens

            permute_tokens[dst_start:dst_end] = tokens[src_offset:src_end]
            if idx_type == 0:
                permute_token_idx[dst_start:dst_end] = numpy.arange(
                    src_offset, src_end, dtype=numpy.int32
                )
            else:
                permute_token_idx[src_offset:src_end] = numpy.arange(
                    dst_start, dst_end, dtype=numpy.int32
                )

            if per_token_scales is not None:
                permute_per_token_scales[dst_start:dst_end] = per_token_scales[
                    src_offset:src_end
                ]

            if expert_topk_weight is not None:
                permute_topk_weight[dst_start:dst_end] = expert_topk_weight[
                    src_offset:src_end
                ]

            src_offset = src_end
            per_expert_offset[expert] = dst_end

    return (
        permute_tokens,
        permute_per_token_scales,
        permute_token_idx,
        expert_token_num,
        permute_topk_weight,
    )


class MoeReRoutingV2TestSpec:
    """MoeReRoutingV2 kernel test spec"""

    @staticmethod
    def golden(
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_topk_weight,
        *,
        expert_token_num_type=1,
        idx_type=0,
        **kwargs,
    ):
        idx_type_val = int(idx_type) if idx_type is not None else 0
        per_token_scales_np = per_token_scales if per_token_scales is not None else None
        expert_topk_weight_np = (
            expert_topk_weight if expert_topk_weight is not None else None
        )

        return moe_re_routing_v2_numpy(
            tokens,
            expert_token_num_per_rank,
            idx_type_val,
            per_token_scales_np,
            expert_topk_weight_np,
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
