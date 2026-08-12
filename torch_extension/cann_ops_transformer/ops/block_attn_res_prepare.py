# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


class BlockAttnResPrepareOpBuilder(OpBuilder):
    def __init__(self):
        super(BlockAttnResPrepareOpBuilder, self).__init__("block_attn_res_prepare")

    def sources(self):
        return ["ops/csrc/block_attn_res_prepare.cpp"]

    def schema(self) -> str:
        return (
            "block_attn_res_prepare(Tensor block_residual, Tensor effective_query, "
            "Tensor valid_blocks, *, float epsilon=0.000001) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def block_attn_res_prepare_meta(
            block_residual, effective_query, valid_blocks, *, epsilon=1e-6
        ):
            total_t = block_residual.size(0)
            total_s = effective_query.size(0)
            total_d = block_residual.size(2)
            output_options = {"dtype": torch.float32, "device": "meta"}
            softmax_max = torch.empty((total_s, total_t), **output_options)
            weighted_output = torch.empty((total_s, total_t, total_d), **output_options)
            softmax_sum = torch.empty((total_s, total_t), **output_options)
            return softmax_max, weighted_output, softmax_sum


block_attn_res_prepare_op_builder = BlockAttnResPrepareOpBuilder()


@impl(AS_LIBRARY, block_attn_res_prepare_op_builder.name, "PrivateUse1")
def block_attn_res_prepare(
    block_residual: torch.Tensor,
    effective_query: torch.Tensor,
    valid_blocks: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Ascend 950 BlockAttnResPrepare Phase 1 operator.

    Args:
        block_residual: FP32 tensor with shape ``[T, N, D]``.
        effective_query: FP32 tensor with shape ``[S, D]``.
        valid_blocks: INT32 device tensor with shape ``[1]``. Values below zero are clipped to zero; zero
            produces the empty online-softmax state, and values above ``N`` are clipped to ``N``.
        epsilon: Positive numerical-stability coefficient. Defaults to ``1e-6``.

    Returns:
        ``(softmax_max, weighted_output, softmax_sum)`` with shapes
        ``[S, T]``, ``[S, T, D]``, and ``[S, T]`` respectively; all outputs are FP32.
    """
    op_module = block_attn_res_prepare_op_builder.load()
    return op_module.block_attn_res_prepare(
        block_residual, effective_query, valid_blocks, epsilon
    )
