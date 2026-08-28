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

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

DEFAULT_EPS = 1.0e-6
T_DIM_INDEX = 0
S_DIM_INDEX = 0
D_DIM_INDEX = 2


class _BlockAttnResPrepareOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_attn_res_prepare", category="attention")

    def sources(self):
        return ["csrc/attention/block_attn_res_prepare.cpp"]

    def schema(self) -> str:
        return (
            "block_attn_res_prepare(Tensor block_res, Tensor valid_blocks, "
            "Tensor pseudo_query, *, float eps=0.000001) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def block_attn_res_prepare_meta(
            block_res, valid_blocks, pseudo_query, *, eps=DEFAULT_EPS
        ):
            total_t = block_res.size(T_DIM_INDEX)
            total_s = pseudo_query.size(S_DIM_INDEX)
            total_d = block_res.size(D_DIM_INDEX)
            output_options = {"dtype": torch.float32, "device": "meta"}
            numerator = torch.empty((total_s, total_t, total_d), **output_options)
            logit_max = torch.empty((total_s, total_t), **output_options)
            exp_sum = torch.empty((total_s, total_t), **output_options)
            return numerator, logit_max, exp_sum


_block_attn_res_prepare_op_builder = _BlockAttnResPrepareOpBuilder()
_block_attn_res_prepare_op_builder._ensure_initialized()


@impl(get_as_library(), _block_attn_res_prepare_op_builder.name, "PrivateUse1")
def _block_attn_res_prepare(
    block_res: torch.Tensor,
    valid_blocks: torch.Tensor,
    pseudo_query: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op_module = _block_attn_res_prepare_op_builder.load()
    return op_module.block_attn_res_prepare(block_res, valid_blocks, pseudo_query, eps)


def block_attn_res_prepare(
    block_res: torch.Tensor,
    valid_blocks: torch.Tensor,
    pseudo_query: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Ascend 950 BlockAttnResPrepare Phase 1 operator.

    Args:
        block_res: FP32 tensor with shape ``[T, N, D]``.
        valid_blocks: INT64 device tensor with shape ``[1]``. Values below zero are clipped to zero; zero
            produces the empty online-softmax state, and values above ``N`` are clipped to ``N``.
        pseudo_query: FP32 tensor with shape ``[S, D]``.
        eps: Positive numerical-stability coefficient. Defaults to ``1e-6``.

    Returns:
        ``(numerator, logit_max, exp_sum)`` with shapes
        ``[S, T, D]``, ``[S, T]``, and ``[S, T]`` respectively; all outputs are FP32.
    """
    return torch.ops.cann_ops_transformer.block_attn_res_prepare(
        block_res, valid_blocks, pseudo_query, eps=eps
    )
