# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


class BlockAttnResUpdateOpBuilder(OpBuilder):
    def __init__(self):
        super(BlockAttnResUpdateOpBuilder, self).__init__("block_attn_res_update")

    def sources(self):
        return ["ops/csrc/block_attn_res_update.cpp"]

    def schema(self) -> str:
        return (
            "block_attn_res_update("
            "Tensor partial_block, Tensor partial_delta, Tensor pseudo_query, "
            "Tensor numerator, Tensor logit_max, Tensor exp_sum, "
            "float epsilon) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def block_attn_res_update_meta(
            partial_block,
            partial_delta,
            pseudo_query,
            numerator,
            logit_max,
            exp_sum,
            epsilon,
        ):
            torch._check(
                partial_block.dtype == torch.float32,
                lambda: f"partial_block must be float32, but got {partial_block.dtype}.",
            )
            torch._check(
                partial_delta.dtype == torch.bfloat16,
                lambda: f"partial_delta must be bfloat16, but got {partial_delta.dtype}.",
            )
            for name, tensor in (
                ("pseudo_query", pseudo_query),
                ("numerator", numerator),
                ("logit_max", logit_max),
                ("exp_sum", exp_sum),
            ):
                torch._check(
                    tensor.dtype == torch.float32,
                    lambda name=name, tensor=tensor: (
                        f"{name} must be float32, but got {tensor.dtype}."
                    ),
                )

            for name, tensor in (
                ("partial_block", partial_block),
                ("partial_delta", partial_delta),
                ("pseudo_query", pseudo_query),
                ("numerator", numerator),
                ("logit_max", logit_max),
                ("exp_sum", exp_sum),
            ):
                torch._check(
                    tensor.is_contiguous(),
                    lambda name=name: f"{name} must be contiguous.",
                )

            torch._check(
                partial_block.dim() == 2,
                lambda: f"partial_block must be 2D, but got {partial_block.dim()}D.",
            )
            torch._check(
                partial_delta.dim() == 2,
                lambda: f"partial_delta must be 2D, but got {partial_delta.dim()}D.",
            )
            torch._check(
                numerator.dim() == 2,
                lambda: f"numerator must be 2D, but got {numerator.dim()}D.",
            )
            torch._check(
                pseudo_query.dim() == 1,
                lambda: f"pseudo_query must be 1D, but got {pseudo_query.dim()}D.",
            )
            torch._check(
                logit_max.dim() == 1,
                lambda: f"logit_max must be 1D, but got {logit_max.dim()}D.",
            )
            torch._check(
                exp_sum.dim() == 1,
                lambda: f"exp_sum must be 1D, but got {exp_sum.dim()}D.",
            )

            torch._check(
                partial_block.shape[0] == partial_delta.shape[0],
                lambda: (
                    "partial_delta token dimension must match partial_block, but got "
                    f"{partial_delta.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[1] == partial_delta.shape[1],
                lambda: (
                    "partial_delta hidden dimension must match partial_block, but got "
                    f"{partial_delta.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[0] == numerator.shape[0],
                lambda: (
                    "numerator token dimension must match partial_block, but got "
                    f"{numerator.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[1] == numerator.shape[1],
                lambda: (
                    "numerator hidden dimension must match partial_block, but got "
                    f"{numerator.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                pseudo_query.shape[0] == partial_block.shape[1],
                lambda: (
                    "pseudo_query length must match the hidden dimension of "
                    f"partial_block, but got {pseudo_query.shape} and "
                    f"{partial_block.shape}."
                ),
            )
            torch._check(
                logit_max.shape[0] == partial_block.shape[0],
                lambda: (
                    "logit_max length must match the token dimension of partial_block, "
                    f"but got {logit_max.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                exp_sum.shape[0] == partial_block.shape[0],
                lambda: (
                    "exp_sum length must match the token dimension of "
                    f"partial_block, but got {exp_sum.shape} and "
                    f"{partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[0] > 0,
                lambda: (
                    "the token dimension T must be greater than 0, but got "
                    f"{partial_block.shape[0]}."
                ),
            )
            torch._check(
                partial_block.shape[1] >= 1,
                lambda: (
                    "the hidden dimension D must be at least 1, but got "
                    f"{partial_block.shape[1]}."
                ),
            )
            torch._check(
                partial_block.shape[1] <= 8192,
                lambda: (
                    "the hidden dimension D must not exceed 8192, but got "
                    f"{partial_block.shape[1]}."
                ),
            )
            torch._check(
                math.isfinite(epsilon) and epsilon > 0.0,
                lambda: f"epsilon must be finite and greater than 0, but got {epsilon}.",
            )

            h = torch.empty(
                partial_delta.shape, dtype=partial_delta.dtype, device="meta"
            )
            partial_blocks = torch.empty(
                partial_block.shape, dtype=torch.float32, device="meta"
            )
            return (h, partial_blocks)


_block_attn_res_update_op_builder = BlockAttnResUpdateOpBuilder()


@impl(AS_LIBRARY, _block_attn_res_update_op_builder.name, "PrivateUse1")
def _block_attn_res_update(
    partial_block,
    partial_delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    epsilon,
):
    op_module = _block_attn_res_update_op_builder.load()
    return op_module.block_attn_res_update(
        partial_block,
        partial_delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        epsilon,
    )


def block_attn_res_update(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    pseudo_query: torch.Tensor,
    numerator: torch.Tensor,
    logit_max: torch.Tensor,
    exp_sum: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BlockAttnRes Phase 2 更新，封装 aclnnBlockAttnResUpdate。

    将当前 ``partial_delta`` 累加到 FP32 ``partial_block``，计算当前
    partial block 的 RMSNorm score，并与 Phase 1 的 online softmax
    中间状态合并，输出当前层结果和更新后的 partial block。

    Args:
        partial_block (Tensor): 已累计的 partial block，shape ``(T, D)``，
            dtype 为 ``torch.float32``，且必须连续；``1 <= D <= 8192``。
        partial_delta (Tensor): 当前层增量，shape ``(T, D)``，dtype 为
            ``torch.bfloat16``，且必须连续。
        pseudo_query (Tensor): 预计算的伪 query，shape ``(D,)``，
            dtype 为 ``torch.float32``，且必须连续。
        numerator (Tensor): Phase 1 历史 online softmax 加权和，
            shape ``(T, D)``，dtype 为 ``torch.float32``，且必须连续。
        logit_max (Tensor): Phase 1 历史最大 logit，shape ``(T,)``，
            dtype 为 ``torch.float32``，且必须连续。
        exp_sum (Tensor): Phase 1 历史 softmax 分母累积值，shape ``(T,)``，
            dtype 为 ``torch.float32``，且必须连续。
        epsilon (float): FP32 RMSNorm 数值稳定项，必须为有限正数。

    Returns:
        Tuple[Tensor, Tensor]: ``(h, partial_blocks)``。``h`` 的 shape 和
        dtype 与 ``partial_delta`` 一致；``partial_blocks`` 的 shape 与
        ``partial_block`` 一致，dtype 为 ``torch.float32``；两个输出均为
        连续 Tensor。
    """
    return torch.ops.cann_ops_transformer.block_attn_res_update(
        partial_block,
        partial_delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        epsilon,
    )
