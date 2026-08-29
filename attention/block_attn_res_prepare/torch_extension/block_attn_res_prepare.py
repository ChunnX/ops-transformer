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

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

DEFAULT_EPS = 1.0e-6
BLOCK_RES_RANK = 3
VALID_BLOCKS_RANK = 1
PSEUDO_QUERY_RANK = 2
T_DIM_INDEX = 0
N_DIM_INDEX = 1
S_DIM_INDEX = 0
D_DIM_INDEX = 2
PSEUDO_QUERY_D_DIM_INDEX = 1
VALID_BLOCKS_VALUE_DIM_INDEX = 0
MIN_BLOCK_NUM = 1
MAX_BLOCK_NUM = 64
MIN_HEAD_DIM = 1
MAX_HEAD_DIM = 8192


def _check_inputs(
    block_res: torch.Tensor,
    valid_blocks: torch.Tensor,
    pseudo_query: torch.Tensor,
    eps: float,
) -> None:
    torch._check(
        block_res.dim() == BLOCK_RES_RANK,
        lambda: f"block_res must be a 3D tensor, but got {block_res.dim()}D",
    )
    torch._check(
        valid_blocks.dim() == VALID_BLOCKS_RANK,
        lambda: f"valid_blocks must be a 1D tensor, but got {valid_blocks.dim()}D",
    )
    torch._check(
        pseudo_query.dim() == PSEUDO_QUERY_RANK,
        lambda: f"pseudo_query must be a 2D tensor, but got {pseudo_query.dim()}D",
    )
    torch._check(
        valid_blocks.size(VALID_BLOCKS_VALUE_DIM_INDEX) == 1,
        lambda: f"valid_blocks must have shape [1], but got {valid_blocks.shape}",
    )
    torch._check(
        block_res.size(N_DIM_INDEX) >= MIN_BLOCK_NUM,
        lambda: (
            f"block_res.size(1) must be in [{MIN_BLOCK_NUM}, {MAX_BLOCK_NUM}], "
            f"but got {block_res.size(N_DIM_INDEX)}"
        ),
    )
    torch._check(
        block_res.size(N_DIM_INDEX) <= MAX_BLOCK_NUM,
        lambda: (
            f"block_res.size(1) must be in [{MIN_BLOCK_NUM}, {MAX_BLOCK_NUM}], "
            f"but got {block_res.size(N_DIM_INDEX)}"
        ),
    )
    torch._check(
        block_res.size(D_DIM_INDEX) >= MIN_HEAD_DIM,
        lambda: (
            f"block_res.size(2) must be in [{MIN_HEAD_DIM}, {MAX_HEAD_DIM}], "
            f"but got {block_res.size(D_DIM_INDEX)}"
        ),
    )
    torch._check(
        block_res.size(D_DIM_INDEX) <= MAX_HEAD_DIM,
        lambda: (
            f"block_res.size(2) must be in [{MIN_HEAD_DIM}, {MAX_HEAD_DIM}], "
            f"but got {block_res.size(D_DIM_INDEX)}"
        ),
    )
    torch._check(
        block_res.size(D_DIM_INDEX) == pseudo_query.size(PSEUDO_QUERY_D_DIM_INDEX),
        lambda: (
            "block_res.size(2) must equal pseudo_query.size(1), but got "
            f"block_res.size(2)={block_res.size(D_DIM_INDEX)} and "
            f"pseudo_query.size(1)={pseudo_query.size(PSEUDO_QUERY_D_DIM_INDEX)}"
        ),
    )
    torch._check(
        block_res.dtype == torch.float32,
        lambda: f"block_res must have dtype torch.float32, but got {block_res.dtype}",
    )
    torch._check(
        valid_blocks.dtype == torch.int64,
        lambda: (
            f"valid_blocks must have dtype torch.int64, but got {valid_blocks.dtype}"
        ),
    )
    torch._check(
        pseudo_query.dtype == torch.float32,
        lambda: (
            f"pseudo_query must have dtype torch.float32, but got {pseudo_query.dtype}"
        ),
    )
    torch._check(
        block_res.device == valid_blocks.device == pseudo_query.device,
        lambda: (
            "all inputs must be on the same device, but got "
            f"block_res={block_res.device}, valid_blocks={valid_blocks.device}, "
            f"pseudo_query={pseudo_query.device}"
        ),
    )
    torch._check(block_res.is_contiguous(), lambda: "block_res must be contiguous")
    torch._check(
        valid_blocks.is_contiguous(), lambda: "valid_blocks must be contiguous"
    )
    torch._check(
        pseudo_query.is_contiguous(), lambda: "pseudo_query must be contiguous"
    )
    torch._check(
        math.isfinite(eps) and eps > 0.0,
        lambda: f"eps must be finite and greater than zero, but got {eps}",
    )


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
            _check_inputs(block_res, valid_blocks, pseudo_query, eps)
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
    _check_inputs(block_res, valid_blocks, pseudo_query, eps)
    return torch.ops.cann_ops_transformer.block_attn_res_prepare(
        block_res, valid_blocks, pseudo_query, eps=eps
    )
