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
from typing import Optional, Tuple, Union

import torch
from torch import _check as torch_check
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

DEFAULT_NORM_EPS = 1.0e-5
PARTIAL_BLOCK_RANK = 2
BLOCK_RES_RANK = 3
NORM_WEIGHT_RANK = 1
T_DIM_INDEX = 0
N_DIM_INDEX = 1
H_DIM_INDEX = 1
BLOCK_RES_H_DIM_INDEX = 2
MIN_HIDDEN_SIZE = 1
MIN_BLOCK_NUM = 1
MAX_BLOCK_NUM = 100
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _resolve_need_backward(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    need_backward: Optional[bool],
) -> bool:
    if need_backward is not None:
        return need_backward
    return all(
        t.requires_grad for t in (partial_block, block_res, proj_weight, norm_weight)
    )


def _resolve_valid_block_num(
    block_res: torch.Tensor, valid_block_num: Optional[int]
) -> int:
    if valid_block_num is None or valid_block_num < 0:
        return block_res.size(N_DIM_INDEX)
    return valid_block_num


def _check_inputs(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    valid_block_num: int,
    norm_eps: float,
) -> None:
    torch_check(
        partial_block.dim() == PARTIAL_BLOCK_RANK,
        lambda: f"partial_block must be a 2D tensor, but got {partial_block.dim()}D",
    )
    torch_check(
        block_res.dim() == BLOCK_RES_RANK,
        lambda: f"block_res must be a 3D tensor, but got {block_res.dim()}D",
    )
    torch_check(
        norm_weight.dim() == NORM_WEIGHT_RANK,
        lambda: f"norm_weight must be a 1D tensor, but got {norm_weight.dim()}D",
    )
    torch_check(
        proj_weight.dim() == 1 or (proj_weight.dim() == 2 and proj_weight.size(0) == 1),
        lambda: f"proj_weight must be [H] or [1,H], but got {tuple(proj_weight.shape)}",
    )

    compute_dtype = partial_block.dtype
    torch_check(
        compute_dtype in SUPPORTED_DTYPES,
        lambda: (
            "partial_block dtype must be float16, bfloat16 or float32, "
            f"but got {compute_dtype}"
        ),
    )
    for tensor_name, tensor in (
        ("block_res", block_res),
        ("proj_weight", proj_weight),
        ("norm_weight", norm_weight),
    ):
        torch_check(
            tensor.dtype == compute_dtype,
            lambda tensor_name=tensor_name, tensor=tensor: (
                f"{tensor_name} dtype must match partial_block ({compute_dtype}), "
                f"but got {tensor.dtype}"
            ),
        )

    torch_check(
        partial_block.device
        == block_res.device
        == proj_weight.device
        == norm_weight.device,
        lambda: (
            "all inputs must be on the same device, but got "
            f"partial_block={partial_block.device}, block_res={block_res.device}, "
            f"proj_weight={proj_weight.device}, norm_weight={norm_weight.device}"
        ),
    )

    num_tokens = partial_block.size(T_DIM_INDEX)
    hidden_size = partial_block.size(H_DIM_INDEX)
    num_blocks = block_res.size(N_DIM_INDEX)
    torch_check(
        num_tokens >= 0
        and hidden_size >= MIN_HIDDEN_SIZE
        and MIN_BLOCK_NUM <= num_blocks <= MAX_BLOCK_NUM,
        lambda: (
            f"shape range requires T>=0, H>=1 and 1<=N<={MAX_BLOCK_NUM}, "
            f"but got T={num_tokens}, H={hidden_size}, N={num_blocks}"
        ),
    )
    torch_check(
        block_res.size(T_DIM_INDEX) == num_tokens
        and block_res.size(BLOCK_RES_H_DIM_INDEX) == hidden_size,
        lambda: f"block_res shape must be [T,N,H], but got {tuple(block_res.shape)}",
    )
    torch_check(
        proj_weight.size(-1) == hidden_size,
        lambda: (
            f"proj_weight last dim must equal H={hidden_size}, "
            f"but got {tuple(proj_weight.shape)}"
        ),
    )
    torch_check(
        norm_weight.size(0) == hidden_size,
        lambda: f"norm_weight must be [H], but got {tuple(norm_weight.shape)}",
    )
    torch_check(
        valid_block_num == num_blocks,
        lambda: f"only default valid_block_num=N is supported, got {valid_block_num}",
    )
    torch_check(
        math.isfinite(norm_eps) and norm_eps > 0.0,
        lambda: f"norm_eps must be finite and greater than zero, but got {norm_eps}",
    )


class _BlockAttentionResidualsOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_attention_residuals", category="mhc")

    def sources(self):
        return ["csrc/mhc/block_attention_residuals.cpp"]

    def schema(self) -> str:
        return (
            "block_attention_residuals(Tensor partial_block, Tensor block_res, "
            "Tensor proj_weight, Tensor norm_weight, *, int? valid_block_num=None, "
            f"float norm_eps={DEFAULT_NORM_EPS}, bool? need_backward=None) "
            "-> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def block_attention_residuals_meta(
            partial_block,
            block_res,
            proj_weight,
            norm_weight,
            *,
            valid_block_num=None,
            norm_eps=DEFAULT_NORM_EPS,
            need_backward=None,
        ):
            valid_block_num = _resolve_valid_block_num(block_res, valid_block_num)
            need_backward = _resolve_need_backward(
                partial_block, block_res, proj_weight, norm_weight, need_backward
            )
            _check_inputs(
                partial_block,
                block_res,
                proj_weight,
                norm_weight,
                valid_block_num,
                norm_eps,
            )
            num_tokens = partial_block.size(T_DIM_INDEX)
            hidden_size = partial_block.size(H_DIM_INDEX)
            block_count = block_res.size(N_DIM_INDEX) + 1
            hidden = torch.empty(
                num_tokens, hidden_size, dtype=partial_block.dtype, device="meta"
            )
            if need_backward:
                inv_norm = torch.empty(
                    num_tokens, block_count, dtype=torch.float32, device="meta"
                )
                probs = torch.empty(
                    num_tokens, block_count, dtype=torch.float32, device="meta"
                )
            else:
                inv_norm = torch.empty(0, dtype=torch.float32, device="meta")
                probs = torch.empty(0, dtype=torch.float32, device="meta")
            return hidden, inv_norm, probs


_block_attention_residuals_op_builder = _BlockAttentionResidualsOpBuilder()
_block_attention_residuals_op_builder.ensure_initialized()


@impl(get_as_library(), _block_attention_residuals_op_builder.name, "PrivateUse1")
def _block_attention_residuals(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    valid_block_num: Optional[int] = None,
    norm_eps: float = DEFAULT_NORM_EPS,
    need_backward: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_block_num = _resolve_valid_block_num(block_res, valid_block_num)
    need_backward = _resolve_need_backward(
        partial_block, block_res, proj_weight, norm_weight, need_backward
    )
    op_module = _block_attention_residuals_op_builder.load()
    return op_module.block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num,
        norm_eps,
        need_backward,
    )


def block_attention_residuals(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    valid_block_num: Optional[int] = None,
    norm_eps: float = DEFAULT_NORM_EPS,
    need_backward: Optional[bool] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Attention Residual 加权融合前向，封装 aclnnBlockAttentionResiduals。

    Args:
        partial_block (Tensor): 拼接后的第 N 行 value，shape 为 ``[T, H]``，
            dtype 支持 float16 / bfloat16 / float32。
        block_res (Tensor): 前 N 行 value，shape 为 ``[T, N, H]``，dtype 与 partial_block 一致。
        proj_weight (Tensor): 投影权重，shape 为 ``[H]`` 或 ``[1, H]``，dtype 与 partial_block 一致。
        norm_weight (Tensor): RMS 缩放权重，shape 为 ``[H]``，dtype 与 partial_block 一致。
        valid_block_num (int, optional): 有效 block 数。不传或传入 ``None`` / ``-1`` 时默认
            为 ``N``；当前仅支持等于 ``N``。
        norm_eps (float): RMS 数值稳定项，必须为有限正数，默认 ``1e-5``。
        need_backward (bool, optional): 是否写出反向中间量。``None`` 时若四个输入均
            ``requires_grad`` 则按 ``True`` 处理。

    Returns:
        need_backward 为 False 时返回 ``hidden_states``，shape ``[T, H]``，dtype 同 partial_block。
        为 True 时返回 ``(hidden_states, inv_norm, probs)``，后两者为 float32，shape ``[T, N+1]``。
    """
    valid_block_num = _resolve_valid_block_num(block_res, valid_block_num)
    need_backward = _resolve_need_backward(
        partial_block, block_res, proj_weight, norm_weight, need_backward
    )
    _check_inputs(
        partial_block, block_res, proj_weight, norm_weight, valid_block_num, norm_eps
    )
    hidden, inv_norm, probs = torch.ops.cann_ops_transformer.block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num=valid_block_num,
        norm_eps=norm_eps,
        need_backward=need_backward,
    )
    if need_backward:
        return hidden, inv_norm, probs
    return hidden
