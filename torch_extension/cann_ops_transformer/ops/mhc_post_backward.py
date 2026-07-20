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
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


_TND_RANK = 3
_BSND_RANK = 4


def _check_mhc_post_backward_inputs(grad_output, x, h_res, h_out, h_post):
    supported_dtypes = (torch.float16, torch.bfloat16)

    torch._check(
        x.dim() in (_TND_RANK, _BSND_RANK),
        lambda: f"x must be 3D or 4D, but got {x.dim()}D",
    )
    torch._check(
        grad_output.dim() == x.dim(),
        lambda: (
            f"grad_output and x must have the same rank, but got "
            f"{grad_output.dim()}D and {x.dim()}D"
        ),
    )
    torch._check(
        h_res.dim() == x.dim(),
        lambda: f"h_res and x must have the same rank, but got {h_res.dim()}D and {x.dim()}D",
    )
    torch._check(
        h_out.dim() == x.dim() - 1,
        lambda: f"h_out rank must be {x.dim() - 1}, but got {h_out.dim()}",
    )
    torch._check(
        h_post.dim() == x.dim() - 1,
        lambda: f"h_post rank must be {x.dim() - 1}, but got {h_post.dim()}",
    )

    for tensor, name in (
        (grad_output, "grad_output"),
        (x, "x"),
        (h_res, "h_res"),
        (h_out, "h_out"),
        (h_post, "h_post"),
    ):
        torch._check(tensor.numel() > 0, lambda name=name: f"{name} must not be empty")

    torch._check(
        grad_output.dtype in supported_dtypes,
        lambda: f"grad_output must be float16 or bfloat16, but got {grad_output.dtype}",
    )
    torch._check(
        x.dtype == grad_output.dtype,
        lambda: f"x dtype must match grad_output dtype, but got {x.dtype} and {grad_output.dtype}",
    )
    torch._check(
        h_out.dtype == grad_output.dtype,
        lambda: (
            f"h_out dtype must match grad_output dtype, but got "
            f"{h_out.dtype} and {grad_output.dtype}"
        ),
    )
    torch._check(
        h_res.dtype == torch.float32,
        lambda: f"h_res must be float32, but got {h_res.dtype}",
    )
    torch._check(
        h_post.dtype == torch.float32,
        lambda: f"h_post must be float32, but got {h_post.dtype}",
    )

    for dim in range(x.dim()):
        torch._check(
            grad_output.shape[dim] == x.shape[dim],
            lambda dim=dim: (
                f"grad_output.shape[{dim}] must match x.shape[{dim}], but got "
                f"{grad_output.shape[dim]} and {x.shape[dim]}"
            ),
        )

    leading_dims = x.dim() - 2
    for dim in range(leading_dims):
        torch._check(
            h_res.shape[dim] == x.shape[dim],
            lambda dim=dim: (
                f"h_res.shape[{dim}] must match x.shape[{dim}], but got "
                f"{h_res.shape[dim]} and {x.shape[dim]}"
            ),
        )
        torch._check(
            h_out.shape[dim] == x.shape[dim],
            lambda dim=dim: (
                f"h_out.shape[{dim}] must match x.shape[{dim}], but got "
                f"{h_out.shape[dim]} and {x.shape[dim]}"
            ),
        )
        torch._check(
            h_post.shape[dim] == x.shape[dim],
            lambda dim=dim: (
                f"h_post.shape[{dim}] must match x.shape[{dim}], but got "
                f"{h_post.shape[dim]} and {x.shape[dim]}"
            ),
        )

    torch._check(
        h_res.shape[-2] == x.shape[-2],
        lambda: (
            f"h_res.shape[-2] must match x.shape[-2], but got "
            f"{h_res.shape[-2]} and {x.shape[-2]}"
        ),
    )
    torch._check(
        h_res.shape[-1] == x.shape[-2],
        lambda: (
            f"h_res.shape[-1] must match x.shape[-2], but got "
            f"{h_res.shape[-1]} and {x.shape[-2]}"
        ),
    )
    torch._check(
        h_out.shape[-1] == x.shape[-1],
        lambda: (
            f"h_out.shape[-1] must match x.shape[-1], but got "
            f"{h_out.shape[-1]} and {x.shape[-1]}"
        ),
    )
    torch._check(
        h_post.shape[-1] == x.shape[-2],
        lambda: (
            f"h_post.shape[-1] must match x.shape[-2], but got "
            f"{h_post.shape[-1]} and {x.shape[-2]}"
        ),
    )


class MhcPostBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPostBackwardOpBuilder, self).__init__("mhc_post_backward")

    def sources(self):
        return ["ops/csrc/mhc_post_backward.cpp"]

    def schema(self) -> str:
        return (
            "mhc_post_backward(Tensor grad_output, Tensor x, Tensor h_res, Tensor h_out, Tensor h_post) -> "
            "(Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def mhc_post_backward_meta(grad_output, x, h_res, h_out, h_post):
            _check_mhc_post_backward_inputs(grad_output, x, h_res, h_out, h_post)
            grad_x = torch.empty_like(x)
            grad_h_res = torch.empty_like(h_res)
            grad_h_out = torch.empty_like(h_out)
            grad_h_post = torch.empty_like(h_post)
            return (grad_x, grad_h_res, grad_h_out, grad_h_post)


mhc_post_backward_op_builder = MhcPostBackwardOpBuilder()


@impl(AS_LIBRARY, mhc_post_backward_op_builder.name, "PrivateUse1")
def _mhc_post_backward_dispatch(grad_output, x, h_res, h_out, h_post):
    op_module = mhc_post_backward_op_builder.load()
    return op_module.mhc_post_backward(grad_output, x, h_res, h_out, h_post)


def mhc_post_backward(grad_output, x, h_res, h_out, h_post):
    """计算 MHC Post Mapping 的反向梯度。

    Args:
        grad_output (Tensor): 上游梯度，shape 为 ``[T, N, D]`` 或
            ``[B, S, N, D]``，dtype 支持 ``float16``、``bfloat16``。
        x (Tensor): MHC 层输入，shape 和 dtype 与 ``grad_output`` 相同。
        h_res (Tensor): Res Mapping 矩阵，shape 为 ``[T, N, N]`` 或
            ``[B, S, N, N]``，dtype 为 ``float32``。
        h_out (Tensor): Atten/MLP 层输出，shape 为 ``[T, D]`` 或
            ``[B, S, D]``，dtype 与 ``grad_output`` 相同。
        h_post (Tensor): Post Mapping 矩阵，shape 为 ``[T, N]`` 或
            ``[B, S, N]``，dtype 为 ``float32``。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: ``grad_x``、``grad_h_res``、
        ``grad_h_out`` 和 ``grad_h_post``，分别与对应输入的 shape 和 dtype 一致。

    Note:
        所有输入必须非空且位于同一 NPU。Atlas A2/A3 上 ``N`` 仅支持 4，
        ``D`` 需按 128 对齐；Ascend 950 上 ``N`` 支持 4、6、8。
    """
    return torch.ops.cann_ops_transformer.mhc_post_backward(
        grad_output, x, h_res, h_out, h_post
    )
