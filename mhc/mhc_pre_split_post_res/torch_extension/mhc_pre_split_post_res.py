# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch import _check as torch_check
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

DEFAULT_NORM_EPS = 1.0e-6

X_RANK_TND = 3
X_RANK_BSND = 4
DIM_0 = 0
DIM_1 = 1
DIM_2 = 2

N_VALUE = 4
D_VALUES = (2560, 5120)
MIN_T = 1
MAX_T = 512 * 1024

X_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _check_inputs(x, phi, alpha, bias, norm_eps):
    torch_check(
        x.dim() == X_RANK_TND or x.dim() == X_RANK_BSND,
        lambda: f"x must be a 3D ([T,n,D]) or 4D ([B,S,n,D]) tensor, but got {x.dim()}D",
    )

    if x.dim() == X_RANK_TND:
        num_tokens = x.size(DIM_0)
        n = x.size(DIM_1)
        d = x.size(DIM_2)
    else:
        num_tokens = x.size(DIM_0) * x.size(DIM_1)
        n = x.size(DIM_2)
        d = x.size(-1)

    torch_check(
        x.dtype in X_SUPPORTED_DTYPES,
        lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
    )
    torch_check(
        num_tokens >= MIN_T and num_tokens <= MAX_T,
        lambda: (
            f"shape range requires {MIN_T}<=T=B*S<={MAX_T}, but got T={num_tokens}"
        ),
    )
    torch_check(n == N_VALUE, lambda: f"n must be {N_VALUE}, but got n={n}")
    torch_check(
        d in D_VALUES,
        lambda: f"D must be one of {D_VALUES}, but got D={d}",
    )

    phi_rows = n * n + n
    torch_check(
        phi.dim() == 2,
        lambda: f"phi must be a 2D tensor, but got {phi.dim()}D",
    )
    torch_check(
        phi.dtype == torch.float32,
        lambda: f"phi dtype must be float32, but got {phi.dtype}",
    )
    torch_check(
        tuple(phi.shape) == (phi_rows, n * d),
        lambda: f"phi shape must be [{phi_rows},{n * d}], but got {tuple(phi.shape)}",
    )

    torch_check(
        alpha.dim() == 1 and alpha.size(DIM_0) == 2,
        lambda: f"alpha shape must be [2], but got {tuple(alpha.shape)}",
    )
    torch_check(
        alpha.dtype == torch.float32,
        lambda: f"alpha dtype must be float32, but got {alpha.dtype}",
    )

    torch_check(
        bias.dim() == 1 and bias.size(DIM_0) == phi_rows,
        lambda: f"bias shape must be [{phi_rows}], but got {tuple(bias.shape)}",
    )
    torch_check(
        bias.dtype == torch.float32,
        lambda: f"bias dtype must be float32, but got {bias.dtype}",
    )

    torch_check(
        x.device == phi.device == alpha.device == bias.device,
        lambda: (
            "all inputs must be on the same device, but got "
            f"x={x.device}, phi={phi.device}, alpha={alpha.device}, bias={bias.device}"
        ),
    )
    torch_check(
        torch.isfinite(norm_eps) and norm_eps > 0.0,
        lambda: f"norm_eps must be finite and greater than zero, but got {norm_eps}",
    )


def _create_output_shapes(x):
    if x.dim() == X_RANK_TND:
        return (x.size(DIM_0), x.size(DIM_1)), (x.size(DIM_0), x.size(DIM_1), x.size(DIM_1))
    b, s, n = x.size(DIM_0), x.size(DIM_1), x.size(DIM_2)
    return (b, s, n), (b, s, n, n)


class MhcPreSplitPostResOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSplitPostResOpBuilder, self).__init__(
            "mhc_pre_split_post_res", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_split_post_res.cpp"]

    def schema(self) -> str:
        return (
            "mhc_pre_split_post_res(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            f"*, float norm_eps={DEFAULT_NORM_EPS}) -> (Tensor h_post, Tensor h_res)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_split_post_res_meta(x, phi, alpha, bias, norm_eps=DEFAULT_NORM_EPS):
            _check_inputs(x, phi, alpha, bias, norm_eps)
            h_post_shape, h_res_shape = _create_output_shapes(x)
            return (
                torch.empty(h_post_shape, device="meta", dtype=torch.float32),
                torch.empty(h_res_shape, device="meta", dtype=torch.float32),
            )


mhc_pre_split_post_res_op_builder = MhcPreSplitPostResOpBuilder()
mhc_pre_split_post_res_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_split_post_res_op_builder.name, "PrivateUse1")
def _mhc_pre_split_post_res_dispatch(x, phi, alpha, bias, norm_eps=DEFAULT_NORM_EPS):
    return mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps)


def mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps=DEFAULT_NORM_EPS):
    """mHC pre 阶段 post/Res 分支前向，封装 aclnnMhcPreSplitPostRes。

    输入 x 支持 [B,S,n,D] 或 [T,n,D]，返回 (h_post, h_res)，均为 float32。
    """
    _check_inputs(x, phi, alpha, bias, norm_eps)
    op_module = mhc_pre_split_post_res_op_builder.load()
    return op_module.mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps)
