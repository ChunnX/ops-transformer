# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


_OP_NAME = "mhc_sandwich_norm_post_preonly"


def _output_shapes(h_out: torch.Tensor, residual: torch.Tensor):
    if h_out.dim() == 2:
        t, d = h_out.shape
        n = residual.shape[1]
        return (t, d), (t, n, d)
    b, s, d = h_out.shape
    n = residual.shape[2]
    return (b, s, d), (b, s, n, d)


class MhcSandwichNormPostPreonlyOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__(_OP_NAME, category="mhc")

    def sources(self):
        return ["csrc/mhc/mhc_sandwich_norm_post_preonly.cpp"]

    def schema(self) -> List[str]:
        return [
            f"{_OP_NAME}(Tensor h_out, Tensor residual, Tensor h_post, Tensor h_res, "
            "Tensor phi, Tensor alpha_0, Tensor bias, Tensor gamma_0, Tensor gamma_1, "
            "Tensor? gamma_2=None, float norm_eps=1e-6, float mhc_pre_norm_eps=1e-6, "
            "float hc_eps=1e-6, bool return_h_in_f32=False) -> "
            "(Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        @impl(get_as_library(), _OP_NAME, "Meta")
        def meta(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha_0,
            bias,
            gamma_0,
            gamma_1,
            gamma_2=None,
            norm_eps=1e-6,
            mhc_pre_norm_eps=1e-6,
            hc_eps=1e-6,
            return_h_in_f32=False,
        ):
            h_in_shape, x2_shape = _output_shapes(h_out, residual)
            h_in_f32 = (
                torch.empty(h_in_shape, dtype=torch.float32, device="meta")
                if return_h_in_f32
                else torch.empty(0, dtype=torch.float32, device="meta")
            )
            return (
                torch.empty(h_in_shape, dtype=h_out.dtype, device="meta"),
                torch.empty(x2_shape, dtype=h_out.dtype, device="meta"),
                h_in_f32,
            )


op_builder = MhcSandwichNormPostPreonlyOpBuilder()
op_builder._ensure_initialized()


@impl(get_as_library(), _OP_NAME, "PrivateUse1")
def _dispatch(
    h_out,
    residual,
    h_post,
    h_res,
    phi,
    alpha_0,
    bias,
    gamma_0,
    gamma_1,
    gamma_2=None,
    norm_eps=1e-6,
    mhc_pre_norm_eps=1e-6,
    hc_eps=1e-6,
    return_h_in_f32=False,
):
    return op_builder.load().mhc_sandwich_norm_post_preonly(
        h_out,
        residual,
        h_post,
        h_res,
        phi,
        alpha_0,
        bias,
        gamma_0,
        gamma_1,
        gamma_2,
        norm_eps,
        mhc_pre_norm_eps,
        hc_eps,
        return_h_in_f32,
    )


def mhc_sandwich_norm_post_preonly(
    h_out: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    gamma_1: torch.Tensor,
    *,
    gamma_0: Optional[torch.Tensor] = None,
    gamma_2: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    mhc_pre_norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    return_h_in_f32: bool = False,
):
    """Fused RMSNorm + MHC post + MHC pre-only + RMSNorm operator."""
    if gamma_0 is None:
        raise ValueError("gamma_0=None is reserved but is not supported in this release")
    outputs = op_builder.load().mhc_sandwich_norm_post_preonly(
        h_out,
        residual,
        h_post,
        h_res,
        phi,
        alpha,
        bias,
        gamma_0,
        gamma_1,
        gamma_2,
        norm_eps,
        mhc_pre_norm_eps,
        hc_eps,
        return_h_in_f32,
    )
    return outputs if return_h_in_f32 else outputs[:2]
