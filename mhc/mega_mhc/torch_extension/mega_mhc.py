# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 950 mega_mhc: four/five-stage forward with explicit rounding boundaries."""

import math
from typing import Optional, Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _check_inputs(
    h_out,
    residual,
    h_post,
    h_res,
    phi,
    alpha,
    bias,
    gamma_1,
    pre_mix,
    gamma_0,
    gamma_2,
    norm_eps,
    mhc_pre_norm_eps,
    hc_eps,
    compute_mode,
):
    """Validate the common eager/Meta contract; leading token axes remain symbolic."""
    torch._check(
        pre_mix is None, lambda: "pre_mix is reserved and must be None in this version"
    )
    torch._check(h_out.dim() in (2, 3), lambda: "h_out must be [T,D] or [B,S,D]")
    torch._check(
        h_out.dtype in (torch.float16, torch.bfloat16),
        lambda: "h_out must have float16 or bfloat16 dtype",
    )
    d = h_out.shape[-1]
    torch._check(
        (d >= 2560) & (d <= 7168) & (d % 256 == 0),
        lambda: "D must be in [2560,7168] and aligned to 256",
    )
    leading = h_out.shape[:-1]
    rows = 1
    for size in leading:
        torch._check(size > 0, lambda: "token dimensions must be positive")
        rows *= size
    torch._check(rows <= 2**32 - 1, lambda: "flattened token count exceeds uint32")
    tensors = [
        ("residual", residual, (*leading, 4, d), h_out.dtype),
        ("h_post", h_post, (*leading, 4), torch.float32),
        ("h_res", h_res, (*leading, 4, 4), torch.float32),
        ("phi", phi, (24, 4 * d), torch.float32),
        ("alpha", alpha, (3,), torch.float32),
        ("bias", bias, (24,), torch.float32),
        ("gamma_1", gamma_1, (d,), torch.float32),
        ("gamma_0", gamma_0, (d,), torch.float32),
    ]
    if gamma_2 is not None:
        tensors.append(("gamma_2", gamma_2, (4 * d,), torch.float32))
    for name, value, shape, dtype in tensors:
        torch._check(value is not None, lambda: f"{name} is required")
        torch._check(value.dtype == dtype, lambda: f"{name} has incorrect dtype")
        torch._check(
            value.device == h_out.device, lambda: f"{name} must be on h_out.device"
        )
        torch._check(value.dim() == len(shape), lambda: f"{name} has incorrect rank")
        for actual, expected in zip(value.shape, shape):
            torch._check(actual == expected, lambda: f"{name} has incorrect shape")
    torch._check(
        compute_mode in (0, 1), lambda: "compute_mode must be 0 (FP32) or 1 (HF32)"
    )
    torch._check(
        math.isfinite(norm_eps)
        and 0 < norm_eps <= 3.4028234663852886e38
        and norm_eps > 2**-150,
        lambda: "norm_eps must remain finite and positive in FP32",
    )
    torch._check(
        math.isfinite(mhc_pre_norm_eps)
        and 2**-150 < mhc_pre_norm_eps <= 3.4028234663852886e38,
        lambda: "mhc_pre_norm_eps must remain finite and positive in FP32",
    )
    torch._check(
        math.isfinite(hc_eps) and 0 <= hc_eps <= 3.4028234663852886e38,
        lambda: "hc_eps must remain finite and nonnegative in FP32",
    )


class MegaMhcOpBuilder(OpBuilder):
    """Collect the repository-local wrapper through the standard mhc JIT builder."""

    def __init__(self):
        """Register the public name mega_mhc in the mhc category."""
        super().__init__("mega_mhc", category="mhc")

    def sources(self):
        """Resolve the path produced by torch_extension/setup.py."""
        return ["csrc/mhc/mega_mhc.cpp"]

    def schema(self) -> str:
        """gamma_0 is required; gamma_2 alone selects the optional Mid stage."""
        return (
            "mega_mhc(Tensor h_out, Tensor residual, Tensor h_post, Tensor h_res, "
            "Tensor phi, Tensor alpha, Tensor bias, Tensor gamma_1, Tensor? pre_mix=None, Tensor? gamma_0=None, *, "
            "Tensor? gamma_2=None, float norm_eps=1e-6, float mhc_pre_norm_eps=1e-6, "
            "float hc_eps=1e-6, int compute_mode=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)"
        )

    def register_meta(self):
        """Infer four tensor outputs and reserved None without loading the NPU extension."""

        @impl(get_as_library(), self.name, "Meta")
        def mega_mhc_meta(
            h_out,
            residual,
            h_post,
            h_res,
            phi,
            alpha,
            bias,
            gamma_1,
            pre_mix=None,
            gamma_0=None,
            *,
            gamma_2=None,
            norm_eps=1e-6,
            mhc_pre_norm_eps=1e-6,
            hc_eps=1e-6,
            compute_mode=0,
        ):
            _check_inputs(
                h_out,
                residual,
                h_post,
                h_res,
                phi,
                alpha,
                bias,
                gamma_1,
                pre_mix,
                gamma_0,
                gamma_2,
                norm_eps,
                mhc_pre_norm_eps,
                hc_eps,
                compute_mode,
            )
            leading = h_out.shape[:-1]
            return (
                h_out.new_empty(h_out.shape),
                h_out.new_empty((*leading, 4), dtype=torch.float32),
                h_out.new_empty((*leading, 4, 4), dtype=torch.float32),
                h_out.new_empty(residual.shape),
                None,
            )


_mega_mhc_op_builder = MegaMhcOpBuilder()
_mega_mhc_op_builder._ensure_initialized()


@impl(get_as_library(), _mega_mhc_op_builder.name, "PrivateUse1")
def _mega_mhc(
    h_out,
    residual,
    h_post,
    h_res,
    phi,
    alpha,
    bias,
    gamma_1,
    pre_mix=None,
    gamma_0=None,
    *,
    gamma_2=None,
    norm_eps=1e-6,
    mhc_pre_norm_eps=1e-6,
    hc_eps=1e-6,
    compute_mode=0,
):
    """Dispatch validated NPU tensors to the ACLNN wrapper, compiling on first use."""
    _check_inputs(
        h_out,
        residual,
        h_post,
        h_res,
        phi,
        alpha,
        bias,
        gamma_1,
        pre_mix,
        gamma_0,
        gamma_2,
        norm_eps,
        mhc_pre_norm_eps,
        hc_eps,
        compute_mode,
    )
    return _mega_mhc_op_builder.load().mega_mhc(
        h_out,
        residual,
        h_post,
        h_res,
        phi,
        alpha,
        bias,
        gamma_1,
        pre_mix,
        gamma_0,
        gamma_2,
        norm_eps,
        mhc_pre_norm_eps,
        hc_eps,
        compute_mode,
    )


def mega_mhc(
    h_out: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    gamma_1: torch.Tensor,
    pre_mix: Optional[torch.Tensor] = None,
    gamma_0: Optional[torch.Tensor] = None,
    *,
    gamma_2: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    mhc_pre_norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    compute_mode: int = 0,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    """Compute RMS0 -> Post -> optional RMSMid -> Pre -> RMS1 on Ascend 950.

    Args:
        h_out: FP16/BF16 [T,D] or [B,S,D], 2560 <= D <= 7168 and D % 256 == 0.
        residual: Same dtype and leading axes, followed by [4,D].
        h_post: FP32 [...,4], Post weights.
        h_res: FP32 [...,4,4], indexed [source head, output head].
        phi: FP32 [24,4D], projection weights.
        alpha: FP32 [3], scales for pre/post/residual gates.
        bias: FP32 [24], gate offsets.
        gamma_1: FP32 [D], final RMS scale.
        pre_mix: Reserved optional FP32 [...,4] ND contiguous input; must be None in this version.
        gamma_0: Required FP32 [D], initial RMS scale.
        gamma_2: Optional FP32 [4D]; provided means five-stage execution.
        norm_eps: Finite positive FP32-representable RMS epsilon.
        mhc_pre_norm_eps: Independent finite positive FP32-representable Pre RMS epsilon.
        hc_eps: Finite nonnegative FP32-representable pre-gate epsilon.
        compute_mode: 0 for FP32 projection, 1 for HF32; no 3BF16.

    Returns:
        (h_in_prime, h_post_new, h_res_new, x_2_out, h_pre), shaped like
        (h_out, h_post, h_res, residual, None), dtypes (input, FP32, FP32, input, None).
        h_pre is a reserved optional output and is always None in this version.
        Noncontiguous inputs are accepted. Forward only; no Sinkhorn stage.
    """
    return torch.ops.cann_ops_transformer.mega_mhc(
        h_out,
        residual,
        h_post,
        h_res,
        phi,
        alpha,
        bias,
        gamma_1,
        pre_mix,
        gamma_0,
        gamma_2=gamma_2,
        norm_eps=norm_eps,
        mhc_pre_norm_eps=mhc_pre_norm_eps,
        hc_eps=hc_eps,
        compute_mode=compute_mode,
    )
