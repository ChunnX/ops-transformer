# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MhcPreSinkhornPremixOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornPremixOpBuilder, self).__init__(
            "mhc_pre_sinkhorn_premix", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_sinkhorn_premix.cpp"]

    def schema(self) -> str:
        return (
            "mhc_pre_sinkhorn_premix(Tensor x, Tensor phi, Tensor alpha, Tensor bias, Tensor? premix, "
            "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_premix_meta(
            x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag
        ):
            n = hc_mult
            if x.dim() == 3:
                t = x.size(0)
                c = x.size(2)
                hin = torch.empty(t, c, dtype=x.dtype, device="meta")
                h_post = torch.empty(t, n, dtype=phi.dtype, device="meta")
                h_res = torch.empty(t, n * n, dtype=phi.dtype, device="meta")
                if out_flag:
                    h_pre = torch.empty(t, n, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(
                        t, n * n + 2 * n, dtype=phi.dtype, device="meta"
                    )
                    inv_rms = torch.empty(t, 1, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(
                        2 * num_iters, t, n, dtype=phi.dtype, device="meta"
                    )
                    norm_out = torch.empty(
                        2 * num_iters, t, n, n, dtype=phi.dtype, device="meta"
                    )
                else:
                    h_pre = torch.empty(0, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(0, dtype=phi.dtype, device="meta")
                    inv_rms = torch.empty(0, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(0, dtype=phi.dtype, device="meta")
                    norm_out = torch.empty(0, dtype=phi.dtype, device="meta")
            else:
                b = x.size(0)
                s = x.size(1)
                c = x.size(3)
                hin = torch.empty(b, s, c, dtype=x.dtype, device="meta")
                h_post = torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                h_res = torch.empty(b, s, n * n, dtype=phi.dtype, device="meta")
                if out_flag:
                    h_pre = torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(
                        b, s, n * n + 2 * n, dtype=phi.dtype, device="meta"
                    )
                    inv_rms = torch.empty(b, s, 1, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(
                        2 * num_iters, b, s, n, dtype=phi.dtype, device="meta"
                    )
                    norm_out = torch.empty(
                        2 * num_iters, b, s, n, n, dtype=phi.dtype, device="meta"
                    )
                else:
                    h_pre = torch.empty(0, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(0, dtype=phi.dtype, device="meta")
                    inv_rms = torch.empty(0, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(0, dtype=phi.dtype, device="meta")
                    norm_out = torch.empty(0, dtype=phi.dtype, device="meta")

            return (
                hin,
                h_post,
                h_res,
                h_pre,
                hc_before_norm,
                inv_rms,
                sum_out,
                norm_out,
            )


mhc_pre_sinkhorn_premix_op_builder = MhcPreSinkhornPremixOpBuilder()
mhc_pre_sinkhorn_premix_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_sinkhorn_premix_op_builder.name, "PrivateUse1")
def _mhc_pre_sinkhorn_premix_dispatch(
    x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag
):
    op_module = mhc_pre_sinkhorn_premix_op_builder.load()
    return op_module.mhc_pre_sinkhorn_premix(
        x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag
    )


def mhc_pre_sinkhorn_premix(
    x,
    phi,
    alpha,
    bias,
    premix: Optional[torch.Tensor] = None,
    hc_mult: int = 4,
    num_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    need_backward: bool = False,
):
    """MhcPreSinkhornPremix 算子接口（experimental）。

    - premix: 可选输入，shape 为 x.shape[:-1]（即 (t, n) 或 (b, s, n)），dtype float32。
      传入时 hin 使用 premix 作为加权系数（通常填上一轮输出的 h_pre），而非本轮内部计算的 hPre。
      仅 Ascend950 支持。
    - need_backward=True 时，额外返回 h_pre/hc_before_norm/inv_rms/sum_out/norm_out 中间变量
      （无论是否传入 premix，均输出本轮内部计算的 h_pre，可供反向使用）。
    - 反向梯度计算复用正式算子 mhc_pre_sinkhorn_backward。
    """
    op_module = mhc_pre_sinkhorn_premix_op_builder.load()
    hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
        op_module.mhc_pre_sinkhorn_premix(
            x,
            phi,
            alpha,
            bias,
            premix,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            need_backward,
        )
    )
    if need_backward:
        return hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out
    return hin, h_post, h_res
