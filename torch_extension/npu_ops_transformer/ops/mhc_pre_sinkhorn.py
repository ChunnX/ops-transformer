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
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MhcPreSinkhornFunction(torch.autograd.Function):
    """Autograd wrapper: forward -> mhc_pre_sinkhorn, backward -> mhc_pre_sinkhorn_backward"""

    @staticmethod
    def forward(ctx, x, phi, alpha, bias, hcMult, numIters, hcEps, normEps):
        # autograd 路径强制 outFlag=True，以保存 backward 所需的中间结果
        hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut = \
            op_module.mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, True)

        ctx.save_for_backward(x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut)
        ctx.hcEps = hcEps

        return hin, hPost, hRes

    @staticmethod
    def backward(ctx, gradHin, gradHPost, gradHRes):
        x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut = ctx.saved_tensors
        hcEps = ctx.hcEps

        from npu_ops_transformer.ops.mhc_pre_sinkhorn_backward import mhc_pre_sinkhorn_backward
        gradX, gradPhi, gradAlpha, gradBias = mhc_pre_sinkhorn_backward(
            gradHin, gradHPost, gradHRes,
            x, phi, alpha, bias,
            hPre, hcBeforeNorm, invRms,
            sumOut, normOut, hcEps
        )

        # 与 forward 参数一一对应，非 Tensor 参数的梯度填 None
        return gradX, gradPhi, gradAlpha, gradBias, None, None, None, None


class MhcPreSinkhornOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornOpBuilder, self).__init__("mhc_pre_sinkhorn")

    def sources(self):
        return ['ops/csrc/mhc_pre_sinkhorn.cpp']

    def schema(self) -> str:
        return "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, " \
               "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> " \
               "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def mhc_pre_sinkhorn_meta(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag):
            B = x.size(0)
            S = x.size(1)
            N = x.size(2)
            C = x.size(3)

            hin = torch.empty(B, S, C, dtype=x.dtype, device="meta")
            hPost = torch.empty(B, S, N, dtype=phi.dtype, device="meta")
            hRes = torch.empty(B, S, N * N, dtype=phi.dtype, device="meta")

            if outFlag:
                hPre = torch.empty(B, S, N, dtype=phi.dtype, device="meta")
                hcBeforeNorm = torch.empty(B, S, N * N + 2 * N, dtype=phi.dtype, device="meta")
                invRms = torch.empty(B, S, 1, dtype=phi.dtype, device="meta")
                sumOut = torch.empty(2 * numIters, B, S, N, dtype=phi.dtype, device="meta")
                normOut = torch.empty(2 * numIters, B, S, N, N, dtype=phi.dtype, device="meta")
            else:
                hPre = torch.empty(0, dtype=phi.dtype, device="meta")
                hcBeforeNorm = torch.empty(0, dtype=phi.dtype, device="meta")
                invRms = torch.empty(0, dtype=phi.dtype, device="meta")
                sumOut = torch.empty(0, dtype=phi.dtype, device="meta")
                normOut = torch.empty(0, dtype=phi.dtype, device="meta")

            return (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut)


mhc_pre_sinkhorn_op_builder = MhcPreSinkhornOpBuilder()
op_module = mhc_pre_sinkhorn_op_builder.load()


@impl(AS_LIBRARY, mhc_pre_sinkhorn_op_builder.name, "PrivateUse1")
def _mhc_pre_sinkhorn_dispatch(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag):
    return op_module.mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag)


def mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps=1e-6, normEps=1e-6):
    """自动反向传播接口，正向始终只返回 (hin, hPost, hRes)。

    - 若任一输入设置了 requires_grad=True，走 autograd 路径，自动保存中间变量用于 backward。
    - 若无输入需梯度，走普通路径，中间变量不保存（节省内存）。
    - backward 通过 tensor.grad 自动获取梯度。
    """
    needs_grad = x.requires_grad or phi.requires_grad or alpha.requires_grad or bias.requires_grad

    if needs_grad:
        # autograd 路径：outFlag=True，中间变量由 ctx.save_for_backward 保存
        return MhcPreSinkhornFunction.apply(
            x, phi, alpha, bias, hcMult, numIters, hcEps, normEps
        )
    else:
        # 普通路径：outFlag=False，不保存中间变量，只返回 3 个主输出
        hin, hPost, hRes, _, _, _, _, _ = op_module.mhc_pre_sinkhorn(
            x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, False
        )
        return hin, hPost, hRes