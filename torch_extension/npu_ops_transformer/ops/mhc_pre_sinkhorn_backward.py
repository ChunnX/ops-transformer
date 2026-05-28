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


class MhcPreSinkhornBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornBackwardOpBuilder, self).__init__("npu_mhc_pre_sinkhorn_backward")

    def sources(self):
        return ['ops/csrc/mhc_pre_sinkhorn_backward.cpp']

    def schema(self) -> str:
        return "npu_mhc_pre_sinkhorn_backward(Tensor gradHin, Tensor gradHPost, Tensor gradHRes, " \
               "Tensor x, Tensor phi, Tensor alpha, Tensor bias, " \
               "Tensor hPre, Tensor hcBeforeNorm, Tensor invRms, " \
               "Tensor sumOut, Tensor normOut, float hcEps) -> " \
               "(Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mhc_pre_sinkhorn_backward_meta(gradHin, gradHPost, gradHRes,
                                                x, phi, alpha, bias,
                                                hPre, hcBeforeNorm, invRms,
                                                sumOut, normOut, hcEps):
            gradX = torch.empty_like(x)
            gradPhi = torch.empty_like(phi)
            gradAlpha = torch.empty_like(alpha)
            gradBias = torch.empty_like(bias)
            return (gradX, gradPhi, gradAlpha, gradBias)


mhc_pre_sinkhorn_backward_op_builder = MhcPreSinkhornBackwardOpBuilder()
op_module = mhc_pre_sinkhorn_backward_op_builder.load()


@impl(AS_LIBRARY, mhc_pre_sinkhorn_backward_op_builder.name, "PrivateUse1")
def npu_mhc_pre_sinkhorn_backward(gradHin, gradHPost, gradHRes,
                                   x, phi, alpha, bias,
                                   hPre, hcBeforeNorm, invRms,
                                   sumOut, normOut, hcEps):
    return op_module.npu_mhc_pre_sinkhorn_backward(gradHin, gradHPost, gradHRes,
                                                     x, phi, alpha, bias,
                                                     hPre, hcBeforeNorm, invRms,
                                                     sumOut, normOut, hcEps)