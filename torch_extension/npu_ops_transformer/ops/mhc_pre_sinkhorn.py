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


class MhcPreSinkhornOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornOpBuilder, self).__init__("npu_mhc_pre_sinkhorn")

    def sources(self):
        return ['ops/csrc/mhc_pre_sinkhorn.cpp']

    def schema(self) -> str:
        return "npu_mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, " \
               "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> " \
               "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mhc_pre_sinkhorn_meta(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag):
            B = x.size(0)
            S = x.size(1)
            N = x.size(2)
            C = x.size(3)

            hin = torch.empty(B, S, C, dtype=torch.bfloat16, device="meta")
            hPost = torch.empty(B, S, N, dtype=torch.float32, device="meta")
            hRes = torch.empty(B, S, N * N, dtype=torch.float32, device="meta")

            if outFlag:
                hPre = torch.empty(B, S, N, dtype=torch.float32, device="meta")
                hcBeforeNorm = torch.empty(B, S, N * N + 2 * N, dtype=torch.float32, device="meta")
                invRms = torch.empty(B, S, 1, dtype=torch.float32, device="meta")
                sumOut = torch.empty(2 * numIters, B, S, N, dtype=torch.float32, device="meta")
                normOut = torch.empty(2 * numIters, B, S, N, N, dtype=torch.float32, device="meta")
            else:
                hPre = torch.empty(0, dtype=torch.float32, device="meta")
                hcBeforeNorm = torch.empty(0, dtype=torch.float32, device="meta")
                invRms = torch.empty(0, dtype=torch.float32, device="meta")
                sumOut = torch.empty(0, dtype=torch.float32, device="meta")
                normOut = torch.empty(0, dtype=torch.float32, device="meta")

            return (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut)


mhc_pre_sinkhorn_op_builder = MhcPreSinkhornOpBuilder()
op_module = mhc_pre_sinkhorn_op_builder.load()


@impl(AS_LIBRARY, mhc_pre_sinkhorn_op_builder.name, "PrivateUse1")
def npu_mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag):
    return op_module.npu_mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag)