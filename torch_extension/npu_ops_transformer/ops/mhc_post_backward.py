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


class MhcPostBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPostBackwardOpBuilder, self).__init__("npu_mhc_post_backward")

    def sources(self):
        return ['ops/csrc/mhc_post_backward.cpp']

    def schema(self) -> str:
        return "npu_mhc_post_backward(Tensor gradOutput, Tensor x, Tensor hRes, Tensor hOut, Tensor hPost) -> (Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mhc_post_backward_meta(gradOutput, x, hRes, hOut, hPost):
            gradX = torch.empty_like(x)
            gradHres = torch.empty_like(hRes)
            gradHout = torch.empty_like(hOut)
            gradHpost = torch.empty_like(hPost)
            return (gradX, gradHres, gradHout, gradHpost)


mhc_post_backward_op_builder = MhcPostBackwardOpBuilder()
op_module = mhc_post_backward_op_builder.load()


@impl(AS_LIBRARY, mhc_post_backward_op_builder.name, "PrivateUse1")
def npu_mhc_post_backward(gradOutput, x, hRes, hOut, hPost):
    return op_module.npu_mhc_post_backward(gradOutput, x, hRes, hOut, hPost)