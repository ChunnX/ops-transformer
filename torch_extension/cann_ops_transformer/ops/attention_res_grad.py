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


class AttentionResGradOpBuilder(OpBuilder):
    def __init__(self):
        super(AttentionResGradOpBuilder, self).__init__("attention_res_grad")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/attention_res_grad.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "attention_res_grad(Tensor prefix_sum, Tensor block_residual, "
            "Tensor proj_weight, Tensor norm_weight, Tensor grad_output, "
            "Tensor inv_rms, Tensor probs) -> "
            "(Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """Register shape/dtype inference for fake tensors and autograd."""

        @impl(AS_LIBRARY, self.name, "Meta")
        def attention_res_grad_meta(
            prefix_sum,
            block_residual,
            proj_weight,
            norm_weight,
            grad_output,
            inv_rms,
            probs,
        ):
            grad_prefix_sum = torch.empty_like(prefix_sum)
            grad_block_residual = torch.empty_like(block_residual)
            grad_proj_weight = torch.empty_like(proj_weight)
            grad_norm_weight = torch.empty_like(norm_weight)
            return (
                grad_prefix_sum,
                grad_block_residual,
                grad_proj_weight,
                grad_norm_weight,
            )


attention_res_grad_op_builder = AttentionResGradOpBuilder()


@impl(AS_LIBRARY, attention_res_grad_op_builder.name, "PrivateUse1")
def attention_res_grad(
    prefix_sum,
    block_residual,
    proj_weight,
    norm_weight,
    grad_output,
    inv_rms,
    probs,
):
    """Dispatch implementation for NPU (PrivateUse1)."""
    op_module = attention_res_grad_op_builder.load()
    return op_module.attention_res_grad(
        prefix_sum,
        block_residual,
        proj_weight,
        norm_weight,
        grad_output,
        inv_rms,
        probs,
    )
