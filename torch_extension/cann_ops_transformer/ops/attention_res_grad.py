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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class AttentionResGradOpBuilder(OpBuilder):
    def __init__(self):
        super(AttentionResGradOpBuilder, self).__init__("attention_res_grad")

    def sources(self):
        return ['ops/csrc/attention_res_grad.cpp']

    def schema(self) -> str:
        return (
            "attention_res_grad(Tensor prefix_sum, Tensor block_residual, Tensor proj_weight, "
            "Tensor norm_weight, Tensor grad_output, Tensor inv_rms, Tensor probs) "
            "-> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def attention_res_grad_meta(prefix_sum, block_residual, proj_weight, norm_weight, grad_output,
                                    inv_rms, probs):
            B = prefix_sum.size(0)
            H = prefix_sum.size(1)
            N = block_residual.size(1)
            grad_prefix_sum = torch.empty(B, H, dtype=prefix_sum.dtype, device="meta")
            grad_block_residual = torch.empty(B, N, H, dtype=prefix_sum.dtype, device="meta")
            grad_proj_weight = torch.empty(1, H, dtype=prefix_sum.dtype, device="meta")
            grad_norm_weight = torch.empty(H, dtype=prefix_sum.dtype, device="meta")
            return grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight


attention_res_grad_op_builder = AttentionResGradOpBuilder()


@impl(AS_LIBRARY, attention_res_grad_op_builder.name, "PrivateUse1")
def _attention_res_grad_dispatch(prefix_sum, block_residual, proj_weight, norm_weight,
                                 grad_output, inv_rms, probs):
    op_module = attention_res_grad_op_builder.load()
    return op_module.attention_res_grad(prefix_sum, block_residual, proj_weight, norm_weight,
                                        grad_output, inv_rms, probs)


def attention_res_grad(prefix_sum, block_residual, proj_weight, norm_weight, grad_output, inv_rms, probs):
    """Attention Residual 加权融合反向（Eager / PrivateUse1）。

    输入为 attn_res_fwd(need_backward=True) 的中间结果与 grad_output：
      prefix_sum [B, H], block_residual [B, N, H], proj_weight [1, H], norm_weight [H],
      grad_output [B, H], inv_rms [B, N+1], probs [B, N+1]
    返回 (grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight)，dtype 与输入一致。
    """
    op_module = attention_res_grad_op_builder.load()
    return op_module.attention_res_grad(prefix_sum, block_residual, proj_weight, norm_weight,
                                        grad_output, inv_rms, probs)
