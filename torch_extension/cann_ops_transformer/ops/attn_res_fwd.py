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


class AttnResFwdOpBuilder(OpBuilder):
    def __init__(self):
        super(AttnResFwdOpBuilder, self).__init__("attn_res_fwd")

    def sources(self):
        return ['ops/csrc/attn_res_fwd.cpp']

    def schema(self) -> str:
        return (
            "attn_res_fwd(Tensor prefix_sum, Tensor block_residual, Tensor proj_weight, "
            "Tensor norm_weight, float norm_eps=1e-5, bool need_backward=False) "
            "-> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def attn_res_fwd_meta(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=1e-5,
                              need_backward=False):
            T = prefix_sum.size(0)
            H = prefix_sum.size(1)
            B = block_residual.size(1) + 1
            hidden = torch.empty(T, H, dtype=prefix_sum.dtype, device="meta")
            inv_rms = torch.empty(T, B, dtype=torch.float32, device="meta")
            probs = torch.empty(T, B, dtype=torch.float32, device="meta")
            return hidden, inv_rms, probs


attn_res_fwd_op_builder = AttnResFwdOpBuilder()


@impl(AS_LIBRARY, attn_res_fwd_op_builder.name, "PrivateUse1")
def _attn_res_fwd_dispatch(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=1e-5,
                           need_backward=False):
    op_module = attn_res_fwd_op_builder.load()
    return op_module.attn_res_fwd(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps,
                                  need_backward)


def attn_res_fwd(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=1e-5, need_backward=False):
    """Attention Residual 加权融合前向（Eager / PrivateUse1）。

    need_backward=False 时仅返回 hidden_states；
    need_backward=True 时返回 (hidden_states, inv_rms, probs)，dtype 分别为 BF16 / FP32 / FP32。
    """
    op_module = attn_res_fwd_op_builder.load()
    hidden, inv_rms, probs = op_module.attn_res_fwd(
        prefix_sum, block_residual, proj_weight, norm_weight, norm_eps, need_backward)
    if need_backward:
        return hidden, inv_rms, probs
    return hidden
