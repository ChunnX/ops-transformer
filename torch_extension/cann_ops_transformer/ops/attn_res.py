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

from cann_ops_transformer.ops.attn_res_fwd import attn_res_fwd
from cann_ops_transformer.ops.attention_res_grad import attention_res_grad


class AttnResFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix_sum, block_residual, proj_weight, norm_weight, norm_eps):
        hidden, inv_rms, probs = attn_res_fwd(
            prefix_sum, block_residual, proj_weight, norm_weight,
            norm_eps=norm_eps, need_backward=True)
        ctx.save_for_backward(prefix_sum, block_residual, proj_weight, norm_weight, inv_rms, probs)
        return hidden

    @staticmethod
    def backward(ctx, grad_output):
        prefix_sum, block_residual, proj_weight, norm_weight, inv_rms, probs = ctx.saved_tensors
        grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight = attention_res_grad(
            prefix_sum, block_residual, proj_weight, norm_weight,
            grad_output, inv_rms, probs)
        return grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight, None


def attn_res(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=1e-5):
    """Attention Residual 加权融合前向 + 反向（autograd 绑定，参考 mhc_post）。

    输入需可微时返回可微算子（forward=attn_res_fwd，backward=attention_res_grad）；
    否则仅执行正向返回 hidden_states。
    """
    needs_grad = (
        prefix_sum.requires_grad
        or block_residual.requires_grad
        or proj_weight.requires_grad
        or norm_weight.requires_grad
    )

    if needs_grad:
        return AttnResFunction.apply(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps)

    hidden = attn_res_fwd(
        prefix_sum, block_residual, proj_weight, norm_weight,
        norm_eps=norm_eps, need_backward=False)
    return hidden
