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


class BlockAttentionResidualsOpBuilder(OpBuilder):
    def __init__(self):
        super(BlockAttentionResidualsOpBuilder, self).__init__("block_attention_residuals")

    def sources(self):
        return ['ops/csrc/block_attention_residuals.cpp']

    def schema(self) -> str:
        return (
            "block_attention_residuals(Tensor prefix_sum, Tensor block_res, Tensor proj_weight, "
            "Tensor norm_weight, int valid_block_num=-1, float norm_eps=1e-5, bool? need_backward=None) "
            "-> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def block_attention_residuals_meta(prefix_sum, block_res, proj_weight, norm_weight, valid_block_num=-1, norm_eps=1e-5,
                              need_backward=None):
            if need_backward is None:
                need_backward = all(t.requires_grad for t in
                                    (prefix_sum, block_res, proj_weight, norm_weight))
            T = prefix_sum.size(0)
            H = prefix_sum.size(1)
            B = block_res.size(1) + 1
            hidden = torch.empty(T, H, dtype=prefix_sum.dtype, device="meta")
            inv_norm = torch.empty(T, B, dtype=torch.float32, device="meta")
            probs = torch.empty(T, B, dtype=torch.float32, device="meta")
            return hidden, inv_norm, probs


block_attention_residuals_op_builder = BlockAttentionResidualsOpBuilder()


def _resolve_need_backward(prefix_sum, block_res, proj_weight, norm_weight, need_backward):
    if need_backward is not None:
        return need_backward
    return all(t.requires_grad for t in (prefix_sum, block_res, proj_weight, norm_weight))


@impl(AS_LIBRARY, block_attention_residuals_op_builder.name, "PrivateUse1")
def _block_attention_residuals_dispatch(prefix_sum, block_res, proj_weight, norm_weight, valid_block_num=-1, norm_eps=1e-5,
                           need_backward=None):
    if valid_block_num < 0:
        valid_block_num = block_res.size(1)
    need_backward = _resolve_need_backward(prefix_sum, block_res, proj_weight, norm_weight,
                                           need_backward)
    op_module = block_attention_residuals_op_builder.load()
    return op_module.block_attention_residuals(prefix_sum, block_res, proj_weight, norm_weight, valid_block_num, norm_eps,
                                  need_backward)


def block_attention_residuals(prefix_sum, block_res, proj_weight, norm_weight, valid_block_num=-1, norm_eps=1e-5, need_backward=None):
    """Attention Residual 加权融合前向（Eager / PrivateUse1）。

    need_backward=None（默认）时根据是否所有输入（prefix_sum/block_res/proj_weight/norm_weight）
    都有梯度（requires_grad=True）自动决定：全有梯度则返回中间量，否则不返回。
    need_backward=False 时仅返回 hidden_states；
    need_backward=True 时返回 (hidden_states, inv_norm, probs)，dtype 分别为 BF16 / FP32 / FP32。
    """
    if valid_block_num < 0:
        valid_block_num = block_res.size(1)
    need_backward = _resolve_need_backward(prefix_sum, block_res, proj_weight, norm_weight,
                                           need_backward)
    op_module = block_attention_residuals_op_builder.load()
    hidden, inv_norm, probs = op_module.block_attention_residuals(
        prefix_sum, block_res, proj_weight, norm_weight, valid_block_num, norm_eps, need_backward)
    if need_backward:
        return hidden, inv_norm, probs
    return hidden
