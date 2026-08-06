# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Function

import npu_ops_transformer_ext
lightning_attn_forward = torch.ops.npu_ops_transformer_ext.lightning_attention_fwd
lightning_attn_backward = torch.ops.npu_ops_transformer_ext.lightning_attention_bwd

sys.path.append(os.getcwd())


class LightningAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, slope, isTraining: bool):
        kv, out = lightning_attn_forward(q, k, v, slope, isTraining)
        ctx.save_for_backward(q, k, v, slope, kv)
        ctx.isTraining = isTraining
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (
            q,
            k,
            v,
            slope,
            kv,
        ) = ctx.saved_tensors
        grad_q, grad_k, grad_v = lightning_attn_backward(grad_out.contiguous(), q, k, v, slope, kv)
        return grad_q, grad_k, grad_v, None, None, None  # Must be equal to the number of forward inputs


class LightningAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        use_decay: bool = True,
        isTraining: bool = False,
        block_size: int = 128,        
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_decay = use_decay
        self.isTraining = isTraining
        self.block_size = block_size

    def _prepare_shape(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Reshapes tensors for multi-head attention
        Input: (batch, seq, hidden)
        Output: (batch, num_heads, seq, head_size)
        """
        batch_size, seq_len, hidden_size = query.shape
        head_size = hidden_size // self.num_heads

        # Reshape and transpose tensors
        q = query.view(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2).contiguous()
        k = key.view(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2).contiguous()
        v = value.view(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2).contiguous()

        return q, k, v

    def forward(self, q, k, v, slope):
        # Reshape inputs
        q, k, v = self._prepare_shape(q, k, v)
        batch_size, num_heads, seq_len, head_size = q.shape

        if not self.use_decay:
            raise ValueError("Only lightning attention with decay supported")

        slope = slope.float().to(q.device)
        output = LightningAttentionFunction.apply(q, k, v, slope, self.isTraining)

        return output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_size)
