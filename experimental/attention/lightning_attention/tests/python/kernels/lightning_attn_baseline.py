# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from typing import Optional, Tuple
import logging

log = logging.getLogger()


class LightningAttentionBaseline(torch.nn.Module):
    def __init__(self, num_heads, dtype, use_decay=False, block_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.dtype = dtype
        self.use_decay = use_decay
        self.block_size = block_size

    def decay_prepare(self, slope: torch.Tensor):
        assert slope is not None
        if not self.use_decay:
            shape = slope.reshape(-1, 1, 1).shape
            slope = torch.zeros(shape, device=slope.device, dtype=slope.dtype)

        array = torch.arange(self.block_size).to(slope) + 1
        q_decay = torch.exp(-slope * array.reshape(-1, 1))
        k_decay = torch.exp(-slope * (self.block_size - array.reshape(-1, 1)))
        index = array[:, None] - array[None, :]
        s_index = (
            slope
            * index[
                None,
                None,
            ]
        )
        s_index = torch.where(index >= 0, -s_index, float("-inf"))
        diag_decay = torch.exp(s_index)
        block_decay = torch.exp(-slope * array)

        return q_decay, k_decay, diag_decay, block_decay

    def O_intra(self, q, k, v, diag_decay, qk=None):
        if qk is None:
            qk = torch.matmul(q, k.transpose(-1, -2)).to(self.dtype) * diag_decay
        else:
            qk = qk * diag_decay
        return torch.matmul(qk, v.to(self.dtype))

    def O_inter(self, q, kv, q_decay):
        return torch.matmul(q * q_decay, kv).to(self.dtype)

    def kv_collect(self, k, v, kv, k_decay, block_decay):
        return block_decay * kv + torch.matmul((k * k_decay).transpose(-1, -2).to(self.dtype), v)

    def __prepare_shape(self, query, key, value):
        """
        Reshapes query/key/value tensors for multi-head attention.

        3D Input: (batch_size, seq_len, hidden_size)
            -> reshaped to (batch_size, num_heads, seq_len, head_size)

        Args:
            query: Input query tensor
            key: Input key tensor
            value: Input value tensor

        Returns:
            Tuple of reshaped (query, key, value) tensors
        """

        assert query.dim() == 3, "Shape must be in (batch_size, seq_len, hidden_size)"
        assert query.shape == key.shape
        assert query.shape == value.shape

        batch_size, seq_len = query.size(0), query.size(1)

        # Handle 3D tensors (combine heads)
        hidden_size = query.size(2)
        head_size = hidden_size // self.num_heads
        # Reshape to: (batch_size, seq_len, num_heads, head_size)
        query = query.reshape(batch_size, seq_len, self.num_heads, head_size)
        key = key.reshape(batch_size, seq_len, self.num_heads, head_size)
        value = value.reshape(batch_size, seq_len, self.num_heads, head_size)

        # Transpose heads and sequence dimensions
        # New shape: (batch_size, num_heads, seq_len, head_size)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        return query, key, value

    def forward(
        self,
        q,
        k,
        v,
        slope: torch.Tensor,  # (h, 1, 1)
        attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        q, k, v = self.__prepare_shape(q, k, v)
        b, h, n, d = q.shape
        e = v.shape[-1]  # In general, d != e

        kv = torch.zeros(b, h, d, e).to(self.dtype).to(q.device)
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        q_decay, k_decay, diag_decay, block_decay = self.decay_prepare(slope)

        num_block = (n + self.block_size - 1) // self.block_size
        for i in range(num_block):
            si = i * self.block_size
            ei = min(si + self.block_size, n)
            m = ei - si
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            O_intra = self.O_intra(qi, ki, vi, diag_decay[:, :, :m, :m])
            O_inter = self.O_inter(qi, kv, q_decay[:, :m])
            output[:, :, si:ei] = O_inter + O_intra
            kv = self.kv_collect(ki, vi, kv, k_decay[:, -m:], block_decay[:, :, -1:])

        output = output.permute(0, 2, 1, 3).reshape(output.shape[0], output.shape[2], -1)

        return output

    def process_inter_chunk(
        self,
        q,
        k,
        v,
        slope: Optional[torch.Tensor] = None,  # (n, 1, 1)
    ):

        b, n, s, h = q.shape
        num_block = (s + self.block_size - 1) // self.block_size

        kv = torch.zeros(b, n, h, h).to(self.dtype).to(q.device)
        output = torch.empty((b, n, s, h), dtype=q.dtype, device=q.device)

        q_decay, k_decay, diag_decay, block_decay = self.decay_prepare(slope)

        for i in range(num_block):
            si = i * self.block_size
            ei = min(si + self.block_size, s)
            m = ei - si
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            output[:, :, si:ei] = self.O_inter(qi, kv, q_decay[:, :m])
            kv = self.kv_collect(ki, vi, kv, k_decay[:, -m:], block_decay[:, :, m - 1 :])

        return output

    def process_intra_chunk(
        self,
        q,
        k,
        v,
        slope: Optional[torch.Tensor] = None,  # (n, 1, 1)
    ):

        b, n, s, h = q.shape
        num_block = (s + self.block_size - 1) // self.block_size

        output = torch.empty((b, n, s, h), dtype=q.dtype, device=q.device)
        q_decay, k_decay, diag_decay, block_decay = self.decay_prepare(slope)

        for i in range(num_block):
            si = i * self.block_size
            ei = min(si + self.block_size, s)
            m = ei - si
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            output[:, :, si:ei] = self.O_intra(qi, ki, vi, diag_decay[:, :, :m, :m])

        return output
