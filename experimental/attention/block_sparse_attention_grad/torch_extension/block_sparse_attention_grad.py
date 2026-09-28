# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional

import torch
import torch_npu  # noqa: F401
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class BlockSparseAttentionGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_sparse_attention_grad")

    def sources(self):
        # setup.py 会把 csrc 收敛到包内 csrc/attention/ 下（wheel 内相对包根的路径）
        return ["csrc/attention/block_sparse_attention_grad.cpp"]

    def schema(self) -> str:
        return (
            "block_sparse_attention_grad("
            "Tensor d_out, Tensor query, Tensor key, Tensor value, "
            "Tensor attention_out, Tensor softmax_lse, Tensor block_sparse_mask, "
            "int[]? block_shape=None, *, "
            'str q_input_layout="BNSD", '
            'str kv_input_layout="BNSD", '
            "int num_key_value_heads=1, "
            "float scale_value=1.0, "
            "int[]? actual_seq_lengths=None, "
            "int[]? actual_seq_lengths_kv=None, "
            "Tensor? atten_mask=None, "
            "int mask_type=0"
            ") -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta(
            d_out: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_out: torch.Tensor,
            softmax_lse: torch.Tensor,
            block_sparse_mask: torch.Tensor,
            block_shape: Optional[List[int]],
            *,
            q_input_layout: str = "BNSD",
            kv_input_layout: str = "BNSD",
            num_key_value_heads: int = 1,
            scale_value: float = 1.0,
            actual_seq_lengths: Optional[List[int]] = None,
            actual_seq_lengths_kv: Optional[List[int]] = None,
            atten_mask: Optional[torch.Tensor] = None,
            mask_type: int = 0,
        ):
            dq = query.new_empty(query.shape)
            dk = key.new_empty(key.shape)
            dv = value.new_empty(value.shape)
            return dq, dk, dv


_op_builder = BlockSparseAttentionGradOpBuilder()
_op_module = _op_builder.load()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def block_sparse_attention_grad(
    d_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    block_sparse_mask: torch.Tensor,
    block_shape: Optional[List[int]] = None,
    *,
    q_input_layout: str = "BNSD",
    kv_input_layout: str = "BNSD",
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
    atten_mask: Optional[torch.Tensor] = None,
    mask_type: int = 0,
):
    """block sparse attention backward（aclnnBlockSparseAttentionGrad 的 torch 扩展封装）。

    mask_type=0 → atten_mask 必须为 None（无 counts）；
    mask_type=1 → atten_mask 必传（INT32 counts：1D 前缀计数 或 4D [B,N,>=maxBlocks,2]）。
    返回 (dq, dk, dv)。
    """
    return _op_module.npu_block_sparse_attention_backward(
        d_out,
        query,
        key,
        value,
        attention_out,
        softmax_lse,
        block_sparse_mask,
        block_shape,
        q_input_layout,
        kv_input_layout,
        num_key_value_heads,
        scale_value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        atten_mask,
        mask_type,
    )
