# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class KvQuantFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(KvQuantFlashAttentionOpBuilder, self).__init__(
            "kv_quant_flash_attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/kv_quant_flash_attention.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "kv_quant_flash_attention(Tensor query, Tensor key, *, "
            "Tensor? value=None, Tensor? sparse_indices=None, "
            "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
            "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_kv=None, Tensor? sinks=None, "
            "float scale_value=1.0, int key_quant_mode=1, int value_quant_mode=1, "
            "int sparse_block_size=1, str layout_query='BSND', str layout_kv='BSND', "
            "int sparse_mode=3, int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, int attention_mode=0, "
            "int quant_scale_repo_mode=1, int tile_size=128, int rope_head_dim=64, "
            "bool return_softmax_lse=False, int? key_dtype=None, int? value_dtype=None) "
            "-> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def kv_quant_flash_attention_meta(
            query,
            key,
            value=None,
            sparse_indices=None,
            key_dequant_scale=None,
            value_dequant_scale=None,
            block_table=None,
            actual_seq_lengths_query=None,
            actual_seq_lengths_kv=None,
            sinks=None,
            scale_value=1.0,
            key_quant_mode=1,
            value_quant_mode=1,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=0,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            return_softmax_lse=False,
            key_dtype=None,
            value_dtype=None,
        ):
            if query.numel() == 0:
                raise ValueError("The shape size of query should not be 0")
            if layout_query == "BSND":
                attn_out_shape = list(query.shape)
                attn_out_shape[3] = attn_out_shape[3] - rope_head_dim
            else:
                attn_out_shape = list(query.shape)
                attn_out_shape[2] = attn_out_shape[2] - rope_head_dim
            attn_out = torch.empty(attn_out_shape, dtype=query.dtype, device="meta")
            if return_softmax_lse:
                if layout_query == "BSND":
                    kv_head_num = key.shape[2]
                    softmax_max = torch.empty(
                        [
                            query.shape[0],
                            kv_head_num,
                            query.shape[1],
                            query.shape[2] // kv_head_num,
                        ],
                        dtype=torch.float32,
                        device="meta",
                    )
                    softmax_sum = torch.empty(
                        [
                            query.shape[0],
                            kv_head_num,
                            query.shape[1],
                            query.shape[2] // kv_head_num,
                        ],
                        dtype=torch.float32,
                        device="meta",
                    )
                else:
                    if layout_kv == "PA_BSND":
                        kv_head_num = key.shape[2]
                    else:
                        kv_head_num = key.shape[1]
                    softmax_max = torch.empty(
                        [kv_head_num, query.shape[0], query.shape[1] // kv_head_num],
                        dtype=torch.float32,
                        device="meta",
                    )
                    softmax_sum = torch.empty(
                        [kv_head_num, query.shape[0], query.shape[1] // kv_head_num],
                        dtype=torch.float32,
                        device="meta",
                    )
            else:
                softmax_max = torch.empty([], dtype=torch.float32, device="meta")
                softmax_sum = torch.empty([], dtype=torch.float32, device="meta")
            return (attn_out, softmax_max, softmax_sum)


# Instantiate the builder
kv_quant_flash_attention_op_builder = KvQuantFlashAttentionOpBuilder()


@impl(AS_LIBRARY, kv_quant_flash_attention_op_builder.name, "PrivateUse1")
def kv_quant_flash_attention(
    query,
    key,
    value=None,
    sparse_indices=None,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sinks=None,
    scale_value=1.0,
    key_quant_mode=1,
    value_quant_mode=1,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=0,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    return_softmax_lse=False,
    key_dtype=None,
    value_dtype=None,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = kv_quant_flash_attention_op_builder.load()
    return op_module.kv_quant_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        sinks,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        return_softmax_lse,
        key_dtype,
        value_dtype,
    )
