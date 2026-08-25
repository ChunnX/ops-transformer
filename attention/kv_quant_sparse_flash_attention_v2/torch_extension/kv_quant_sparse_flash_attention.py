# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

INT64_MAX = 9223372036854775807


class KvQuantSparseFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(KvQuantSparseFlashAttentionOpBuilder, self).__init__(
            "kv_quant_sparse_flash_attention", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/kv_quant_sparse_flash_attention.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "kv_quant_sparse_flash_attention(Tensor query, Tensor key, Tensor value, "
            "Tensor sparse_indices, float scale_value, int key_quant_mode, int value_quant_mode, *, "
            "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
            "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_kv=None, Tensor? sinks=None, "
            'int sparse_block_size=1, str layout_query="BSND", str layout_kv="BSND", '
            "int sparse_mode=3, int pre_tokens=" + str(INT64_MAX) + ", "
            "int next_tokens=" + str(INT64_MAX) + ", "
            "int attention_mode=0, int quant_scale_repo_mode=1, int tile_size=128, "
            "int rope_head_dim=64, bool return_softmax_lse=False, "
            "int? key_dtype=None, int? value_dtype=None) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def kv_quant_sparse_flash_attention_meta(
            query,
            key,
            value,
            sparse_indices,
            scale_value,
            key_quant_mode,
            value_quant_mode,
            *,
            key_dequant_scale=None,
            value_dequant_scale=None,
            block_table=None,
            actual_seq_lengths_query=None,
            actual_seq_lengths_kv=None,
            sinks=None,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=3,
            pre_tokens=INT64_MAX,
            next_tokens=INT64_MAX,
            attention_mode=0,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            return_softmax_lse=False,
            key_dtype=None,
            value_dtype=None,
        ):
            torch._check(query is not None, lambda: "query should not be None")
            torch._check(key is not None, lambda: "key should not be None")
            torch._check(value is not None, lambda: "value should not be None")
            torch._check(
                sparse_indices is not None, lambda: "sparse_indices should not be None"
            )
            torch._check(query.numel() > 0, lambda: "Input query should not be empty.")
            torch._check(key.numel() > 0, lambda: "Input key should not be empty.")
            torch._check(value.numel() > 0, lambda: "Input value should not be empty.")
            torch._check(
                sparse_indices.numel() > 0,
                lambda: "Input sparse_indices should not be empty.",
            )

            if layout_query == "BSND":
                torch._check(
                    query.dim() == 4,
                    lambda: "When the layout of query is BSND, the query dimension must be 4, but got "
                    + str(query.dim()),
                )
                attn_out_shape = [
                    query.shape[0],
                    query.shape[1],
                    query.shape[2],
                    query.shape[3] - rope_head_dim,
                ]
            else:
                torch._check(
                    query.dim() == 3,
                    lambda: "When the layout of query is TND, the query dimension must be 3, but got "
                    + str(query.dim()),
                )
                attn_out_shape = [
                    query.shape[0],
                    query.shape[1],
                    query.shape[2] - rope_head_dim,
                ]
            attn_out = torch.empty(attn_out_shape, dtype=query.dtype, device="meta")

            if return_softmax_lse:
                if layout_query == "BSND":
                    kv_head_num = key.shape[2]
                    lse_shape = [
                        query.shape[0],
                        kv_head_num,
                        query.shape[1],
                        query.shape[2] // kv_head_num,
                    ]
                else:
                    if layout_kv == "PA_BSND":
                        kv_head_num = key.shape[2]
                    else:
                        kv_head_num = key.shape[1]
                    lse_shape = [
                        kv_head_num,
                        query.shape[0],
                        query.shape[1] // kv_head_num,
                    ]
                softmax_max = torch.empty(lse_shape, dtype=torch.float32, device="meta")
                softmax_sum = torch.empty(lse_shape, dtype=torch.float32, device="meta")
            else:
                softmax_max = torch.empty([0], dtype=torch.float32, device="meta")
                softmax_sum = torch.empty([0], dtype=torch.float32, device="meta")

            return (attn_out, softmax_max, softmax_sum)


# Instantiate the builder
kv_quant_sparse_flash_attention_op_builder = KvQuantSparseFlashAttentionOpBuilder()
kv_quant_sparse_flash_attention_op_builder._ensure_initialized()


@impl(get_as_library(), kv_quant_sparse_flash_attention_op_builder.name, "PrivateUse1")
def kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sinks=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
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
    op_module = kv_quant_sparse_flash_attention_op_builder.load()
    return op_module.kv_quant_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        sinks,
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
