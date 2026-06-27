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


class SparseAttentionScorePrefillOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_sparse_attention_score_prefill")

    def sources(self):
        return ['ops/csrc/sparse_attention_score_prefill.cpp']

    def schema(self) -> str:
        return (
            "npu_sparse_attention_score_prefill("
            "Tensor query, Tensor key, Tensor value, "
            "Tensor block_table, "
            "Tensor k2q_row_ptr, Tensor k2q_q_indices, "
            "Tensor k2q_slot_indices, "
            "Tensor actual_seq_lengths, Tensor actual_seq_lengths_kv, "
            "int num_key_value_heads, float scale_value, "
            "int block_size, int top_k, int inner_precise=4"
            ") -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_attention_score_prefill_meta(
            query, key, value, block_table,
            k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
            actual_seq_lengths, actual_seq_lengths_kv,
            num_key_value_heads, scale_value,
            block_size, top_k, inner_precise=4
        ):
            total_q_tokens = query.size(0)
            num_q_heads = query.size(1)
            D = query.size(2)
            return torch.empty(
                (total_q_tokens, num_q_heads, D),
                dtype=query.dtype, device='meta')


sparse_attn_prefill_builder = SparseAttentionScorePrefillOpBuilder()
op_module = sparse_attn_prefill_builder.load()


@impl(AS_LIBRARY, sparse_attn_prefill_builder.name, "PrivateUse1")
def npu_sparse_attention_score_prefill(
    query, key, value, block_table,
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
    actual_seq_lengths, actual_seq_lengths_kv,
    num_key_value_heads, scale_value,
    block_size, top_k, inner_precise=4
):
    return op_module.npu_sparse_attention_score_prefill(
        query, key, value, block_table,
        k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_key_value_heads, scale_value,
        block_size, top_k, inner_precise)
