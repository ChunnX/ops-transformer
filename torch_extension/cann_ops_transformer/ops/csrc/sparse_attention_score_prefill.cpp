/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attention_score_prefill.cpp
 * \brief aclnn binding for SparseAttentionScorePrefill custom op.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {

void CheckInt32Tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
        name, " dtype must be int32, got ", tensor.scalar_type());
}

} // namespace

at::Tensor npu_sparse_attention_score_prefill(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &block_table,
    const at::Tensor &k2q_row_ptr,
    const at::Tensor &k2q_q_indices,
    const at::Tensor &k2q_slot_indices,
    const at::Tensor &actual_seq_lengths,
    const at::Tensor &actual_seq_lengths_kv,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t block_size,
    int64_t top_k,
    int64_t inner_precise)
{
    TORCH_CHECK(block_size == 0 || block_table.has_value(),
        "block_table is required when block_size is non-zero");
    if (block_table.has_value()) {
        CheckInt32Tensor(block_table.value(), "block_table");
    }
    CheckInt32Tensor(k2q_row_ptr, "k2q_row_ptr");
    CheckInt32Tensor(k2q_q_indices, "k2q_q_indices");
    CheckInt32Tensor(k2q_slot_indices, "k2q_slot_indices");
    CheckInt32Tensor(actual_seq_lengths, "actual_seq_lengths");
    CheckInt32Tensor(actual_seq_lengths_kv, "actual_seq_lengths_kv");

    int64_t total_q_tokens = query.size(0);
    int64_t num_q_heads = query.size(1);
    int64_t D = query.size(2);

    at::Tensor attention_out = at::empty({total_q_tokens, num_q_heads, D},
                                         query.options().dtype(query.dtype()));
    at::Tensor softmax_lse = at::empty({0}, query.options().dtype(at::kFloat));
    bool softmax_lse_flag = false;

    ACLNN_CMD(aclnnSparseAttentionScorePrefill,
              query, key, value, block_table,
              k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
              actual_seq_lengths, actual_seq_lengths_kv,
              num_key_value_heads, scale_value, block_size, top_k,
              inner_precise, softmax_lse_flag, attention_out, softmax_lse);

    return attention_out;
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attention_score_prefill_with_lse(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &block_table,
    const at::Tensor &k2q_row_ptr,
    const at::Tensor &k2q_q_indices,
    const at::Tensor &k2q_slot_indices,
    const at::Tensor &actual_seq_lengths,
    const at::Tensor &actual_seq_lengths_kv,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t block_size,
    int64_t top_k,
    int64_t inner_precise)
{
    TORCH_CHECK(block_size == 0 || block_table.has_value(),
        "block_table is required when block_size is non-zero");
    if (block_table.has_value()) {
        CheckInt32Tensor(block_table.value(), "block_table");
    }
    CheckInt32Tensor(k2q_row_ptr, "k2q_row_ptr");
    CheckInt32Tensor(k2q_q_indices, "k2q_q_indices");
    CheckInt32Tensor(k2q_slot_indices, "k2q_slot_indices");
    CheckInt32Tensor(actual_seq_lengths, "actual_seq_lengths");
    CheckInt32Tensor(actual_seq_lengths_kv, "actual_seq_lengths_kv");

    const int64_t total_q_tokens = query.size(0);
    const int64_t num_q_heads = query.size(1);
    const int64_t head_dim = query.size(2);
    at::Tensor attention_out = at::empty(
        {total_q_tokens, num_q_heads, head_dim}, query.options().dtype(query.dtype()));
    at::Tensor softmax_lse = at::empty(
        {total_q_tokens, num_q_heads, 1}, query.options().dtype(at::kFloat));
    bool softmax_lse_flag = true;

    ACLNN_CMD(aclnnSparseAttentionScorePrefill,
              query, key, value, block_table,
              k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
              actual_seq_lengths, actual_seq_lengths_kv,
              num_key_value_heads, scale_value, block_size, top_k,
              inner_precise, softmax_lse_flag, attention_out, softmax_lse);

    return {attention_out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_attention_score_prefill",
          &npu_sparse_attention_score_prefill,
          "SparseAttentionScorePrefill forward");
    m.def("npu_sparse_attention_score_prefill_with_lse",
          &npu_sparse_attention_score_prefill_with_lse,
          "SparseAttentionScorePrefill forward with softmax LSE");
}

} // namespace op_api
