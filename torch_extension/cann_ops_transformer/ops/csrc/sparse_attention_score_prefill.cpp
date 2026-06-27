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
    const at::Tensor &block_table,
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
    CheckInt32Tensor(block_table, "block_table");
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

    ACLNN_CMD(aclnnSparseAttentionScorePrefill,
              query, key, value, block_table,
              k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
              actual_seq_lengths, actual_seq_lengths_kv,
              num_key_value_heads, scale_value, block_size, top_k,
              inner_precise, attention_out);

    return attention_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_attention_score_prefill",
          &npu_sparse_attention_score_prefill,
          "SparseAttentionScorePrefill forward");
}

} // namespace op_api
