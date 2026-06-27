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

at::Tensor npu_sparse_attention_score_prefill(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &select_idx,
    const at::Tensor &block_table,
    const at::Tensor &kv_to_q_select_idx,
    const at::Tensor &kv_to_q_select_num,
    const at::Tensor &kv_to_q_slot_idx,
    const at::Tensor &kv_block_valid_size,
    const at::Tensor &actual_seq_lengths,
    const at::Tensor &actual_seq_lengths_kv,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t block_size,
    int64_t top_k,
    int64_t inner_precise)
{
    // Output shape: same as query [total_q_tokens, num_q_heads, D]
    int64_t total_q_tokens = query.size(0);
    int64_t num_q_heads = query.size(1);
    int64_t D = query.size(2);

    at::Tensor attention_out = at::empty({total_q_tokens, num_q_heads, D},
                                         query.options().dtype(query.dtype()));

    ACLNN_CMD(aclnnSparseAttentionScorePrefill,
              query, key, value, select_idx, block_table,
              kv_to_q_select_idx, kv_to_q_select_num,
              kv_to_q_slot_idx, kv_block_valid_size,
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
