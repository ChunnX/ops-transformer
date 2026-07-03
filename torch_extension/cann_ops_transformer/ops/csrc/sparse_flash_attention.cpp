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
 * \file sparse_flash_attention.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;
constexpr int64_t SIZE = 8;

at::Tensor construct_sparse_flash_attention_output_tensor(const at::Tensor &query, const std::string &layout_query)
{
    TORCH_CHECK(layout_query == "BSND" || layout_query == "TND",
                "The layout of query only support BSND and TND, but got ", layout_query);
    for (int64_t i = 0; i < query.dim(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater than 0, but shape[", i,
                    "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ", query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }
    return at::empty(output_size, query.options().dtype(query.dtype()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention_v2(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, bool return_softmax_lse,
    const c10::optional<at::Tensor> &sinks)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    at::Tensor attention_out = construct_sparse_flash_attention_output_tensor(query, layout_query_str);
    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::SmallVector<int64_t, SIZE> softmax_max_size;
    at::SmallVector<int64_t, SIZE> softmax_sum_size;
    if (return_softmax_lse) {
        if (query.dim() == DIM_3) {
            softmax_max_size = {key.size(DIM_1), query.size(DIM_0), query.size(DIM_1) / key.size(DIM_1)};
            softmax_sum_size = {key.size(DIM_1), query.size(DIM_0), query.size(DIM_1) / key.size(DIM_1)};
        } else {
            softmax_max_size = {query.size(DIM_0), key.size(DIM_2), query.size(DIM_1),
                                query.size(DIM_2) / key.size(DIM_2)};
            softmax_sum_size = {query.size(DIM_0), key.size(DIM_2), query.size(DIM_1),
                                query.size(DIM_2) / key.size(DIM_2)};
        }
    } else {
        softmax_max_size = {0};
        softmax_sum_size = {0};
    }
    softmax_max = at::empty(softmax_max_size, query.options().dtype(torch::kFloat32));
    softmax_sum = at::empty(softmax_sum_size, query.options().dtype(torch::kFloat32));

    ACLNN_CMD(aclnnSparseFlashAttentionV2, query,
              key, value, sparse_indices, block_table, actual_seq_lengths_query, actual_seq_lengths_kv, query_rope,
              key_rope, sinks, scale_value, sparse_block_size, layout_query_ptr, layout_kv_ptr, sparse_mode,
              pre_tokens, next_tokens, attention_mode, return_softmax_lse, attention_out, softmax_max, softmax_sum);

    return std::make_tuple(attention_out, softmax_max, softmax_sum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_attention_v2", &npu_sparse_flash_attention_v2, "npu_sparse_flash_attention_v2");
}
} // namespace op_api
