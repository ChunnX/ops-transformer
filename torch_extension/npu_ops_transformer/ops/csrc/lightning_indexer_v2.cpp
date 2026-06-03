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
* \file lightning_indexer_v2.cpp
* \brief
*/

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> construct_lightning_indexer_v2_output_tensor(const at::Tensor& query,
                                                                                const at::Tensor& key,
                                                                                int64_t topk,
                                                                                std::string query_layout_str,
                                                                                std::string key_layout_str,
                                                                                bool return_value)
{
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, topk};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), topk};
    }
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = at::empty(output_size, at::kFloat);
    } else {
        sparse_values_out = at::empty({0}, at::kFloat);
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer_v2(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w,
    int64_t topk,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmpResidualK,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &output_idx_offset,
    const c10::optional<at::Tensor> &metadata,
    int64_t max_seqlen_q,
    c10::string_view layout_q, c10::string_view layout_k,
    int64_t mask_mode, int64_t cmp_ratio, int64_t return_value)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")

    std::string query_layout_str = std::string(layout_q);
    std::string key_layout_str = std::string(layout_k);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> lightning_indexer_v2_output
        = construct_lightning_indexer_v2_output_tensor(q, k, topk, query_layout_str,
                                                    key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(lightning_indexer_v2_output);
    at::Tensor sparse_values_out = std::get<1>(lightning_indexer_v2_output);
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    ACLNN_CMD(aclnnLightningIndexerV2, q, k, w, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmpResidualK,
              block_table, output_idx_offset, metadata, topk, max_seqlen_q, query_layout_ptr, key_layout_ptr,
              mask_mode, cmp_ratio, return_value, sparse_indices_out, sparse_values_out);
    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_lightning_indexer_v2", &npu_lightning_indexer_v2, "lightning_indexer_v2");
}
} // namespace op_api