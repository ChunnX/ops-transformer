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
* \file lightning_indexer.cpp
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

constexpr int64_t LI_V2_METADATA_SIZE = 1024;

at::Tensor lightning_indexer_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t topk,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k,
    c10::string_view layout_q, c10::string_view layout_k, int64_t mask_mode, int64_t cmp_ratio)
{
    at::Device output_device = at::Device(std::string("npu"));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_k.has_value()) {
        output_device = cu_seqlens_k.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_k.has_value()) {
        output_device = seqused_k.value().device();
    } else if (cmp_residual_k.has_value()) {
        output_device = cmp_residual_k.value().device();
    }

    at::Tensor output = torch::empty({LI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_k_val = get_valid_tensor(cu_seqlens_k, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_k_val = get_valid_tensor(seqused_k, output_device);
    auto cmp_residual_k_val = get_valid_tensor(cmp_residual_k, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_k_str = std::string(layout_k);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k_str.c_str());

    ACLNN_CMD(aclnnLightningIndexerV2Metadata, cu_seqlens_q_val, cu_seqlens_k_val, seqused_q_val, seqused_k_val,
              cmp_residual_k_val, num_heads_q, num_heads_k, head_dim, topk, batch_size, max_seqlen_q, max_seqlen_k,
              layout_q_ptr, layout_k_ptr, mask_mode, cmp_ratio, output);
    return output;
}

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> construct_lightning_indexer_output_tensor(const at::Tensor& query,
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
        sparse_values_out = at::empty(output_size, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> lightning_indexer(
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
    std::tuple<at::Tensor, at::Tensor> lightning_indexer_output
        = construct_lightning_indexer_output_tensor(q, k, topk, query_layout_str,
                                                    key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(lightning_indexer_output);
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
    m.def("lightning_indexer_metadata", &lightning_indexer_metadata, "lightning_indexer_metadata");
    m.def("lightning_indexer", &lightning_indexer, "lightning_indexer");
}
} // namespace op_api