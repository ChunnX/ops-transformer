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
 * \file lightning_indexer_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
constexpr int64_t OUTPUT_SIZE = 1024;

const c10::optional<at::Tensor> li_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_lightning_indexer_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode, 
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key, 
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, const c10::string_view layout_query, 
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, 
    int64_t cmp_ratio)
{
    const c10::string_view device = "npu:0";
    at::Device output_device = at::Device(std::string(device));
    if (actual_seq_lengths_query.has_value()) {
        output_device = actual_seq_lengths_query.value().device();
    } else if (actual_seq_lengths_key.has_value()) {
        output_device = actual_seq_lengths_key.value().device();
    }

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto actual_seq_lengths_query_val = li_get_valid_tensor(actual_seq_lengths_query, output_device);
    auto actual_seq_lengths_key_val = li_get_valid_tensor(actual_seq_lengths_key, output_device);

    // convert str
    std::string layout_query_str = std::string(layout_query);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    std::string layout_key_str = std::string(layout_key);
    char *layout_key_ptr = const_cast<char *>(layout_key_str.c_str());

    ACLNN_CMD(aclnnLightningIndexerMetadata, actual_seq_lengths_query_val, actual_seq_lengths_key_val,
              num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, batch_size, 
              max_seqlen_q, max_seqlen_k, layout_query_ptr, layout_key_ptr, sparse_count, 
              sparse_mode, pre_tokens, next_tokens, cmp_ratio, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_lightning_indexer_metadata", &npu_lightning_indexer_metadata, "npu_lightning_indexer_metadata");
}
} // namespace op_api