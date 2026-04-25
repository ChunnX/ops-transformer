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
 * \file kv_quant_sparse_flash_attention_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
constexpr int64_t OUTPUT_SIZE = 1024;

const c10::optional<at::Tensor> qsfa_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_kv_quant_sparse_flash_attention_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t key_quant_mode, int64_t value_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t topk, int64_t sparse_block_size, 
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode, int64_t pre_tokens, 
    int64_t next_tokens, int64_t attention_mode, int64_t quant_scale_repo_mode, int64_t tile_size, int64_t rope_head_dim)
{
    const c10::string_view device = "npu:0";
    at::Device output_device = at::Device(std::string(device));
    if (actual_seq_lengths_query.has_value()) {
        output_device = actual_seq_lengths_query.value().device();
    } else if (actual_seq_lengths_kv.has_value()) {
        output_device = actual_seq_lengths_kv.value().device();
    }
    at::Tensor output = at::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device).memory_format(c10::MemoryFormat::Contiguous));

    auto actual_seq_lengths_query_val = qsfa_get_valid_tensor(actual_seq_lengths_query, output_device);
    auto actual_seq_lengths_kv_val = qsfa_get_valid_tensor(actual_seq_lengths_kv, output_device);

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    ACLNN_CMD(aclnnKvQuantSparseFlashAttentionMetadata, actual_seq_lengths_query_val, actual_seq_lengths_kv_val, 
              num_heads_q, num_heads_kv, head_dim, key_quant_mode, value_quant_mode, batch_size, max_seqlen_q, 
              max_seqlen_kv, topk, sparse_block_size, layout_query_ptr, layout_kv_ptr, sparse_mode, pre_tokens, 
              next_tokens, attention_mode, quant_scale_repo_mode, tile_size, rope_head_dim, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_kv_quant_sparse_flash_attention_metadata", &npu_kv_quant_sparse_flash_attention_metadata, "npu_kv_quant_sparse_flash_attention_metadata");
}
} // namespace op_api