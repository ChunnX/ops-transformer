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
 * \file quant_block_sparse_attention_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
constexpr int64_t OUTPUT_SIZE = 2048;

const c10::optional<at::Tensor> qbsa_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt,
                                                      at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_quant_block_sparse_attention_metadata(
    const at::Tensor &sparse_seq_len, int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,const c10::optional<at::Tensor> &cu_seqlens_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size,  int64_t sparse_block_size_q, int64_t sparse_block_size_kv, int64_t quant_mode,
    int64_t mask_mode, int64_t max_seqlen_q, int64_t max_seqlen_kv, c10::string_view layout_q,
    c10::string_view layout_kv, c10::string_view layout_sparse_indices)
{
    at::Device output_device = sparse_seq_len.device();
    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto sparse_seq_len_val = qbsa_get_valid_tensor(sparse_seq_len, output_device);
    auto cu_seqlens_q_val = qbsa_get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_kv_val = qbsa_get_valid_tensor(cu_seqlens_kv, output_device);
    auto seqused_q_val = qbsa_get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = qbsa_get_valid_tensor(seqused_kv, output_device);

    // convert str
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    std::string layout_sparse_indices_str = std::string(layout_sparse_indices);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_sparse_indices_ptr = const_cast<char *>(layout_sparse_indices_str.c_str());

    ACLNN_CMD(aclnnQuantBlockSparseAttentionMetadata, sparse_seq_len, cu_seqlens_q_val, cu_seqlens_kv_val,
              seqused_q_val, seqused_kv_val, batch_size, num_heads_q, num_heads_kv, head_dim, sparse_block_size_q,
              sparse_block_size_kv, quant_mode, mask_mode, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_kv_ptr,
              layout_sparse_indices_ptr, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_quant_block_sparse_attention_metadata", &npu_quant_block_sparse_attention_metadata,
          "npu_quant_block_sparse_attention_metadata");
}
} // namespace op_api
