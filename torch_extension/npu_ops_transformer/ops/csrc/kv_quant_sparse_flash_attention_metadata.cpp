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

const c10::optional<at::Tensor> get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_kv_quant_sparse_flash_attention_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t kv_quant_mode, 
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &topk_length, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t topk, int64_t tile_size, int64_t rope_head_dim, int64_t mask_mode, int64_t win_left, int64_t win_right,
    c10::string_view layout_q, c10::string_view layout_kv)
{
    //const c10::string_view device = "npu:0";
    //at::Tensor output = at::empty({OUTPUT_SIZE}, at::TensorOptions().dtype(at::kInt32).
    //                              device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    //if (cu_seqlens_q.has_value()) {
    //    output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_q.value().device()));
    //} else if (cu_seqlens_kv.has_value()) {
    //    output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(cu_seqlens_kv.value().device()));
    //} else if (seqused_q.has_value()) {
    //    output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(seqused_q.value().device()));
    //} else if (seqused_kv.has_value()) {
    //    output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(seqused_kv.value().device()));
    //} else if (topk_length.has_value()) {
    //    output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(topk_length.value().device()));
    //}

    at::Device output_device = c10::DeviceType::PrivateUse1;
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_kv.has_value()) {
        output_device = cu_seqlens_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    } else if (topk_length.has_value()) {
        output_device = topk_length.value().device();
    }
    at::Tensor output = at::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device).
                                  memory_format(c10::MemoryFormat::Contiguous));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_kv_val = get_valid_tensor(cu_seqlens_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);
    auto topk_length_val = get_valid_tensor(topk_length, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    ACLNN_CMD(aclnnKvQuantSparseFlashAttentionMetadata, cu_seqlens_q_val, cu_seqlens_kv_val, seqused_q_val, 
              seqused_kv_val, topk_length_val, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, 
              max_seqlen_kv, topk, kv_quant_mode, tile_size, rope_head_dim, mask_mode, win_left, win_right, 
              layout_q_ptr, layout_kv_ptr, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_kv_quant_sparse_flash_attention_metadata", &npu_kv_quant_sparse_flash_attention_metadata, "npu_kv_quant_sparse_flash_attention_metadata");
}
} // namespace op_api