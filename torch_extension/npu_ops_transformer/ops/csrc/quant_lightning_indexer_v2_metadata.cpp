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
 * \file quant_lightning_indexer_v2_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
constexpr int64_t OUTPUT_SIZE = 1024;

const c10::optional<at::Tensor> qli_v2_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_quant_lightning_indexer_v2_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t topk, int64_t q_quant_mode,
    int64_t k_quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, c10::string_view layout_q,
    c10::string_view layout_k, int64_t mask_mode, int64_t cmp_ratio)
{
    const c10::string_view device = "npu";
    at::Device output_device = at::Device(std::string(device));
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

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto cu_seqlens_q_val = qli_v2_get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_k_val = qli_v2_get_valid_tensor(cu_seqlens_k, output_device);
    auto seqused_q_val = qli_v2_get_valid_tensor(seqused_q, output_device);
    auto seqused_k_val = qli_v2_get_valid_tensor(seqused_k, output_device);
    auto cmp_residual_k_val = qli_v2_get_valid_tensor(cmp_residual_k, output_device);

    // convert str
    std::string layout_q_str = std::string(layout_q);
    std::string layout_k_str = std::string(layout_k);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k_str.c_str());

    ACLNN_CMD(aclnnQuantLightningIndexerV2Metadata, cu_seqlens_q_val, cu_seqlens_k_val, seqused_q_val, seqused_k_val,
              cmp_residual_k_val,
              num_heads_q, num_heads_k, head_dim, topk, q_quant_mode, k_quant_mode,
              batch_size, max_seqlen_q, max_seqlen_k, layout_q_ptr, layout_k_ptr, mask_mode, cmp_ratio,
              output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_quant_lightning_indexer_v2_metadata", &npu_quant_lightning_indexer_v2_metadata,
        "npu_quant_lightning_indexer_v2_metadata");
}
} // namespace op_api