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
 * \file sparse_flash_mla_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
constexpr int64_t OUTPUT_SIZE = 1024;

const c10::optional<at::Tensor> qsas_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_sparse_flash_mla_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length,
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t ori_topk, int64_t cmp_topk, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv,
    bool has_ori_kv, bool has_cmp_kv, int64_t batch_invariant)
{
    const c10::string_view device = "npu";
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_ori_kv.has_value()) {
        output_device = seqused_ori_kv.value().device();
    } else if (seqused_cmp_kv.has_value()) {
        output_device = seqused_cmp_kv.value().device();
    } else if (cmp_residual_kv.has_value()) {
        output_device = cmp_residual_kv.value().device();
    } else if (ori_topk_length.has_value()) {
        output_device = ori_topk_length.value().device();
    } else if (cmp_topk_length.has_value()) {
        output_device = cmp_topk_length.value().device();
    }

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto cu_seqlens_q_val = qsas_get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = qsas_get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = qsas_get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = qsas_get_valid_tensor(seqused_q, output_device);
    auto seqused_ori_kv_val = qsas_get_valid_tensor(seqused_ori_kv, output_device);
    auto seqused_cmp_kv_val = qsas_get_valid_tensor(seqused_cmp_kv, output_device);
    auto cmp_residual_kv_val = qsas_get_valid_tensor(cmp_residual_kv, output_device);
    auto ori_topk_length_val = qsas_get_valid_tensor(ori_topk_length, output_device);
    auto cmp_topk_length_val = qsas_get_valid_tensor(cmp_topk_length, output_device);

    // convert str
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaMetadata,
              cu_seqlens_q_val, cu_seqlens_ori_kv_val, cu_seqlens_cmp_kv_val,
              seqused_q_val, seqused_ori_kv_val, seqused_cmp_kv_val,
              cmp_residual_kv_val, ori_topk_length_val, cmp_topk_length_val,
              num_heads_q, num_heads_kv, head_dim,
              batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk,
              cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
              layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv, batch_invariant,
              output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla_metadata", &npu_sparse_flash_mla_metadata, 
    "npu_sparse_flash_mla_metadata");
}
} // namespace op_api
