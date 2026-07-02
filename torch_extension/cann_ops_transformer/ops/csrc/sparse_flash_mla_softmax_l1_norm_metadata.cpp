/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <string>
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t SMLA_METADATA_SIZE = 64;

at::Device get_sparse_flash_mla_softmax_l1_norm_metadata_device(
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &topk_length)
{
    if (cu_seqlens_q.has_value() && cu_seqlens_q.value().defined()) {
        return cu_seqlens_q.value().device();
    }
    if (cu_seqlens_k.has_value() && cu_seqlens_k.value().defined()) {
        return cu_seqlens_k.value().device();
    }
    if (seqused_q.has_value() && seqused_q.value().defined()) {
        return seqused_q.value().device();
    }
    if (seqused_k.has_value() && seqused_k.value().defined()) {
        return seqused_k.value().device();
    }
    if (cmp_residual_k.has_value() && cmp_residual_k.value().defined()) {
        return cmp_residual_k.value().device();
    }
    if (topk_length.has_value() && topk_length.value().defined()) {
        return topk_length.value().device();
    }
    return at::Device(torch_npu::utils::get_npu_device_type());
}

at::Tensor sparse_flash_mla_softmax_l1_norm_metadata(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &topk_length,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t topk,
    std::string layout_q,
    std::string layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio)
{
    TORCH_CHECK((num_heads_q > 0), "The num_heads_q should be greater than 0, current is: ", num_heads_q);
    TORCH_CHECK((num_heads_k > 0), "The num_heads_k should be greater than 0, current is: ", num_heads_k);
    TORCH_CHECK((head_dim > 0), "The head_dim should be greater than 0, current is: ", head_dim);
    TORCH_CHECK((num_heads_q % num_heads_k == 0),
                "num_heads_q should be divisible by num_heads_k, but got nq=", num_heads_q, " nk=", num_heads_k);
    TORCH_CHECK((cmp_ratio >= 1 && cmp_ratio <= 128),
                "cmp_ratio should be in [1, 128], current is: ", cmp_ratio);

    at::Tensor output{nullptr};
    {
        auto local_device = get_sparse_flash_mla_softmax_l1_norm_metadata_device(
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, topk_length);
        const c10::OptionalDeviceGuard device_guard(local_device);
        output = at::empty(
            {SMLA_METADATA_SIZE},
            at::TensorOptions().dtype(at::kInt).device(local_device));
    }

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaSoftmaxL1NormMetadata,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, topk_length,
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k, head_dim, topk, cmp_ratio, mask_mode,
        layout_q_ptr, layout_k_ptr, output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "sparse_flash_mla_softmax_l1_norm_metadata",
        &sparse_flash_mla_softmax_l1_norm_metadata,
        "sparse_flash_mla_softmax_l1_norm_metadata");
}

}  // namespace op_api
