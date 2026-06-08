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
 * \file sparse_flash_mla_grad_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor npu_sparse_flash_mla_grad_metadata(
               int64_t num_heads_q,
               int64_t num_heads_kv,
               int64_t head_dim,
               const c10::optional<at::Tensor> &cu_seqlens_q,
               const c10::optional<at::Tensor> &cu_seqlens_ori_kv, 
               const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, 
               const c10::optional<at::Tensor> &seqused_q, 
               const c10::optional<at::Tensor> &seqused_ori_kv, 
               const c10::optional<at::Tensor> &seqused_cmp_kv, 
               const c10::optional<at::Tensor> &cmp_residual_kv, 
               const c10::optional<at::Tensor> &ori_topk_length, 
               const c10::optional<at::Tensor> &cmp_topk_length,
               c10::optional<int64_t> batch_size, 
               c10::optional<int64_t> max_seqlen_q,
               c10::optional<int64_t> max_seqlen_ori_kv,
               c10::optional<int64_t> max_seqlen_cmp_kv,
               c10::optional<int64_t> ori_topk,
               c10::optional<int64_t> cmp_topk,
               c10::optional<int64_t> cmp_ratio,
               c10::optional<int64_t> ori_mask_mode, 
               c10::optional<int64_t> cmp_mask_mode,
               c10::optional<int64_t> ori_win_left, 
               c10::optional<int64_t> ori_win_right, 
               c10::optional<c10::string_view> layout_q, 
               c10::optional<c10::string_view> layout_kv,
               c10::optional<bool> has_ori_kv,
               c10::optional<bool> has_cmp_kv)
{
    at::Tensor metadata;
    return metadata;
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla_grad_metadata", &npu_sparse_flash_mla_grad_metadata, "npu_sparse_flash_mla_grad_metadata");
}
} // namespace op_api