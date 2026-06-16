/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor npu_sparse_flash_mla_metadata(
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv, const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv, const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length, int64_t num_heads_q, int64_t num_heads_kv,
    int64_t head_dim, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_ori_kv,
    int64_t max_seqlen_cmp_kv, int64_t ori_topk, int64_t cmp_topk, int64_t cmp_ratio, int64_t ori_mask_mode,
    int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q,
    c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv, const at::Tensor &metadata)
{
    std::string layoutQStr = std::string(layout_q);
    std::string layoutKvStr = std::string(layout_kv);
    const char *layoutQPtr = layoutQStr.c_str();
    const char *layoutKvPtr = layoutKvStr.c_str();

    ACLNN_CMD(aclnnSparseFlashMlaMetadata, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
              seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_topk_length, cmp_topk_length, num_heads_q,
              num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv, ori_topk,
              cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layoutQPtr, layoutKvPtr,
              has_ori_kv, has_cmp_kv, metadata);
    return metadata;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla_metadata", &npu_sparse_flash_mla_metadata, "npu_sparse_flash_mla_metadata");
}
} // namespace op_api
