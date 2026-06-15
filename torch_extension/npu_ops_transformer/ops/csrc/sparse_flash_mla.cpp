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
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

void CheckQueryShape(const at::Tensor &q, const std::string &layoutQ)
{
    TORCH_CHECK(layoutQ == "BSND" || layoutQ == "TND",
                "The layout of query only supports BSND and TND, but got ", layoutQ);
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.");
    for (int64_t i = 0; i < q.dim(); ++i) {
        TORCH_CHECK(q.size(i) > 0, "All values within query's shape should be greater than 0, but shape[",
                    i, "] is ", q.size(i));
    }
    if (layoutQ == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", q.dim());
    } else {
        TORCH_CHECK(q.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ", q.dim());
    }
}

int64_t GetKvHeadNum(const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
                     const std::string &layoutKv)
{
    TORCH_CHECK(oriKv.has_value() || cmpKv.has_value(),
                "ori_kv or cmp_kv is required when return_softmax_lse is true.");
    const at::Tensor &kv = oriKv.has_value() ? oriKv.value() : cmpKv.value();
    if (layoutKv == "TND") {
        return kv.size(DIM_1);
    }
    return kv.size(DIM_2);
}

std::tuple<at::Tensor, at::Tensor> MakeSparseFlashMlaOutputs(const at::Tensor &q,
                                                             const c10::optional<at::Tensor> &oriKv,
                                                             const c10::optional<at::Tensor> &cmpKv,
                                                             const std::string &layoutQ,
                                                             const std::string &layoutKv,
                                                             bool returnSoftmaxLse)
{
    CheckQueryShape(q, layoutQ);
    at::Tensor attnOut = at::empty_like(q);
    at::Tensor softmaxLse;
    if (!returnSoftmaxLse) {
        softmaxLse = at::empty({}, q.options().dtype(torch::kFloat32));
        return {attnOut, softmaxLse};
    }

    int64_t kvHeadNum = GetKvHeadNum(oriKv, cmpKv, layoutKv);
    if (layoutQ == "BSND") {
        softmaxLse = at::empty({q.size(DIM_0), kvHeadNum, q.size(DIM_1), q.size(DIM_2) / kvHeadNum},
                               q.options().dtype(torch::kFloat32));
    } else {
        softmaxLse = at::empty({kvHeadNum, q.size(DIM_0), q.size(DIM_1) / kvHeadNum},
                               q.options().dtype(torch::kFloat32));
    }
    return {attnOut, softmaxLse};
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_mla(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &ori_sparse_indices, const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &ori_block_table, const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv, const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv, const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode,
    int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q,
    c10::string_view layout_kv, int64_t topk_value_mode, bool return_softmax_lse)
{
    std::string layoutQStr = std::string(layout_q);
    std::string layoutKvStr = std::string(layout_kv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    auto outputs = MakeSparseFlashMlaOutputs(q, ori_kv, cmp_kv, layoutQStr, layoutKvStr, return_softmax_lse);
    at::Tensor attnOut = std::get<0>(outputs);
    at::Tensor softmaxLse = std::get<1>(outputs);

    ACLNN_CMD(aclnnSparseFlashMla, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices, ori_block_table,
              cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv,
              seqused_cmp_kv, cmp_residual_kv, ori_topk_length, cmp_topk_length, sinks, metadata, softmax_scale,
              cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layoutQPtr, layoutKvPtr,
              topk_value_mode, return_softmax_lse, attnOut, softmaxLse);
    return {attnOut, softmaxLse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla", &npu_sparse_flash_mla, "npu_sparse_flash_mla");
}
} // namespace op_api
