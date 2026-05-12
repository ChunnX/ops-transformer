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
 * \file npu_flash_attn.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
/**
 * @brief Warpper for moe_distribute_combine_v5
 */
std::tuple<at::Tensor, at::Tensor>
npu_flash_attn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
               const c10::optional<at::Tensor> &block_able, const c10::optional<at::Tensor> &cu_seqlens_q,
               const c10::optional<at::Tensor> &cu_seqlens_kv, const c10::optional<at::Tensor> &seqused_q,
               const c10::optional<at::Tensor> &seqused_kv, const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
               const c10::optional<at::Tensor> &metadata, double softmax_scale, int64_t mask_mode, int64_t win_left,
               int64_t win_right, int64_t max_seqlen_q, int64_t max_seqlen_kv, string layout_q, string layout_kv,
               string layout_out, int64_t return_softmax_lse, int64_t deterministic)
{ // 先不判断
    int64_t tSize = 0;
    int64_t nSize = 0;
    int64_t dSize = 0;
    int64_t sSize = 0;
    int64_t bSize = 0;
    at::SmallVector<int64_t, MAX_DIM_SIZE> attentionOutSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> softmaxOutSize;
    if (layout_q == "TND") {
        tSize = q.size(0);
        nSize = q.size(1);
        dSize = v.size(2);
    } else if (layout_q == "BSND") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2);
        dSize = v.size(3);
    } else {
        bSize = q.size(0);
        nSize = q.size(1);
        sSize = q.size(2);
        dSize = v.size(3);
    }
    if (return_softmax_lse) {
        if (q.dim() == DIM_THREE) {
            softmaxOutSize = {nSize, tSize};
        } else {
            softmaxOutSize = {bSize, nSize, sSize};
        }
    } else {
        softmaxOutSize = {0};
    }
    at::Tensor softmaxLse = at::empty(softmaxOutSize, q.options().dtype(at::kFloat));

    if (layout_out == "TND") {
        attentionOutSize = {tSize, nSize, dSize};
    } else if (layout_out == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, dSize};
    } else {
        attentionOutSize = {bSize, sSize, nSize, dSize};
    }
    at::Tensor attentionOutput = at::empty(attentionOutSize, q.options().dtype(q.dtype()));

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out.c_str());

    // 考虑直接调，对照aclnn接口
    ACLNN_CMD(aclnnFlashAttn, q, k, v, block_able, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata,
              softmax_scale, mask_mode, win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_kv_ptr,
              layout_out_ptr, return_softmax_lse, deterministic, attentionOutput, softmaxLse);

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_flash_attn", &npu_flash_attn, "flash_attn");
}
} // namespace op_api