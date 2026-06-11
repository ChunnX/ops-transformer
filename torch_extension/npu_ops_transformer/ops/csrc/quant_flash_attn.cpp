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
 * \file quant_flash_attn.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t FLOAT4_E2M1 = 296;


/**
 * @brief Warpper for moe_distribute_combine_v5
 */
std::tuple<at::Tensor, at::Tensor>
npu_quant_flash_attn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale,
                int64_t q_quant_mode, int64_t k_quant_mode, int64_t v_quant_mode,
                const c10::optional<at::Tensor> &block_able,
                const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
                const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
                const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
                const c10::optional<at::Tensor> &metadata,
                int64_t q_dtype, int64_t k_dtype, int64_t v_dtype,
                int64_t q_descale_dtype, int64_t k_descale_dtype, int64_t v_descale_dtype,
                int64_t quant_block_size_qs, int64_t quant_block_size_ks, int64_t quant_block_size_vs,
                double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right,
                int64_t max_seqlen_q, int64_t max_seqlen_kv,
                string layout_q, string layout_kv, string layout_out,
                int64_t softmax_precision, int64_t return_softmax_lse
                )
{ // 先不判断
    const c10::string_view device = "npu";
    at::Device outputDevice = at::Device(std::string(device));
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
    at::Tensor softmaxLse = at::empty(softmaxOutSize, torch::dtype(at::kFloat).device(outputDevice));

    int32_t qDtypeRatio = 1;
    if (q_dtype == FLOAT4_E2M1) {
        qDtypeRatio = 2;
    }
    if (layout_out == "TND") {
        attentionOutSize = {tSize, nSize, qDtypeRatio * dSize};
    } else if (layout_out == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, qDtypeRatio * dSize};
    } else {
        attentionOutSize = {bSize, sSize, nSize, qDtypeRatio * dSize};
    }
    at::Tensor attentionOutput = at::empty(attentionOutSize, torch::dtype(at::kBFloat16).device(outputDevice));

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out.c_str());

    TensorWrapper q_wrapper = make_wrapper(q, q_dtype);
    TensorWrapper k_wrapper = make_wrapper(k, k_dtype);
    TensorWrapper v_wrapper = make_wrapper(v, v_dtype);
    TensorWrapper q_descale_wrapper = make_wrapper(q_descale, q_descale_dtype);
    TensorWrapper k_descale_wrapper = make_wrapper(k_descale, k_descale_dtype);
    TensorWrapper v_descale_wrapper = make_wrapper(v_descale, v_descale_dtype);
    // 考虑直接调，对照aclnn接口
    ACLNN_CMD(aclnnQuantFlashAttn, q_wrapper, k_wrapper, v_wrapper, q_descale_wrapper, k_descale_wrapper, v_descale_wrapper, block_able, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata,
              q_quant_mode, k_quant_mode, v_quant_mode,
              quant_block_size_qs, quant_block_size_ks, quant_block_size_vs,
              softmax_scale, mask_mode, win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_kv_ptr,
              layout_out_ptr, softmax_precision, return_softmax_lse, attentionOutput, softmaxLse);

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_quant_flash_attn", &npu_quant_flash_attn, "quant_flash_attn");
}
} // namespace op_api
