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
 * \file sparse_flash_mla_grad.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_sparse_flash_mla_grad(const at::Tensor &q, const at::Tensor &dout, const at::Tensor &attn_out, 
               const at::Tensor &softmax_lse, const c10::optional<at::Tensor> &ori_kv, 
               const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
               const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &cu_seqlens_q,
               const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, 
               const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_ori_kv, 
               const c10::optional<at::Tensor> &seqused_cmp_kv, const c10::optional<at::Tensor> &cmp_residual_kv, 
               const c10::optional<at::Tensor> &ori_topk_length, const c10::optional<at::Tensor> &cmp_topk_length, 
               const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
               c10::optional<double> softmax_scale, c10::optional<int64_t> cmp_ratio, 
               c10::optional<int64_t> ori_mask_mode, c10::optional<int64_t> cmp_mask_mode,
               c10::optional<int64_t> ori_win_left, c10::optional<int64_t> ori_win_right, 
               c10::optional<c10::string_view> layout_q, 
               c10::optional<c10::string_view> layout_kv)
{
    const at::Tensor &ori_kv_const = ori_kv.value_or(at::Tensor());
    const at::Tensor &cmp_kv_const = cmp_kv.value_or(at::Tensor());
    const at::Tensor &ori_sparse_indices_const = ori_sparse_indices.value_or(at::Tensor());
    const at::Tensor &cmp_sparse_indices_const = cmp_sparse_indices.value_or(at::Tensor());
    const at::Tensor &cu_seqlens_q_const = cu_seqlens_q.value_or(at::Tensor());
    const at::Tensor &cu_seqlens_ori_kv_const = cu_seqlens_ori_kv.value_or(at::Tensor());
    const at::Tensor &cu_seqlens_cmp_kv_const = cu_seqlens_cmp_kv.value_or(at::Tensor());
    const at::Tensor &seqused_q_const = seqused_q.value_or(at::Tensor());
    const at::Tensor &seqused_ori_kv_const = seqused_ori_kv.value_or(at::Tensor());
    const at::Tensor &seqused_cmp_kv_const = seqused_cmp_kv.value_or(at::Tensor());
    const at::Tensor &cmp_residual_kv_const = cmp_residual_kv.value_or(at::Tensor());
    const at::Tensor &ori_topk_length_const = ori_topk_length.value_or(at::Tensor());
    const at::Tensor &cmp_topk_length_const = cmp_topk_length.value_or(at::Tensor());
    const at::Tensor &sinks_const = sinks.value_or(at::Tensor());
    const at::Tensor &metadata_const = metadata.value_or(at::Tensor());

    c10::string_view layout_q_str_view = layout_q.value_or("BSND");
    char *layout_q_ptr = const_cast<char *>(layout_q_str_view.data());

    c10::string_view layout_kv_str_view = layout_kv.value_or("BSND");
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str_view.data());

    at::Tensor dq = at::empty_like(q);
    at::Tensor dOriKv, dCmpKv;
    if (ori_kv_const.defined()) {
        dOriKv = at::empty_like(ori_kv_const);
    } else {
        dOriKv = at::empty({0}, q.options());
    }
    if (cmp_kv_const.defined()) {
        dCmpKv = at::empty_like(cmp_kv_const);
    } else {
        dCmpKv = at::empty({0}, q.options());
    }
    at::Tensor dSinks;
    if (sinks_const.defined()) {
        dSinks = at::empty_like(sinks_const);
    } else {
        dSinks = at::empty({0}, q.options().dtype(at::kFloat));
    }

    auto headDim = q.size(q.dim() - 1);

    double default_softmax = 1.0 / std::sqrt(headDim);
    const double softmax_scale_const = softmax_scale.value_or(default_softmax);
    const int64_t cmp_ratio_const = cmp_ratio.value_or(1);
    const int64_t ori_mask_mode_const = ori_mask_mode.value_or(0);
    const int64_t cmp_mask_mode_const = cmp_mask_mode.value_or(0);
    const int64_t ori_win_left_const = ori_win_left.value_or(-1);
    const int64_t ori_win_right_const = ori_win_right.value_or(-1);

    at::Tensor oriSoftmaxL1Norm;
    if (ori_sparse_indices_const.defined()) {
        oriSoftmaxL1Norm = at::empty(ori_sparse_indices_const.sizes(), ori_sparse_indices_const.options().dtype(at::kFloat));
    } else {
        oriSoftmaxL1Norm = at::empty({0}, q.options().dtype(at::kFloat));
    }

    at::Tensor cmpSoftmaxL1Norm;
    if (cmp_sparse_indices_const.defined()) {
        cmpSoftmaxL1Norm = at::empty(cmp_sparse_indices_const.sizes(), cmp_sparse_indices_const.options().dtype(at::kFloat));
    } else {
        cmpSoftmaxL1Norm = at::empty({0}, q.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnSparseFlashMlaGrad, q, dout, attn_out, softmax_lse, 
              ori_kv_const, cmp_kv_const, ori_sparse_indices_const, cmp_sparse_indices_const, 
              cu_seqlens_q_const, cu_seqlens_ori_kv_const, cu_seqlens_cmp_kv_const, 
              seqused_q_const, seqused_ori_kv_const, seqused_cmp_kv_const, cmp_residual_kv_const, 
              ori_topk_length_const, cmp_topk_length_const, sinks_const, metadata_const,
              softmax_scale_const, cmp_ratio_const, ori_mask_mode_const, cmp_mask_mode_const, 
              ori_win_left_const, ori_win_right_const, 
              layout_q_ptr, layout_kv_ptr,
              dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                        dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla_grad", &npu_sparse_flash_mla_grad, "npu_sparse_flash_mla_grad");
}
} // namespace op_api