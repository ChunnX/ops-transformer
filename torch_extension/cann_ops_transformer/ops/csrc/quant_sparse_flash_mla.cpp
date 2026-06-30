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
 * \file quant_sparse_flash_mla.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

constexpr int64_t QSMLA_METADATA_SIZE = 1024;

inline TensorWrapper qsmla_make_wrapper(const at::Tensor &tensor, aclDataType tensor_acltype)
{
    return {tensor, tensor_acltype};
}

at::Tensor quant_sparse_flash_mla_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t qkv_quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv, const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv, const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length, int64_t batch_size, int64_t max_seqlen_q,
    int64_t max_seqlen_ori_kv, int64_t max_seqlen_cmp_kv, int64_t ori_topk, int64_t cmp_topk,
    int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv)
{
    at::Device output_device = at::Device(std::string("npu"));
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

    at::Tensor output = torch::empty({QSMLA_METADATA_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_ori_kv_val = get_valid_tensor(seqused_ori_kv, output_device);
    auto seqused_cmp_kv_val = get_valid_tensor(seqused_cmp_kv, output_device);
    auto cmp_residual_kv_val = get_valid_tensor(cmp_residual_kv, output_device);
    auto ori_topk_length_val = get_valid_tensor(ori_topk_length, output_device);
    auto cmp_topk_length_val = get_valid_tensor(cmp_topk_length, output_device);

    // convert str
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    ACLNN_CMD(aclnnQuantSparseFlashMlaMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val, cu_seqlens_cmp_kv_val,
              seqused_q_val, seqused_ori_kv_val, seqused_cmp_kv_val, cmp_residual_kv_val, ori_topk_length_val,
              cmp_topk_length_val, num_heads_q, num_heads_kv, head_dim, qkv_quant_mode, batch_size, max_seqlen_q,
              max_seqlen_ori_kv, max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio, ori_mask_mode,
              cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv,
              output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> construct_quant_sparse_flash_mla_atten_out_tensor(
    const at::Tensor& q, const at::Tensor& ori_kv, std::string layout_q_str, bool return_softmax_lse)
{
    TORCH_CHECK(layout_q_str == "BSND" || layout_q_str == "TND",
                "The layout of query only support BSND and TND, but got ", layout_q_str);
    for (auto i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", q.size(i));
    }
    at::SmallVector<int64_t, SIZE> atten_out_size;
    at::SmallVector<int64_t, SIZE> softmax_lse_size;
    if (layout_q_str == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", q.dim());
        atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3)};
    } else {
        TORCH_CHECK(q.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ", q.dim());
        atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2)};
    }
    at::Tensor atten_out = at::empty(atten_out_size, q.options().dtype(torch::kBFloat16));
    at::Tensor softmax_lse;

    if (return_softmax_lse) {
        if (layout_q_str == "BSND") {
            // 对齐 Python: [q.shape[0], ori_kv.shape[2], q.shape[1], q.shape[2] // ori_kv.shape[2]]
            softmax_lse_size = {
                q.size(DIM_0),
                ori_kv.size(DIM_2),
                q.size(DIM_1),
                q.size(DIM_2) / ori_kv.size(DIM_2)
            };
        } else {
            // 对齐 Python: [ori_kv.shape[1], q.shape[0], q.shape[2] // ori_kv.shape[2]]
            softmax_lse_size = {
                ori_kv.size(DIM_1),
                q.size(DIM_0),
                q.size(DIM_2) / ori_kv.size(DIM_2)
            };
        }
    } else {
        // 不返回时tensor传空
        softmax_lse_size = {};
    }
    softmax_lse = at::empty(softmax_lse_size, q.options().dtype(torch::kFloat32));
    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> quant_sparse_flash_mla(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &q_descale, const c10::optional<at::Tensor> &ori_kv_descale,
    const c10::optional<at::Tensor> &cmp_kv_descale, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv, const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length, const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t qkv_quant_mode,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, int64_t topk_value_mode,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(qkv_quant_mode == 1, "qkv_quant_mode only support 1.")

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    const at::Tensor& ori_kv_val = *ori_kv;
    // convert str
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> quant_sparse_flash_mla_atten_out =
        op_api::construct_quant_sparse_flash_mla_atten_out_tensor(q, ori_kv_val, layout_q_str, return_softmax_lse);
    at::Tensor atten_out = std::get<0>(quant_sparse_flash_mla_atten_out);
    at::Tensor softmax_lse = std::get<1>(quant_sparse_flash_mla_atten_out);
    
    if (qkv_quant_mode == 1) { // HIFLOAT8 FULL QUANT
        at::Tensor null_tensor;
        auto ori_kv_value = ori_kv.has_value() ? ori_kv.value() : null_tensor;
        auto cmp_kv_value = cmp_kv.has_value() ? cmp_kv.value() : null_tensor;
        TensorWrapper q_wrapper = qsmla_make_wrapper(q, ACL_HIFLOAT8);
        TensorWrapper ori_kv_wrapper = qsmla_make_wrapper(ori_kv_value, ACL_HIFLOAT8);
        TensorWrapper cmp_kv_wrapper = qsmla_make_wrapper(cmp_kv_value, ACL_HIFLOAT8);
        ACLNN_CMD(aclnnQuantSparseFlashMla, q_wrapper, ori_kv_wrapper, cmp_kv_wrapper, q_descale, ori_kv_descale,
                  cmp_kv_descale, ori_sparse_indices, cmp_sparse_indices, ori_block_table, cmp_block_table,
                  cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv,
                  cmp_residual_kv, ori_topk_length, cmp_topk_length, sinks, metadata, qkv_quant_mode, softmax_scale,
                  cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr, layout_kv_ptr,
                  topk_value_mode, return_softmax_lse, atten_out, softmax_lse);
    }
    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_sparse_flash_mla_metadata", &quant_sparse_flash_mla_metadata,
          "quant_sparse_flash_mla_metadata");
    m.def("quant_sparse_flash_mla", &quant_sparse_flash_mla, "quant_sparse_flash_mla");
}
} // namespace op_api