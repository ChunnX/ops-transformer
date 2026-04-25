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
 * \file kv_quant_sparse_flash_mla.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

at::Tensor construct_kv_quant_sparse_flash_mla_output_tensor(
    const at::Tensor& query, std::string layout_query_str,
    std::string layout_kv_str, const uint64_t &rope_head_dim)
{
    TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
                "The layout of query only support BSND and TND, but got ", layout_query_str,
                OPS_ERROR(ErrCode::PARAM));
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "BSND") {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3) - rope_head_dim};
    } else {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) - rope_head_dim};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(query.dtype()));

    return output;
}

at::Tensor npu_kv_quant_sparse_flash_mla(
    const at::Tensor &q, int64_t kv_quant_mode,
    const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &ori_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, const c10::optional<at::Tensor> &metadata,
    int64_t tile_size, int64_t rope_head_dim, double softmax_scale, int64_t ori_mask_mode,
    c10::string_view layout_query, c10::string_view layout_kv,  bool return_softmax_lse,
    c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype)
    // const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    // const at::Tensor &sparse_indices, double scale_value,
    // int64_t key_quant_mode, int64_t value_quant_mode,
    // const c10::optional<at::Tensor> &key_dequant_scale,
    // const c10::optional<at::Tensor> &value_dequant_scale,
    // const c10::optional<at::Tensor> &block_table,
    // const c10::optional<at::Tensor> &actual_seq_lengths_query,
    // const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    // const c10::optional<at::Tensor> &metadata,
    // int64_t sparse_block_size, c10::string_view layout_query, c10::string_view layout_kv,
    // int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t attention_mode, int64_t quant_scale_repo_mode,
    // int64_t tile_size, int64_t rope_head_dim,
    // c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype)
{
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.")

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor output = op_api::construct_quant_sparse_infer_output_tensor(
        q, layout_query_str, layout_kv_str, rope_head_dim);
    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::SmallVector<int64_t, SIZE> softmax_max_size;
    at::SmallVector<int64_t, SIZE> softmax_sum_size;
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    if (q.dim() == DIM_3) {
        softmax_max_size = {ori_kv.size(1), q.size(0), q.size(1) / ori_kv.size(1)};
        softmax_sum_size = {ori_kv.size(1), q.size(0), q.size(1) / ori_kv.size(1)};
    } else {
        softmax_max_size = {q.size(0), ori_kv.size(2), q.size(1), q.size(2) / ori_kv.size(2)};
        softmax_sum_size = {q.size(0), ori_kv.size(2), q.size(1), q.size(2) / ori_kv.size(2)};
    }
    softmax_max = at::empty(softmax_max_size, query.options().dtype(at::kFloat));
    softmax_sum = at::empty(softmax_sum_size, query.options().dtype(at::kFloat));
    // bool is_hifloat8_kv = key_dtype.has_value() && c10_npu::GetAclDataType(key_dtype.value()) == aclDataType::ACL_HIFLOAT8;
    // if (is_hifloat8_kv) {
    //     TensorWrapper key_wrapper = make_wrapper(key, key_dtype);
    //     TensorWrapper value_wrapper = make_wrapper(value, value_dtype);
    //     EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttention, query, key_wrapper,
    //         value_wrapper, sparse_indices, key_dequant_scale, value_dequant_scale, block_table, actual_seq_lengths_query,
    //         actual_seq_lengths_kv, metadata, scale_value, key_quant_mode, value_quant_mode, sparse_block_size, layout_query_ptr,
    //         layout_kv_ptr, sparse_mode, pre_tokens, next_tokens, attention_mode, quant_scale_repo_mode, tile_size,
    //         rope_head_dim, output);
    // } else {kvQuantSparseFlashMla
    ACLNN_CMD(kvQuantSparseFlashMla, q,
              ori_kv, ori_sparse_indices, ori_block_table, cu_seqlens_q, cu_seqlens_ori_kv, seqused_q, seqused_kv,
              metadata, kv_quant_mode, tile_size, softmax_scale, ori_mask_mode, layout_query, layout_kv,
              return_softmax_lse, atten_out, softmax_lse);
    // }
    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kvQuantSparseFlashMla", &kvQuantSparseFlashMla, "kvQuantSparseFlashMla");
}
} // namespace op_api