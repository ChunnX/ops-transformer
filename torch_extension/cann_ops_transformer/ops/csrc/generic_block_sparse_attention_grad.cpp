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
 * \file generic_block_sparse_attention_grad.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t kMaxHeadDim = 128;

using OptionalTensor = c10::optional<at::Tensor>;
using OptionalIntList = c10::optional<std::vector<int64_t>>;

// Validate input parameters.
static void check_params(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                         const OptionalTensor &cu_seq_lengths, const OptionalTensor &cu_seq_lengths_kv,
                         const OptionalTensor &actual_seq_lengths, const OptionalTensor &actual_seq_lengths_kv,
                         c10::string_view q_input_layout, c10::string_view kv_input_layout)
{
    // Q/K/V must use the same dtype.
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
                "query, key, value must have the same dtype, got query=", query.scalar_type(), ", key=", key.scalar_type(),
                ", value=", value.scalar_type());

    // The kernel supports head_dim up to 128.
    int64_t head_dim = query.size(-1);
    TORCH_CHECK(head_dim <= kMaxHeadDim, "head_dim must be <= ", kMaxHeadDim, ", but got ", head_dim);

    // TND inputs require either a prefix list or per-batch used-length list.
    if (q_input_layout == "TND") {
        TORCH_CHECK((cu_seq_lengths.has_value() && cu_seq_lengths->defined()) ||
                        (actual_seq_lengths.has_value() && actual_seq_lengths->defined()),
                    "cu_seq_lengths or actual_seq_lengths must be specified for TND Q");
    }
    if (kv_input_layout == "TND") {
        TORCH_CHECK((cu_seq_lengths_kv.has_value() && cu_seq_lengths_kv->defined()) ||
                        (actual_seq_lengths_kv.has_value() && actual_seq_lengths_kv->defined()),
                    "cu_seq_lengths_kv or actual_seq_lengths_kv must be specified for TND KV");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_attention_score_backward(
    const at::Tensor &d_out, const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &attention_out, const at::Tensor &softmax_lse, const at::Tensor &rsvd_block_idx,
    const at::Tensor &rsvd_block_count, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &atten_mask, const OptionalIntList block_shape,
    const OptionalTensor cu_seq_lengths, const OptionalTensor cu_seq_lengths_kv,
    const OptionalTensor actual_seq_lengths, const OptionalTensor actual_seq_lengths_kv,
    int64_t is_packed_gqa, c10::string_view q_input_layout, c10::string_view kv_input_layout,
    double scale_value, int64_t mask_type, int64_t softmax_precision, int64_t win_left, int64_t win_right)
{
    check_params(query, key, value, cu_seq_lengths, cu_seq_lengths_kv, actual_seq_lengths, actual_seq_lengths_kv,
                 q_input_layout, kv_input_layout);

    // at::Tensor d_query = npu_preparation::apply_tensor_without_format(query);
    // at::Tensor d_key = npu_preparation::apply_tensor_without_format(key);
    // at::Tensor d_value = npu_preparation::apply_tensor_without_format(value);

    at::Tensor d_query{nullptr};
    at::Tensor d_key{nullptr};
    at::Tensor d_value{nullptr};
    {
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        d_query = at::empty(query.sizes(), query.options());
        d_key = at::empty(key.sizes(), key.options());
        d_value = at::empty(value.sizes(), value.options());
    }
    // Use the default block shape [128, 128] when block_shape is not specified.
    static const int64_t kDefaultBlockShape[2] = {128, 128};
    const at::IntArrayRef block_shape_value =
        (block_shape.has_value() && block_shape->size() >= 2) ? at::IntArrayRef(*block_shape) :
                                                                at::IntArrayRef(kDefaultBlockShape, 2);
    const at::Tensor metadata_value = metadata.has_value() ? *metadata : at::Tensor{nullptr};
    const at::Tensor atten_mask_value = atten_mask.has_value() ? *atten_mask : at::Tensor{nullptr};
    const at::Tensor cu_seq_lengths_value = cu_seq_lengths.has_value() ? *cu_seq_lengths : at::Tensor{nullptr};
    const at::Tensor cu_seq_lengths_kv_value = cu_seq_lengths_kv.has_value() ? *cu_seq_lengths_kv : at::Tensor{nullptr};
    const at::Tensor actual_seq_lengths_value = actual_seq_lengths.has_value() ? *actual_seq_lengths : at::Tensor{nullptr};
    const at::Tensor actual_seq_lengths_kv_value = actual_seq_lengths_kv.has_value() ? *actual_seq_lengths_kv : at::Tensor{nullptr};

    // Pass layout strings through to aclnn. aclnn owns the final validation of supported layouts.
    char *q_input_layout_ptr = const_cast<char *>(q_input_layout.data());
    char *kv_input_layout_ptr = const_cast<char *>(kv_input_layout.data());

    // Call aclnn API.
    ACLNN_CMD(aclnnGenericBlockSparseAttentionGrad, query, key, value, d_out, attention_out, softmax_lse,
              rsvd_block_idx, rsvd_block_count, metadata_value, atten_mask_value, cu_seq_lengths_value,
              cu_seq_lengths_kv_value, actual_seq_lengths_value, actual_seq_lengths_kv_value,
              block_shape_value, is_packed_gqa, q_input_layout_ptr, kv_input_layout_ptr, scale_value,
              mask_type, softmax_precision, win_left, win_right, d_query, d_key, d_value);

    // Return gradients.
    return std::make_tuple(d_query, d_key, d_value);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_attention_score_backward", &sparse_attention_score_backward, "sparse_attention_score_backward");
}
} // namespace op_api
