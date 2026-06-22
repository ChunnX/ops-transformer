// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <tuple>

#include <torch/extension.h>
#include "aclnn_common.h"

namespace {
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
constexpr int64_t MX_QUANT_BLOCK_SIZE = 32;
constexpr int64_t MXFP4_SCALE_BLOCK_SIZE = 64;
constexpr int64_t PAD_TO_EVEN_FACTOR = 2;

bool IsMxQuantMode(int64_t quantMode)
{
    return quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN ||
           quantMode == QUANT_MODE_MXFP4_E2M1;
}

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t AlignUp(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}
}  // namespace

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_token_permute(
    const at::Tensor &tokens,
    const at::Tensor &indices,
    c10::optional<int64_t> num_out_tokens,
    bool padded_mode,
    int64_t quant_mode)
{
    TORCH_CHECK(tokens.dim() == DIM_TWO,
        "tokens must be a 2D tensor, but got ", tokens.dim(), " dimensions.");
    TORCH_CHECK(indices.dim() == DIM_ONE || indices.dim() == DIM_TWO,
        "indices must be a 1D or 2D tensor, but got ", indices.dim(), " dimensions.");

    int64_t flatten_size = indices.numel();
    int64_t num_out_tokens_value = num_out_tokens.value_or(0);
    int64_t actual_num_out_tokens = num_out_tokens_value > 0
        ? std::min(num_out_tokens_value, flatten_size)
        : std::max<int64_t>(num_out_tokens_value + flatten_size, 0);

    bool is_ascend950 = c10_npu::GetSocVersion() == c10_npu::SocVersion::Ascend950;
    int64_t effective_quant_mode = is_ascend950 ? quant_mode : QUANT_MODE_NONE;
    TORCH_CHECK(effective_quant_mode == QUANT_MODE_NONE || IsMxQuantMode(effective_quant_mode),
        "quant_mode only supports -1, 2, 3 or 9 on Ascend 950.");
    TORCH_CHECK(!tokens.requires_grad() || effective_quant_mode == QUANT_MODE_NONE,
        "npu_moe_token_permute does not support autograd when quant_mode is 2, 3 or 9.");

    auto token_options = tokens.options();
    int64_t hidden_size = tokens.size(1);
    at::Tensor permuted_tokens;
    at::Tensor expanded_scale;
    at::Tensor sorted_indices;
    {
        auto local_device = c10::Device(tokens.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        if (effective_quant_mode == QUANT_MODE_MXFP8_E5M2) {
            permuted_tokens = at::empty(
                {actual_num_out_tokens, hidden_size}, token_options.dtype(at::kFloat8_e5m2));
            int64_t scale_cols = AlignUp(CeilDiv(hidden_size, MX_QUANT_BLOCK_SIZE), PAD_TO_EVEN_FACTOR);
            expanded_scale = at::empty(
                {actual_num_out_tokens, scale_cols}, token_options.dtype(at::kFloat8_e8m0fnu));
        } else if (effective_quant_mode == QUANT_MODE_MXFP8_E4M3FN) {
            permuted_tokens = at::empty(
                {actual_num_out_tokens, hidden_size}, token_options.dtype(at::kFloat8_e4m3fn));
            int64_t scale_cols = AlignUp(CeilDiv(hidden_size, MX_QUANT_BLOCK_SIZE), PAD_TO_EVEN_FACTOR);
            expanded_scale = at::empty(
                {actual_num_out_tokens, scale_cols}, token_options.dtype(at::kFloat8_e8m0fnu));
        } else if (effective_quant_mode == QUANT_MODE_MXFP4_E2M1) {
            TORCH_CHECK(hidden_size % 2 == 0, "The hidden size must be even when quant_mode is 9.");
            permuted_tokens = at::empty(
                {actual_num_out_tokens, hidden_size / 2}, token_options.dtype(at::kByte));
            expanded_scale = at::empty(
                {actual_num_out_tokens, CeilDiv(hidden_size, MXFP4_SCALE_BLOCK_SIZE), 2},
                token_options.dtype(at::kByte));
        } else {
            permuted_tokens = at::empty({actual_num_out_tokens, hidden_size}, token_options);
            expanded_scale = at::empty({0}, token_options.dtype(at::kFloat));
        }
        sorted_indices = at::empty({flatten_size}, indices.options().dtype(at::kInt));
    }

    TensorWrapper permuted_tokens_wrapper = {
        permuted_tokens, ConvertToAclDataType(permuted_tokens.scalar_type())};
    TensorWrapper expanded_scale_wrapper = {
        expanded_scale, ConvertToAclDataType(expanded_scale.scalar_type())};
    if (effective_quant_mode == QUANT_MODE_MXFP4_E2M1) {
        permuted_tokens_wrapper.dtype = aclDataType::ACL_FLOAT4_E2M1;
        expanded_scale_wrapper.dtype = aclDataType::ACL_FLOAT8_E8M0;
    }

    ACLNN_CMD(aclnnMoeTokenPermute,
        tokens,
        indices,
        actual_num_out_tokens,
        padded_mode,
        effective_quant_mode,
        permuted_tokens_wrapper,
        sorted_indices,
        expanded_scale_wrapper);

    return std::make_tuple(std::move(permuted_tokens), std::move(sorted_indices), std::move(expanded_scale));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_token_permute", &npu_moe_token_permute, "npu_moe_token_permute");
}
}  // namespace op_api
