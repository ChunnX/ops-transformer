// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>

#include <c10/core/DeviceGuard.h>

#include <cmath>
#include <tuple>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t MATRIX_DIM = 2L;
constexpr int64_t VECTOR_DIM = 1L;
constexpr int64_t TOKEN_DIM = 0L;
constexpr int64_t HIDDEN_DIM = 1L;

void CheckTensorDevice(const at::Tensor &tensor, const at::Device &expectedDevice, const char *name)
{
    TORCH_CHECK(tensor.device() == expectedDevice, name, " must be on device ", expectedDevice,
                ", but got ", tensor.device(), ".");
}

void CheckTensorContiguous(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void CheckInputs(const at::Tensor &partialBlock, const at::Tensor &partialDelta,
                 const at::Tensor &effectiveQuery, const at::Tensor &interMax,
                 const at::Tensor &interExpSum, const at::Tensor &interNumerator, float epsilon)
{
    TORCH_CHECK(partialBlock.scalar_type() == at::kFloat,
                "partial_block must be float32, but got ", partialBlock.scalar_type(), ".");
    TORCH_CHECK(partialDelta.scalar_type() == at::kHalf || partialDelta.scalar_type() == at::kBFloat16,
                "partial_delta must be float16 or bfloat16, but got ", partialDelta.scalar_type(), ".");
    TORCH_CHECK(effectiveQuery.scalar_type() == at::kFloat,
                "effective_query must be float32, but got ", effectiveQuery.scalar_type(), ".");
    TORCH_CHECK(interMax.scalar_type() == at::kFloat,
                "inter_max must be float32, but got ", interMax.scalar_type(), ".");
    TORCH_CHECK(interExpSum.scalar_type() == at::kFloat,
                "inter_exp_sum must be float32, but got ", interExpSum.scalar_type(), ".");
    TORCH_CHECK(interNumerator.scalar_type() == at::kFloat,
                "inter_numerator must be float32, but got ", interNumerator.scalar_type(), ".");
    CheckTensorContiguous(partialBlock, "partial_block");
    CheckTensorContiguous(partialDelta, "partial_delta");
    CheckTensorContiguous(effectiveQuery, "effective_query");
    CheckTensorContiguous(interMax, "inter_max");
    CheckTensorContiguous(interExpSum, "inter_exp_sum");
    CheckTensorContiguous(interNumerator, "inter_numerator");

    TORCH_CHECK(partialBlock.dim() == MATRIX_DIM,
                "partial_block must be 2D, but got ", partialBlock.dim(), "D.");
    TORCH_CHECK(partialDelta.dim() == MATRIX_DIM,
                "partial_delta must be 2D, but got ", partialDelta.dim(), "D.");
    TORCH_CHECK(interNumerator.dim() == MATRIX_DIM,
                "inter_numerator must be 2D, but got ", interNumerator.dim(), "D.");
    TORCH_CHECK(effectiveQuery.dim() == VECTOR_DIM,
                "effective_query must be 1D, but got ", effectiveQuery.dim(), "D.");
    TORCH_CHECK(interMax.dim() == VECTOR_DIM,
                "inter_max must be 1D, but got ", interMax.dim(), "D.");
    TORCH_CHECK(interExpSum.dim() == VECTOR_DIM,
                "inter_exp_sum must be 1D, but got ", interExpSum.dim(), "D.");

    TORCH_CHECK(partialDelta.sizes() == partialBlock.sizes(),
                "partial_delta shape must match partial_block, but got ", partialDelta.sizes(),
                " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(interNumerator.sizes() == partialBlock.sizes(),
                "inter_numerator shape must match partial_block, but got ", interNumerator.sizes(),
                " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(effectiveQuery.size(TOKEN_DIM) == partialBlock.size(HIDDEN_DIM),
                "effective_query length must match the hidden dimension of partial_block, but got ",
                effectiveQuery.sizes(), " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(interMax.size(TOKEN_DIM) == partialBlock.size(TOKEN_DIM),
                "inter_max length must match the token dimension of partial_block, but got ", interMax.sizes(),
                " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(interExpSum.size(TOKEN_DIM) == partialBlock.size(TOKEN_DIM),
                "inter_exp_sum length must match the token dimension of partial_block, but got ",
                interExpSum.sizes(), " and ", partialBlock.sizes(), ".");

    const auto expectedDevice = partialBlock.device();
    CheckTensorDevice(partialDelta, expectedDevice, "partial_delta");
    CheckTensorDevice(effectiveQuery, expectedDevice, "effective_query");
    CheckTensorDevice(interMax, expectedDevice, "inter_max");
    CheckTensorDevice(interExpSum, expectedDevice, "inter_exp_sum");
    CheckTensorDevice(interNumerator, expectedDevice, "inter_numerator");

    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0F,
                "epsilon must be finite and greater than 0, but got ", epsilon, ".");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> block_attn_res_update(
    const at::Tensor &partialBlock, const at::Tensor &partialDelta, const at::Tensor &effectiveQuery,
    const at::Tensor &interMax, const at::Tensor &interExpSum, const at::Tensor &interNumerator, float epsilon)
{
    CheckInputs(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator, epsilon);

    at::Tensor h{nullptr};
    at::Tensor partialBlocks{nullptr};
    {
        const auto localDevice = c10::Device(partialBlock.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        h = at::empty(partialDelta.sizes(), partialDelta.options());
        partialBlocks = at::empty(partialBlock.sizes(), partialBlock.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnBlockAttnResUpdate, partialBlock, partialDelta, effectiveQuery, interMax, interExpSum,
              interNumerator, epsilon, h, partialBlocks);

    return std::make_tuple(h, partialBlocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attn_res_update", &block_attn_res_update, "BlockAttnResUpdate torch wrapper");
}
} // namespace op_api
