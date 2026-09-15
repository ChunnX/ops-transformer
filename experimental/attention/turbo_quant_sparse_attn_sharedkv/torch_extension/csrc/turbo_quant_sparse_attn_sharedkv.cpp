/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
using OptionalTensor = c10::optional<at::Tensor>;
using TensorPair = std::tuple<at::Tensor, at::Tensor>;

namespace {
TensorPair ConstructOutputs(const at::Tensor &query, bool returnSoftmaxLse)
{
    TORCH_CHECK(query.dim() == 3, "query must use TND layout");
    for (int64_t dim = 1; dim < query.dim(); ++dim) {
        TORCH_CHECK(query.size(dim) > 0, "query's N and D dimensions must be greater than 0");
    }
    at::Tensor attnOut = at::empty(query.sizes(), query.options());
    at::Tensor softmaxLse;
    if (returnSoftmaxLse) {
        std::vector<int64_t> lseShape(query.sizes().begin(), query.sizes().end());
        lseShape.back() = 1;
        softmaxLse = at::empty(lseShape, query.options().dtype(at::kFloat));
    } else {
        softmaxLse = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return {attnOut, softmaxLse};
}
} // namespace

TensorPair TurboQuantSparseAttnSharedkv(
    const at::Tensor &query, const OptionalTensor &originalKv, const OptionalTensor &compressedKv,
    const OptionalTensor &originalSparseIndices, const OptionalTensor &compressedSparseIndices,
    const OptionalTensor &originalBlockTable, const OptionalTensor &compressedBlockTable,
    const OptionalTensor &querySequenceOffsets, const OptionalTensor &originalKvSequenceOffsets,
    const OptionalTensor &compressedKvSequenceOffsets, const OptionalTensor &usedQueryLengths,
    const OptionalTensor &usedOriginalKvLengths, const OptionalTensor &usedCompressedKvLengths,
    const OptionalTensor &compressedKvResidual, const OptionalTensor &originalTopkLengths,
    const OptionalTensor &compressedTopkLengths, const OptionalTensor &attentionSinks,
    const OptionalTensor &scheduleMetadata, double scale, int64_t compressionRatio, int64_t originalMaskMode,
    int64_t compressedMaskMode, int64_t windowLeft, int64_t windowRight, c10::string_view queryLayout,
    c10::string_view kvLayout, int64_t topkMode, bool returnSoftmaxLse, int64_t originalKvStride,
    int64_t compressedKvStride, int64_t quantMode)
{
    std::string queryLayoutString(queryLayout);
    std::string kvLayoutString(kvLayout);
    auto outputs = ConstructOutputs(query, returnSoftmaxLse);
    at::Tensor attnOut = std::get<0>(outputs);
    at::Tensor softmaxLse = std::get<1>(outputs);
    if (originalKv.has_value()) {
        originalKvStride = originalKv.value().stride(0);
    }
    if (compressedKv.has_value()) {
        compressedKvStride = compressedKv.value().stride(0);
    }
    char *queryLayoutData = const_cast<char *>(queryLayoutString.c_str());
    char *kvLayoutData = const_cast<char *>(kvLayoutString.c_str());

    ACLNN_CMD(aclnnTurboQuantSparseAttnSharedkv, query, originalKv, compressedKv, originalSparseIndices,
              compressedSparseIndices, originalBlockTable, compressedBlockTable, querySequenceOffsets,
              originalKvSequenceOffsets, compressedKvSequenceOffsets, usedQueryLengths, usedOriginalKvLengths,
              usedCompressedKvLengths, compressedKvResidual, originalTopkLengths, compressedTopkLengths, attentionSinks,
              scheduleMetadata, scale, compressionRatio, originalMaskMode, compressedMaskMode, windowLeft, windowRight,
              queryLayoutData, kvLayoutData, topkMode, returnSoftmaxLse, originalKvStride, compressedKvStride,
              quantMode, attnOut, softmaxLse);
    return {attnOut, softmaxLse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("turbo_quant_sparse_attn_sharedkv", &TurboQuantSparseAttnSharedkv, "turbo_quant_sparse_attn_sharedkv");
}
} // namespace op_api
