/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_0 = 0;

std::tuple<at::Tensor, at::Tensor> MakeSparseFlashMlaOutputs(
    const at::Tensor &q,
    bool returnSoftmaxLse)
{
    TORCH_CHECK(q.defined() && q.numel() > 0, "Tensor q must be defined and non-empty.");
    auto attnOut = at::empty(q.sizes(), q.options().dtype(q.dtype()));

    at::SmallVector<int64_t, 8> lseShape;
    if (returnSoftmaxLse) {
        TORCH_CHECK(q.dim() > 0, "Tensor q must have at least one dimension.");
        for (const auto dim : q.sizes()) {
            lseShape.push_back(dim);
        }
        lseShape[q.dim() - 1] = 1;
    } else {
        lseShape = {DIM_0};
    }
    auto softmaxLse = at::empty(lseShape, q.options().dtype(at::kFloat));
    return std::make_tuple(attnOut, softmaxLse);
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_mla(
    const at::Tensor &q,
    const c10::optional<at::Tensor> &oriKv,
    const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices,
    const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable,
    const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv,
    const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv,
    const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength,
    const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata,
    c10::optional<double> softmaxScale,
    c10::optional<int64_t> cmpRatio,
    c10::optional<int64_t> oriMaskMode,
    c10::optional<int64_t> cmpMaskMode,
    c10::optional<int64_t> oriWinLeft,
    c10::optional<int64_t> oriWinRight,
    c10::string_view layoutQ,
    c10::string_view layoutKv,
    c10::optional<int64_t> topkValueMode,
    bool returnSoftmaxLse)
{
    auto outputs = MakeSparseFlashMlaOutputs(q, returnSoftmaxLse);
    at::Tensor attnOut = std::get<0>(outputs);
    at::Tensor softmaxLse = std::get<1>(outputs);

    const int64_t headDim = q.size(q.dim() - 1);
    const double softmaxScaleValue = softmaxScale.value_or(1.0 / std::sqrt(static_cast<double>(headDim)));
    const int64_t cmpRatioValue = cmpRatio.value_or(1);
    const int64_t oriMaskModeValue = oriMaskMode.value_or(4);
    const int64_t cmpMaskModeValue = cmpMaskMode.value_or(3);
    const int64_t oriWinLeftValue = oriWinLeft.value_or(127);
    const int64_t oriWinRightValue = oriWinRight.value_or(0);
    const int64_t topkValueModeValue = topkValueMode.value_or(1);

    const at::Tensor &oriKvValue = oriKv.value_or(at::Tensor());
    const at::Tensor &cmpKvValue = cmpKv.value_or(at::Tensor());
    const at::Tensor &oriSparseIndicesValue = oriSparseIndices.value_or(at::Tensor());
    const at::Tensor &cmpSparseIndicesValue = cmpSparseIndices.value_or(at::Tensor());
    const at::Tensor &oriBlockTableValue = oriBlockTable.value_or(at::Tensor());
    const at::Tensor &cmpBlockTableValue = cmpBlockTable.value_or(at::Tensor());
    const at::Tensor &cuSeqlensQValue = cuSeqlensQ.value_or(at::Tensor());
    const at::Tensor &cuSeqlensOriKvValue = cuSeqlensOriKv.value_or(at::Tensor());
    const at::Tensor &cuSeqlensCmpKvValue = cuSeqlensCmpKv.value_or(at::Tensor());
    const at::Tensor &sequsedQValue = sequsedQ.value_or(at::Tensor());
    const at::Tensor &sequsedOriKvValue = sequsedOriKv.value_or(at::Tensor());
    const at::Tensor &sequsedCmpKvValue = sequsedCmpKv.value_or(at::Tensor());
    const at::Tensor &cmpResidualKvValue = cmpResidualKv.value_or(at::Tensor());
    const at::Tensor &oriTopkLengthValue = oriTopkLength.value_or(at::Tensor());
    const at::Tensor &cmpTopkLengthValue = cmpTopkLength.value_or(at::Tensor());
    const at::Tensor &sinksValue = sinks.value_or(at::Tensor());
    const at::Tensor &metadataValue = metadata.value_or(at::Tensor());

    std::string layoutQStr(layoutQ);
    std::string layoutKvStr(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    ACLNN_CMD(aclnnSparseFlashMla, q, oriKvValue, cmpKvValue, oriSparseIndicesValue, cmpSparseIndicesValue,
              oriBlockTableValue, cmpBlockTableValue, cuSeqlensQValue, cuSeqlensOriKvValue, cuSeqlensCmpKvValue,
              sequsedQValue, sequsedOriKvValue, sequsedCmpKvValue, cmpResidualKvValue, oriTopkLengthValue,
              cmpTopkLengthValue, sinksValue, metadataValue, softmaxScaleValue, cmpRatioValue, oriMaskModeValue,
              cmpMaskModeValue, oriWinLeftValue, oriWinRightValue, layoutQPtr, layoutKvPtr, topkValueModeValue,
              returnSoftmaxLse, attnOut, softmaxLse);
    return std::make_tuple(attnOut, softmaxLse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla", &npu_sparse_flash_mla, "npu_sparse_flash_mla");
}
}  // namespace op_api
