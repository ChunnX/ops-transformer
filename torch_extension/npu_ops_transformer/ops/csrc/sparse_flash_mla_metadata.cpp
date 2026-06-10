/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include <string>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor npu_sparse_flash_mla_metadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    const c10::optional<at::Tensor> &cuSeqLensQ, const c10::optional<at::Tensor> &cuSeqLensOriKv,
    const c10::optional<at::Tensor> &cuSeqLensCmpKv, const c10::optional<at::Tensor> &seqUsedQ,
    const c10::optional<at::Tensor> &seqUsedOriKv, const c10::optional<at::Tensor> &cmpResidualKv,
    c10::optional<int64_t> batchSize,
    c10::optional<int64_t> maxSeqLenQ, c10::optional<int64_t> maxSeqLenKv, c10::optional<int64_t> oriTopK,
    c10::optional<int64_t> cmpTopK, c10::optional<int64_t> cmpRatio, c10::optional<int64_t> oriMaskMode,
    c10::optional<int64_t> cmpMaskMode, c10::optional<int64_t> oriWinLeft, c10::optional<int64_t> oriWinRight,
    c10::optional<c10::string_view> layoutQ, c10::optional<c10::string_view> layoutKv,
    c10::optional<bool> hasOriKv, c10::optional<bool> hasCmpKv)
{
    static const at::Tensor emptyTensor;
    const at::Tensor &cuSeqLensQValue = cuSeqLensQ.has_value() ? cuSeqLensQ.value() : emptyTensor;
    const at::Tensor &cuSeqLensOriKvValue = cuSeqLensOriKv.has_value() ? cuSeqLensOriKv.value() : emptyTensor;
    const at::Tensor &cuSeqLensCmpKvValue = cuSeqLensCmpKv.has_value() ? cuSeqLensCmpKv.value() : emptyTensor;
    const at::Tensor &seqUsedQValue = seqUsedQ.has_value() ? seqUsedQ.value() : emptyTensor;
    const at::Tensor &seqUsedOriKvValue = seqUsedOriKv.has_value() ? seqUsedOriKv.value() : emptyTensor;
    const at::Tensor &cmpResidualKvValue = cmpResidualKv.has_value() ? cmpResidualKv.value() : emptyTensor;

    const at::Tensor *metadataOptionsTensor = nullptr;
    const at::Tensor *candidates[] = {&cuSeqLensQValue, &cuSeqLensOriKvValue, &cuSeqLensCmpKvValue, &seqUsedQValue,
                                      &seqUsedOriKvValue, &cmpResidualKvValue};
    for (const at::Tensor *candidate : candidates) {
        if (candidate->defined()) {
            metadataOptionsTensor = candidate;
            break;
        }
    }
    TORCH_CHECK(metadataOptionsTensor != nullptr, "At least one length tensor must be provided to infer metadata device.");
    at::Tensor metadata = at::empty({1024}, metadataOptionsTensor->options().dtype(at::kInt));

    std::string layoutQStr = std::string(layoutQ.value_or("BSND"));
    std::string layoutKvStr = std::string(layoutKv.value_or("BSND"));
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    const int64_t batchSizeValue = batchSize.value_or(0);
    const int64_t maxSeqLenQValue = maxSeqLenQ.value_or(0);
    const int64_t maxSeqLenKvValue = maxSeqLenKv.value_or(0);
    const int64_t oriTopKValue = oriTopK.value_or(0);
    const int64_t cmpTopKValue = cmpTopK.value_or(0);
    const int64_t cmpRatioValue = cmpRatio.value_or(1);
    const int64_t oriMaskModeValue = oriMaskMode.value_or(0);
    const int64_t cmpMaskModeValue = cmpMaskMode.value_or(0);
    const int64_t oriWinLeftValue = oriWinLeft.value_or(-1);
    const int64_t oriWinRightValue = oriWinRight.value_or(-1);
    const bool hasOriKvValue = hasOriKv.value_or(true);
    const bool hasCmpKvValue = hasCmpKv.value_or(true);

    ACLNN_CMD(aclnnSparseFlashMlaMetadata, cuSeqLensQValue, cuSeqLensOriKvValue, cuSeqLensCmpKvValue, seqUsedQValue,
              seqUsedOriKvValue, cmpResidualKvValue, numHeadsQ, numHeadsKv, headDim, batchSizeValue,
              maxSeqLenQValue, maxSeqLenKvValue, oriTopKValue, cmpTopKValue, cmpRatioValue, oriMaskModeValue,
              cmpMaskModeValue, oriWinLeftValue, oriWinRightValue, layoutQPtr, layoutKvPtr, hasOriKvValue,
              hasCmpKvValue, metadata);
    return metadata;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla_metadata", &npu_sparse_flash_mla_metadata, "npu_sparse_flash_mla_metadata");
}
} // namespace op_api
