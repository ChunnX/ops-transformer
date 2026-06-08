/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
at::Tensor npu_sparse_flash_mla_metadata(
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t batchSize,
    int64_t maxSeqlenQ,
    int64_t maxSeqlenKv,
    int64_t oriTopK,
    int64_t cmpTopK,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    c10::string_view layoutQ,
    c10::string_view layoutKv,
    bool hasOriKv,
    bool hasCmpKv,
    const at::Tensor &metadata)
{
    std::string layoutQStr(layoutQ);
    std::string layoutKvStr(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaMetadata, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv,
              numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv, oriTopK, cmpTopK, cmpRatio,
              oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr, hasOriKv, hasCmpKv,
              metadata);
    return metadata;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_sparse_flash_mla_metadata", &npu_sparse_flash_mla_metadata, "npu_sparse_flash_mla_metadata");
}
}  // namespace op_api
