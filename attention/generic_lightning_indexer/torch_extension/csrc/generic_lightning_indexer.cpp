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
using namespace at_npu::native;

const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;

static int64_t IndicesLastDim(int64_t topk, int64_t cmpRatio, bool expand, bool appendTail)
{
    if (!expand) {
        return topk;
    }
    int64_t expanded = topk * cmpRatio;
    return appendTail ? (expanded + cmpRatio - 1) : expanded;
}

std::tuple<at::Tensor, at::Tensor> ConstructGenericLightningIndexerOutput(
    const at::Tensor &query, int64_t topk, int64_t cmpRatio, const std::string &layoutQ, bool expand, bool appendTail,
    bool returnValue, int64_t selectMode, const at::Tensor &key, const std::string &layoutK)
{
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    if (selectMode == 1) {
        int64_t s2 = (layoutK == "TND") ? key.size(DIM_0) : key.size(DIM_1);
        int64_t nBlocks = (s2 + cmpRatio - 1) / cmpRatio;
        nBlocks = ((nBlocks + 15) / 16) * 16;
        at::Tensor emptyIdx = at::empty({0}, query.options().dtype(at::kInt));
        at::SmallVector<int64_t, SIZE> valShape;
        if (layoutQ == "BSND") {
            valShape = {query.size(DIM_0), query.size(DIM_1), query.size(2), nBlocks};
        } else {
            valShape = {query.size(DIM_0), query.size(DIM_1), nBlocks};
        }
        return {emptyIdx, at::empty(valShape, query.options().dtype(at::kFloat))};
    }

    int64_t last = IndicesLastDim(topk, cmpRatio, expand, appendTail);
    at::SmallVector<int64_t, SIZE> idxShape;
    at::SmallVector<int64_t, SIZE> valShape;
    if (layoutQ == "BSND") {
        idxShape = {query.size(DIM_0), query.size(DIM_1), last};
        valShape = {query.size(DIM_0), query.size(DIM_1), topk};
    } else {
        idxShape = {query.size(DIM_0), last};
        valShape = {query.size(DIM_0), topk};
    }
    at::Tensor sparseIndicesOut = at::empty(idxShape, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        sparseValuesOut = at::empty(valShape, query.options().dtype(at::kFloat));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return {sparseIndicesOut, sparseValuesOut};
}

std::tuple<at::Tensor, at::Tensor> GenericLightningIndexer(
    const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &weights,
    const c10::optional<at::Tensor> &actualSeqQ, const c10::optional<at::Tensor> &actualSeqK,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &qDescale, const c10::optional<at::Tensor> &kDescale,
    const c10::optional<at::Tensor> &outputIdxOffset, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &startLoc, int64_t topk, int64_t cmpRatio, c10::string_view layoutQ,
    c10::string_view layoutK, int64_t sparseMode, int64_t actType, bool hasWeightedSum, bool expandIndices,
    bool appendTail, int64_t quantMode, bool returnValue, int64_t selectMode, int64_t keyReduceMode, double scoreScale,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t maxSeqlenQ, int64_t initBlocks, int64_t localBlocks)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);
    auto output = ConstructGenericLightningIndexerOutput(query, topk, cmpRatio, queryLayoutStr, expandIndices,
                                                         appendTail, returnValue, selectMode, key, keyLayoutStr);
    at::Tensor sparseIndicesOut = std::get<0>(output);
    at::Tensor sparseValuesOut = std::get<1>(output);

    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());
    float scoreScaleF = static_cast<float>(scoreScale);

    int64_t keyStride0Attr = keyStride0;
    if (keyStride0Attr < 0 && !key.is_contiguous() &&
        (keyLayoutStr == "PA_BBND" || keyLayoutStr == "PA_BSND" || keyLayoutStr == "BBND" || keyLayoutStr == "BNBD")) {
        keyStride0Attr = key.stride(0);
    }
    int64_t kDescaleStride0Attr = kDescaleStride0;
    if (kDescale.has_value() && kDescaleStride0Attr < 0 && !kDescale.value().is_contiguous() &&
        (keyLayoutStr == "PA_BBND" || keyLayoutStr == "PA_BSND" || keyLayoutStr == "BBND" || keyLayoutStr == "BNBD")) {
        kDescaleStride0Attr = kDescale.value().stride(0);
    }

    ACLNN_CMD(aclnnGenericLightningIndexer, query, key, weights, actualSeqQ, actualSeqK, sequsedQ, sequsedK,
              cmpResidualK, blockTable, qDescale, kDescale, outputIdxOffset, metadata, startLoc, topk, cmpRatio,
              queryLayoutPtr, keyLayoutPtr, sparseMode, actType, hasWeightedSum, expandIndices, appendTail, quantMode,
              returnValue, selectMode, keyReduceMode, scoreScaleF, keyStride0Attr, kDescaleStride0Attr, maxSeqlenQ,
              initBlocks, localBlocks, sparseIndicesOut, sparseValuesOut);
    return {sparseIndicesOut, sparseValuesOut};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_lightning_indexer", &GenericLightningIndexer, "generic_lightning_indexer");
}
} // namespace op_api
