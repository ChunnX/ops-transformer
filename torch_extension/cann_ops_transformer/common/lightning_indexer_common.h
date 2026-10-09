/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_COMMON_H
#define LIGHTNING_INDEXER_COMMON_H

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

constexpr int64_t LIGHTNING_INDEXER_MAX_DIM = 8;
constexpr int64_t LIGHTNING_INDEXER_DIM_0 = 0;
constexpr int64_t LIGHTNING_INDEXER_DIM_1 = 1;
constexpr int64_t LIGHTNING_INDEXER_DIM_2 = 2;

inline std::tuple<at::Tensor, at::Tensor> ConstructLightningIndexerOutputTensor(const at::Tensor &query,
                                                                                const at::Tensor &key, int64_t topk,
                                                                                const std::string &queryLayout,
                                                                                const std::string &keyLayout,
                                                                                bool returnValue)
{
    at::SmallVector<int64_t, LIGHTNING_INDEXER_MAX_DIM> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater than 0, but shape[", i,
                    "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater than 0, but shape[", i, "] is ",
                    key.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);

    int64_t keyHeadNum = keyLayout == "TND" ? key.size(LIGHTNING_INDEXER_DIM_1) : key.size(LIGHTNING_INDEXER_DIM_2);
    if (queryLayout == "BSND") {
        outputSize = {query.size(LIGHTNING_INDEXER_DIM_0), query.size(LIGHTNING_INDEXER_DIM_1), keyHeadNum, topk};
    } else {
        outputSize = {query.size(LIGHTNING_INDEXER_DIM_0), keyHeadNum, topk};
    }

    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut = returnValue ? at::empty(outputSize, query.options().dtype(at::kFloat)) :
                                               at::empty({0}, query.options().dtype(at::kFloat));
    return std::make_tuple(sparseIndicesOut, sparseValuesOut);
}

template <bool BatchInvariant>
std::tuple<at::Tensor, at::Tensor> LightningIndexer(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, int64_t topk,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &outputIdxOffset, const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t returnValue)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")

    std::string queryLayout = std::string(layoutQ);
    std::string keyLayout = std::string(layoutK);
    auto [sparseIndicesOut, sparseValuesOut] =
        ConstructLightningIndexerOutputTensor(q, k, topk, queryLayout, keyLayout, returnValue);
    char *queryLayoutPtr = const_cast<char *>(queryLayout.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayout.c_str());

    if constexpr (BatchInvariant) {
        ACLNN_CMD(aclnnLightningIndexerV2BatchInvariant, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK,
                  cmpResidualK, blockTable, outputIdxOffset, metadata, topk, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr,
                  maskMode, cmpRatio, returnValue, sparseIndicesOut, sparseValuesOut);
    } else {
        ACLNN_CMD(aclnnLightningIndexerV2, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK,
                  blockTable, outputIdxOffset, metadata, topk, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode,
                  cmpRatio, returnValue, sparseIndicesOut, sparseValuesOut);
    }
    return std::make_tuple(sparseIndicesOut, sparseValuesOut);
}
} // namespace op_api

#endif // LIGHTNING_INDEXER_COMMON_H
