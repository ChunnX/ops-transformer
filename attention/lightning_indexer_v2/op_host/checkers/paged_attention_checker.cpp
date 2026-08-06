/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log.h"
#include "paged_attention_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus PagedAttentionChecker::CheckParaExistence(const IndexerCheckerInfo &info)
{
    const bool paged = info.keyLayout == IndexerLayout::PA_BBND;
    OP_CHECK_IF(paged != info.blockTable.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "block_table",
                                                         "block_table must be provided iff key layout is PA_BBND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(paged && !info.sequsedK.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "seqused_k",
                                                         "seqused_k is required for PA_BBND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PagedAttentionChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (info.keyLayout != IndexerLayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(info.blockSize < 16 || info.blockSize > 1024 || info.blockSize % 16 != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "block_size",
                                                      std::to_string(info.blockSize).c_str(),
                                                      "The PA block size must be a multiple of 16 in [16, 1024]"),
                return ge::GRAPH_FAILED);
    if (CheckDtype(info.blockTable, "block_table", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.blockTable, "block_table", info) != ge::GRAPH_SUCCESS ||
        CheckRank(info.blockTable, "block_table", 2, info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.blockTable, "block_table", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PagedAttentionChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    const bool paged = info.keyLayout == IndexerLayout::PA_BBND;
    if (paged) {
        if (info.key.shape.empty() || info.key.shape[0] <= 0 || info.blockTable.shape.size() != 2 ||
            info.blockTable.shape[0] != info.batchSize ||
            (info.maxBlockNumPerBatch > 0 && info.blockTable.shape[1] != info.maxBlockNumPerBatch)) {
            const std::string actual = "key=" + ShapeToString(info.key.shape) +
                                       ", block_table=" + ShapeToString(info.blockTable.shape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(info.opName, "key and block_table", actual.c_str(),
                                                   "The block_table shape or number of key blocks is invalid");
            return ge::GRAPH_FAILED;
        }
        const int32_t *data = info.blockTable.tensor->GetData<int32_t>();
        if (data != nullptr) {
            const int64_t count = info.blockTable.shape[0] * info.blockTable.shape[1];
            for (int64_t i = 0; i < count; ++i) {
                const std::string paramName = "block_table[" + std::to_string(i) + "]";
                OP_CHECK_IF(data[i] < 0 || data[i] >= info.key.shape[0],
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                info.opName, paramName.c_str(), std::to_string(data[i]).c_str(),
                                "The block id must be within the key block range"),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    if (CheckStrides(info.key, info.keyStrides, paged, "key", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.quantized &&
        CheckStrides(info.keyDescale, info.keyDescaleStrides, paged, "key_dequant_scale", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PagedAttentionChecker::CheckStrides(const CheckerTensor &tensor, const std::vector<uint32_t> &strides,
                                                    bool allowDimZero, const char *name,
                                                    const IndexerCheckerInfo &info) const
{
    if (strides.empty() || tensor.shape.empty()) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string strideName = std::string(name) + " stride";
    OP_CHECK_IF(strides.size() != tensor.shape.size(),
                OP_LOGE_FOR_INVALID_SHAPEDIM(info.opName, strideName.c_str(),
                                             std::to_string(strides.size()).c_str(),
                                             std::to_string(tensor.shape.size()).c_str()),
                return ge::GRAPH_FAILED);
    uint64_t expected = 1;
    for (size_t offset = 0; offset < tensor.shape.size(); ++offset) {
        const size_t i = tensor.shape.size() - 1 - offset;
        if (!(allowDimZero && i == 0) && strides[i] != expected) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                info.opName, strideName.c_str(), "Only dimension 0 can be non-contiguous in PA mode");
            return ge::GRAPH_FAILED;
        }
        expected *= static_cast<uint64_t>(tensor.shape[i]);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
