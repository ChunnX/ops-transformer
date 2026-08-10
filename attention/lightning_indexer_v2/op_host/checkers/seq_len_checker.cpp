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
#include "seq_len_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus SeqLenChecker::CheckParaExistence(const IndexerCheckerInfo &info)
{
    const bool qTnd = info.queryLayout == IndexerLayout::TND;
    const bool kTnd = info.keyLayout == IndexerLayout::TND;
    OP_CHECK_IF(qTnd != info.cuSeqlensQ.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_q",
                                                         "cu_seqlens_q must be provided iff query layout is TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kTnd != info.cuSeqlensK.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "cu_seqlens_k",
                                                         "cu_seqlens_k must be provided iff key layout is TND"),
                return ge::GRAPH_FAILED);
    if (!info.isArch35 && qTnd && info.sequsedQ.present) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "seqused_q", "seqused_q is unsupported when query layout is TND on Ascend A2/A3");
        return ge::GRAPH_FAILED;
    }
    if (!info.isArch35 && kTnd && info.sequsedK.present) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            info.opName, "seqused_k", "seqused_k is unsupported when key layout is TND on Ascend A2/A3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SeqLenChecker::CheckSeqTensor(const CheckerTensor &tensor, const char *name, int64_t length,
                                              const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckOptionalTensor(tensor, name, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(tensor, name, {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(tensor, name, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckShape(tensor, name, {length}, info);
}
ge::graphStatus SeqLenChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (CheckSeqTensor(info.cuSeqlensQ, "cu_seqlens_q", info.batchSize + 1, info) != ge::GRAPH_SUCCESS ||
        CheckSeqTensor(info.cuSeqlensK, "cu_seqlens_k", info.batchSize + 1, info) != ge::GRAPH_SUCCESS ||
        CheckSeqTensor(info.sequsedQ, "seqused_q", info.batchSize, info) != ge::GRAPH_SUCCESS ||
        CheckSeqTensor(info.sequsedK, "seqused_k", info.batchSize, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SeqLenChecker::CheckCuValues(const CheckerTensor &tensor, int64_t total, const char *name,
                                             const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t *data = tensor.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (data[0] != 0 || data[info.batchSize] != total) {
        const std::string actual = "first=" + std::to_string(data[0]) +
                                   ", last=" + std::to_string(data[info.batchSize]);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, name, actual.c_str(),
                                               "The sequence must start at 0 and end at the flattened length");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < info.batchSize; ++i) {
        const std::string paramName = std::string(name) + "[" + std::to_string(i) + "] and [" +
                                      std::to_string(i + 1) + "]";
        const std::string actual = std::to_string(data[i]) + " and " + std::to_string(data[i + 1]);
        OP_CHECK_IF(data[i] < 0 || data[i] > data[i + 1],
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, paramName.c_str(), actual.c_str(),
                                                           "The values must be nonnegative and nondecreasing"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SeqLenChecker::CheckUsedValues(const CheckerTensor &used, const CheckerTensor &cu,
                                               int64_t fallbackLimit, const char *name,
                                               const IndexerCheckerInfo &info) const
{
    if (!used.present) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t *data = used.tensor->GetData<int32_t>();
    const int32_t *cuData = cu.present ? cu.tensor->GetData<int32_t>() : nullptr;
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t i = 0; i < info.batchSize; ++i) {
        const int64_t limit = cuData == nullptr ? fallbackLimit : static_cast<int64_t>(cuData[i + 1] - cuData[i]);
        const std::string paramName = std::string(name) + "[" + std::to_string(i) + "]";
        OP_CHECK_IF(data[i] < 0 || data[i] > limit,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, paramName.c_str(),
                                                          std::to_string(data[i]).c_str(),
                                                          "The value must be in the corresponding sequence range"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SeqLenChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    if (CheckCuValues(info.cuSeqlensQ, info.querySeqLen, "cu_seqlens_q", info) != ge::GRAPH_SUCCESS ||
        CheckCuValues(info.cuSeqlensK, info.keySeqLen, "cu_seqlens_k", info) != ge::GRAPH_SUCCESS ||
        CheckUsedValues(info.sequsedQ, info.cuSeqlensQ, info.querySeqLen, "seqused_q", info) != ge::GRAPH_SUCCESS ||
        CheckUsedValues(info.sequsedK, info.cuSeqlensK, info.keySeqLen, "seqused_k", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
