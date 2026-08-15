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
#include "compression_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus CompressionChecker::CheckParaExistence(const IndexerCheckerInfo &info)
{
    const bool required = info.cmpRatio > 1 && info.maskMode == 3;
    OP_CHECK_IF(required != info.cmpResidualK.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    info.opName, "cmp_residual_k",
                    "cmp_residual_k must be provided iff cmp_ratio > 1 and mask_mode == 3"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CompressionChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (!info.cmpResidualK.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckDtype(info.cmpResidualK, "cmp_residual_k", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.cmpResidualK, "cmp_residual_k", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckShape(info.cmpResidualK, "cmp_residual_k", {info.batchSize}, info);
}
ge::graphStatus CompressionChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    if (!info.cmpResidualK.present) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t *data = info.cmpResidualK.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t i = 0; i < info.batchSize; ++i) {
        const std::string paramName = "cmp_residual_k[" + std::to_string(i) + "]";
        OP_CHECK_IF(data[i] < 0 || data[i] >= info.cmpRatio,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, paramName.c_str(),
                                                          std::to_string(data[i]).c_str(),
                                                          "The value must be in [0, cmp_ratio)"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
