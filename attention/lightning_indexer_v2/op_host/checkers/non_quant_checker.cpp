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
#include "non_quant_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus NonQuantChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (CheckDtype(info.query, "query", {ge::DT_FLOAT16, ge::DT_BF16}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.key, "key", {ge::DT_FLOAT16, ge::DT_BF16}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.weights, "weights", {ge::DT_FLOAT}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.sparseIndices, "sparse_indices", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.sparseValues, "sparse_values", {ge::DT_FLOAT}, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.query.desc != nullptr && info.key.desc != nullptr &&
        info.query.desc->GetDataType() != info.key.desc->GetDataType()) {
        const std::string actual = DataTypeToString(info.query.desc->GetDataType()) + " and " +
                                   DataTypeToString(info.key.desc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(info.opName, "query and key", actual.c_str(),
                                               "The query and key must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
