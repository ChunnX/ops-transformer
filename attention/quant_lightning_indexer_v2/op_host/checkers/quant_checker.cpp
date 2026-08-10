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
#include "quant_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
ge::graphStatus QuantChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (info.quantMode < 1 || info.quantMode > 5) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "quant_mode", std::to_string(info.quantMode).c_str(), "[1, 5]");
        return ge::GRAPH_FAILED;
    }
    ge::DataType qType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType weightType = ge::DT_FLOAT;
    ge::DataType scaleType = ge::DT_FLOAT;
    if (info.quantMode == 2) {
        qType = ge::DT_INT8;
        weightType = ge::DT_FLOAT16;
        scaleType = ge::DT_FLOAT16;
    } else if (info.quantMode == 3 || info.quantMode == 5) {
        qType = info.quantMode == 3 ? ge::DT_FLOAT8_E4M3FN : ge::DT_FLOAT4_E2M1;
        scaleType = ge::DT_FLOAT8_E8M0;
    } else if (info.quantMode == 4) {
        qType = ge::DT_HIFLOAT8;
    }
    if (CheckDtype(info.query, "query", {qType}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.key, "key", {qType}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.weights, "weights", {weightType}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.queryDescale, "query_dequant_scale", {scaleType}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.keyDescale, "key_dequant_scale", {scaleType}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.sparseIndices, "sparse_indices", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.sparseValues, "sparse_values", {ge::DT_BF16}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.queryDescale, "query_dequant_scale", info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.keyDescale, "key_dequant_scale", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.queryDescale, "query_dequant_scale", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.keyDescale, "key_dequant_scale", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QuantChecker::CheckFeature(const IndexerCheckerInfo &info)
{
    OP_CHECK_IF(!info.isArch35 && info.quantMode != 2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "quant_mode",
                                                      std::to_string(info.quantMode).c_str(),
                                                      "Ascend A2/A3 only supports quant_mode 2"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QuantChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    std::vector<int64_t> qScale;
    std::vector<int64_t> kScale;
    if (info.quantMode == 4) {
        qScale = {1};
        kScale = {1};
    } else if (info.quantMode == 3 || info.quantMode == 5) {
        if (info.queryLayout == IndexerLayout::BSND) {
            qScale = {info.batchSize, info.querySeqLen, info.queryHeadNum, info.headDim / 64, 2};
        } else {
            qScale = {info.querySeqLen, info.queryHeadNum, info.headDim / 64, 2};
        }
        if (info.keyLayout == IndexerLayout::BSND) {
            kScale = {info.batchSize, info.keySeqLen, info.keyHeadNum, info.headDim / 64, 2};
        } else if (info.keyLayout == IndexerLayout::TND) {
            kScale = {info.keySeqLen, info.keyHeadNum, info.headDim / 64, 2};
        } else {
            kScale = {info.key.shape[0], info.blockSize, info.keyHeadNum, info.headDim / 64, 2};
        }
    } else {
        if (info.queryLayout == IndexerLayout::BSND) {
            qScale = {info.batchSize, info.querySeqLen, info.queryHeadNum};
        } else {
            qScale = {info.querySeqLen, info.queryHeadNum};
        }
        if (info.keyLayout == IndexerLayout::BSND) {
            kScale = {info.batchSize, info.keySeqLen, info.keyHeadNum};
        } else if (info.keyLayout == IndexerLayout::TND) {
            kScale = {info.keySeqLen, info.keyHeadNum};
        } else {
            kScale = {info.key.shape[0], info.blockSize, info.keyHeadNum};
        }
    }
    if (CheckShape(info.queryDescale, "query_dequant_scale", qScale, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.keyDescale, "key_dequant_scale", kScale, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
