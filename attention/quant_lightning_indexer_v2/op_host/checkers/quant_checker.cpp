/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_checker.cpp
 * \brief Quantization-specific checker for QuantLightningIndexerV2
 */

#include "log/log.h"
#include "quant_checker.h"

namespace optiling {
namespace lightning_indexer_v2 {
namespace {

constexpr int64_t MX_SCALE_GROUP_SIZE = 64;
constexpr int64_t MX_SCALE_PACK_SIZE = 2;

struct QuantDtypeConfig {
    ge::DataType qk;
    ge::DataType weight;
    ge::DataType descale;
};

ge::graphStatus GetQuantDtypeConfig(int64_t quantMode, QuantDtypeConfig &config)
{
    switch (quantMode) {
        case 1:
            config = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT};
            return ge::GRAPH_SUCCESS;
        case 2:
            config = {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16};
            return ge::GRAPH_SUCCESS;
        case 3:
            config = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};
            return ge::GRAPH_SUCCESS;
        case 4:
            config = {ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT};
            return ge::GRAPH_SUCCESS;
        case 5:
            config = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};
            return ge::GRAPH_SUCCESS;
        default:
            return ge::GRAPH_FAILED;
    }
}

} // namespace

ge::graphStatus QuantChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    QuantDtypeConfig config{};
    OP_CHECK_IF(GetQuantDtypeConfig(info.quantMode, config) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE(info.opName, "quant_mode",
                                          std::to_string(info.quantMode).c_str(), "1, 2, 3, 4 or 5"),
                return ge::GRAPH_FAILED);

    if (CheckDtype(info.query, "q", {config.qk}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.key, "k", {config.qk}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.weights, "w", {config.weight}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.queryDescale, "q_descale", {config.descale}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.keyDescale, "k_descale", {config.descale}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.sparseIndices, "sparse_indices", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.sparseValues, "sparse_values", {ge::DT_BF16}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.queryDescale, "q_descale", info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.keyDescale, "k_descale", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.queryDescale, "q_descale", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.keyDescale, "k_descale", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckFeature(const IndexerCheckerInfo &info)
{
    OP_CHECK_IF(!info.isArch35 && info.quantMode != 2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "quant_mode",
                                                      std::to_string(info.quantMode).c_str(), "Ascend A2/A3 only supports quant_mode=2"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    std::vector<int64_t> queryDescaleShape;
    std::vector<int64_t> keyDescaleShape;
    if (info.quantMode == 4) {
        queryDescaleShape = {1};
        keyDescaleShape = {1};
    } else if (info.quantMode == 3 || info.quantMode == 5) {
        if (info.queryLayout == IndexerLayout::BSND) {
            queryDescaleShape = {info.batchSize, info.querySeqLen, info.queryHeadNum,
                                 info.headDim / MX_SCALE_GROUP_SIZE, MX_SCALE_PACK_SIZE};
        } else {
            queryDescaleShape = {info.querySeqLen, info.queryHeadNum,
                                 info.headDim / MX_SCALE_GROUP_SIZE, MX_SCALE_PACK_SIZE};
        }
        if (info.keyLayout == IndexerLayout::BSND) {
            keyDescaleShape = {info.batchSize, info.keySeqLen, info.keyHeadNum,
                               info.headDim / MX_SCALE_GROUP_SIZE, MX_SCALE_PACK_SIZE};
        } else if (info.keyLayout == IndexerLayout::TND) {
            keyDescaleShape = {info.keySeqLen, info.keyHeadNum,
                               info.headDim / MX_SCALE_GROUP_SIZE, MX_SCALE_PACK_SIZE};
        } else if (!info.key.shape.empty()) {
            keyDescaleShape = {info.key.shape[0], info.blockSize, info.keyHeadNum,
                               info.headDim / MX_SCALE_GROUP_SIZE, MX_SCALE_PACK_SIZE};
        }
    } else {
        if (info.queryLayout == IndexerLayout::BSND) {
            queryDescaleShape = {info.batchSize, info.querySeqLen, info.queryHeadNum};
        } else {
            queryDescaleShape = {info.querySeqLen, info.queryHeadNum};
        }
        if (info.keyLayout == IndexerLayout::BSND) {
            keyDescaleShape = {info.batchSize, info.keySeqLen, info.keyHeadNum};
        } else if (info.keyLayout == IndexerLayout::TND) {
            keyDescaleShape = {info.keySeqLen, info.keyHeadNum};
        } else if (!info.key.shape.empty()) {
            keyDescaleShape = {info.key.shape[0], info.blockSize, info.keyHeadNum};
        }
    }

    if (CheckShape(info.queryDescale, "q_descale", queryDescaleShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.keyDescale, "k_descale", keyDescaleShape, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2
} // namespace optiling
