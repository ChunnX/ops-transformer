/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "log/log.h"
#include "common_checker.h"

namespace optiling {
namespace lightning_indexer_v2 {

namespace {
ge::graphStatus CheckResult(ge::graphStatus status)
{
    return status == ge::GRAPH_SUCCESS ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
} // namespace

ge::graphStatus CommonChecker::CheckParaExistence(const IndexerCheckerInfo &info)
{
    if (CheckRequiredTensor(info.query, "query", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.key, "key", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.weights, "weights", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.sparseIndices, "sparse_indices", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.sparseValues, "sparse_values", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.quantized &&
        (CheckRequiredTensor(info.queryDescale, "query_dequant_scale", info) != ge::GRAPH_SUCCESS ||
         CheckRequiredTensor(info.keyDescale, "key_dequant_scale", info) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    if (info.queryLayout != IndexerLayout::BSND && info.queryLayout != IndexerLayout::TND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_q", LayoutToString(info.queryLayout).c_str(),
                                              "The value of layout_q must be BSND or TND");
        return ge::GRAPH_FAILED;
    }
    if (info.keyLayout != IndexerLayout::BSND && info.keyLayout != IndexerLayout::TND &&
        info.keyLayout != IndexerLayout::PA_BBND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_k", LayoutToString(info.keyLayout).c_str(),
                                              "The value of layout_k must be BSND, TND or PA_BBND");
        return ge::GRAPH_FAILED;
    }
    if ((info.queryLayout == IndexerLayout::BSND && info.keyLayout == IndexerLayout::TND) ||
        (info.queryLayout == IndexerLayout::TND && info.keyLayout == IndexerLayout::BSND)) {
        const std::string layouts = LayoutToString(info.queryLayout) + " and " + LayoutToString(info.keyLayout);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "layout_q and layout_k", layouts.c_str(),
                                               "The query and key layouts do not form a supported combination");
        return ge::GRAPH_FAILED;
    }
    if (info.maxSeqlenQ != -1) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "max_seqlen_q", std::to_string(info.maxSeqlenQ).c_str(), "-1");
        return ge::GRAPH_FAILED;
    }
    if (info.returnValue != 0 && info.returnValue != 1) {
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "return_value", std::to_string(info.returnValue).c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }

    const CheckerTensor *tensors[] = {&info.query, &info.key, &info.weights, &info.sparseIndices, &info.sparseValues};
    const char *names[] = {"query", "key", "weights", "sparse_indices", "sparse_values"};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        if (CheckFormat(*tensors[i], names[i], info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckNoZeroDim(info.query, "query", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.key, "key", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.weights, "weights", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.sparseIndices, "sparse_indices", info) != ge::GRAPH_SUCCESS ||
        (info.returnValue == 1 && CheckNoZeroDim(info.sparseValues, "sparse_values", info) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return CheckOutputOffset(info);
}

ge::graphStatus CommonChecker::CheckFeature(const IndexerCheckerInfo &info)
{
    if (info.batchSize <= 0 || info.batchSize >= 65536 || info.headDim != 128 || info.keyHeadNum != 1 ||
        info.queryHeadNum <= 0 || info.queryHeadNum % info.keyHeadNum != 0 || info.groupSize > 64) {
        const std::string actual = "B=" + std::to_string(info.batchSize) + ", D=" + std::to_string(info.headDim) +
                                   ", N1=" + std::to_string(info.queryHeadNum) +
                                   ", N2=" + std::to_string(info.keyHeadNum) +
                                   ", G=" + std::to_string(info.groupSize);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "B, D, N1, N2 and G", actual.c_str(),
                                               "B, D, N1, N2 or G violates the documented range");
        return ge::GRAPH_FAILED;
    }
    if (info.groupSize <= 0 || info.groupSize != info.queryHeadNum / info.keyHeadNum) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "G", std::to_string(info.groupSize).c_str(),
                                              "G must equal N1 divided by N2 and be in [1, 64]");
        return ge::GRAPH_FAILED;
    }
    if (info.isArch35) {
        if (info.queryHeadNum > 64 || info.topk < 1 || info.topk > 8192 || info.cmpRatio < 1 ||
            info.cmpRatio > 128) {
            const std::string actual = "N1=" + std::to_string(info.queryHeadNum) +
                                       ", topk=" + std::to_string(info.topk) +
                                       ", cmp_ratio=" + std::to_string(info.cmpRatio);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "N1, topk and cmp_ratio", actual.c_str(),
                                                   "The values are unsupported on Ascend 950");
            return ge::GRAPH_FAILED;
        }
    } else {
        const bool ratioSupported = info.cmpRatio >= 1 && info.cmpRatio <= 128 &&
                                    (info.cmpRatio & (info.cmpRatio - 1)) == 0;
        if (info.queryHeadNum != 64 || info.topk < 1 || info.topk > 2048 || !ratioSupported ||
            info.keyLayout != IndexerLayout::PA_BBND || info.outputIdxOffset.present || info.returnValue != 0) {
            const std::string actual = "N1=" + std::to_string(info.queryHeadNum) +
                                       ", topk=" + std::to_string(info.topk) +
                                       ", cmp_ratio=" + std::to_string(info.cmpRatio) +
                                       ", layout_k=" + LayoutToString(info.keyLayout) +
                                       ", return_value=" + std::to_string(info.returnValue);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "A2/A3 feature parameters", actual.c_str(),
                                                   "The feature combination is unsupported on Ascend A2/A3");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    std::vector<int64_t> queryShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> outputShape;
    if (info.queryLayout == IndexerLayout::BSND) {
        queryShape = {info.batchSize, info.querySeqLen, info.queryHeadNum, info.headDim};
        weightShape = {info.batchSize, info.querySeqLen, info.queryHeadNum};
        outputShape = {info.batchSize, info.querySeqLen, info.keyHeadNum, info.topk};
    } else {
        queryShape = {info.querySeqLen, info.queryHeadNum, info.headDim};
        weightShape = {info.querySeqLen, info.queryHeadNum};
        outputShape = {info.querySeqLen, info.keyHeadNum, info.topk};
    }

    std::vector<int64_t> keyShape;
    if (info.keyLayout == IndexerLayout::BSND) {
        keyShape = {info.batchSize, info.keySeqLen, info.keyHeadNum, info.headDim};
    } else if (info.keyLayout == IndexerLayout::TND) {
        keyShape = {info.keySeqLen, info.keyHeadNum, info.headDim};
    } else if (info.key.shape.size() == 4) {
        keyShape = {info.key.shape[0], info.blockSize, info.keyHeadNum, info.headDim};
    }

    if (CheckShape(info.query, "query", queryShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.key, "key", keyShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.weights, "weights", weightShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.sparseIndices, "sparse_indices", outputShape, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const std::vector<int64_t> valueShape = info.returnValue == 1 ? outputShape : std::vector<int64_t>{0};
    return CheckResult(CheckShape(info.sparseValues, "sparse_values", valueShape, info));
}

ge::graphStatus CommonChecker::CheckOutputOffset(const IndexerCheckerInfo &info) const
{
    if (!info.outputIdxOffset.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckOptionalTensor(info.outputIdxOffset, "output_idx_offset", info) != ge::GRAPH_SUCCESS ||
        CheckDtype(info.outputIdxOffset, "output_idx_offset", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.outputIdxOffset, "output_idx_offset", info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.outputIdxOffset, "output_idx_offset", {info.batchSize}, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int32_t *data = info.outputIdxOffset.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t maxIndex = info.keySeqLen * info.cmpRatio - 1;
    for (int64_t i = 0; i < info.batchSize; ++i) {
        if (data[i] <= 0 || static_cast<int64_t>(data[i]) + maxIndex > INT32_MAX) {
            const std::string paramName = "output_idx_offset[" + std::to_string(i) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, paramName.c_str(), std::to_string(data[i]).c_str(),
                                                  "The value must be greater than 0 and must not cause INT32 overflow");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2
} // namespace optiling
