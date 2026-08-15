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
 * \file common_checker.cpp
 * \brief Common parameter checker for the LightningIndexerV2 operator family
 */

#include <limits>
#include "log/log.h"
#include "common_checker.h"

namespace optiling {
namespace lightning_indexer_v2 {
namespace {

bool IsPowerOfTwo(int64_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            return 0;
        }
        size *= dim;
    }
    return size;
}

} // namespace

ge::graphStatus CommonChecker::CheckSinglePara(const IndexerCheckerInfo &info)
{
    OP_CHECK_IF(info.queryLayout != IndexerLayout::BSND && info.queryLayout != IndexerLayout::TND,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_q",
                                                      LayoutToString(info.queryLayout).c_str(), "layout_q only supports BSND and TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.keyLayout != IndexerLayout::BSND && info.keyLayout != IndexerLayout::TND &&
                    info.keyLayout != IndexerLayout::PA_BBND,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "layout_k",
                                                      LayoutToString(info.keyLayout).c_str(), "layout_k only supports BSND, TND and PA_BBND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.maxSeqlenQ != -1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "max_seqlen_q",
                                                      std::to_string(info.maxSeqlenQ).c_str(), "max_seqlen_q is not effective and only supports -1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.returnValue != 0 && info.returnValue != 1,
                OP_LOGE_FOR_INVALID_VALUE(info.opName, "return_value",
                                          std::to_string(info.returnValue).c_str(), "0 or 1"),
                return ge::GRAPH_FAILED);

    const CheckerTensor *requiredTensors[] = {
        &info.query, &info.key, &info.weights, &info.sparseIndices, &info.sparseValues};
    const char *requiredNames[] = {"q", "k", "w", "sparse_indices", "sparse_values"};
    for (size_t i = 0; i < sizeof(requiredTensors) / sizeof(requiredTensors[0]); ++i) {
        if (CheckFormat(*requiredTensors[i], requiredNames[i], info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (CheckNoZeroDim(info.query, "q", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.key, "k", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.weights, "w", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.sparseIndices, "sparse_indices", info) != ge::GRAPH_SUCCESS ||
        (info.returnValue == 1 && CheckNoZeroDim(info.sparseValues, "sparse_values", info) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    const CheckerTensor *optionalTensors[] = {
        &info.cuSeqlensQ, &info.cuSeqlensK, &info.sequsedQ, &info.sequsedK, &info.cmpResidualK,
        &info.blockTable, &info.outputIdxOffset, &info.metadata};
    const char *optionalNames[] = {
        "cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k",
        "block_table", "output_idx_offset", "metadata"};
    for (size_t i = 0; i < sizeof(optionalTensors) / sizeof(optionalTensors[0]); ++i) {
        if (CheckOptionalTensor(*optionalTensors[i], optionalNames[i], info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return CheckOutputOffset(info);
}

ge::graphStatus CommonChecker::CheckParaExistence(const IndexerCheckerInfo &info)
{
    if (CheckRequiredTensor(info.query, "q", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.key, "k", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.weights, "w", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.sparseIndices, "sparse_indices", info) != ge::GRAPH_SUCCESS ||
        CheckRequiredTensor(info.sparseValues, "sparse_values", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (info.quantized &&
        (CheckRequiredTensor(info.queryDescale, "q_descale", info) != ge::GRAPH_SUCCESS ||
         CheckRequiredTensor(info.keyDescale, "k_descale", info) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const IndexerCheckerInfo &info)
{
    OP_CHECK_IF(info.batchSize <= 0 || info.batchSize >= INDEXER_BATCH_LIMIT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "batch_size",
                                                      std::to_string(info.batchSize).c_str(), "batch_size must be in (0, 65536)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.querySeqLen <= 0 || info.keySeqLen <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "q_s/q_t and k_s/k_t",
                                                       (std::to_string(info.querySeqLen) + ", " + std::to_string(info.keySeqLen)).c_str(),
                                                       "query and key sequence dimensions must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.headDim != INDEXER_HEAD_DIM,
                OP_LOGE_FOR_INVALID_VALUE(info.opName, "head_dim",
                                          std::to_string(info.headDim).c_str(), "128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.keyHeadNum != INDEXER_KEY_HEAD_NUM,
                OP_LOGE_FOR_INVALID_VALUE(info.opName, "num_heads_k",
                                          std::to_string(info.keyHeadNum).c_str(), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.queryHeadNum <= 0 || info.queryHeadNum % info.keyHeadNum != 0 ||
                    info.groupSize != info.queryHeadNum / info.keyHeadNum,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "num_heads_q, num_heads_k and group_size",
                                                       (std::to_string(info.queryHeadNum) + ", " + std::to_string(info.keyHeadNum) + ", " +
                                                        std::to_string(info.groupSize))
                                                           .c_str(),
                                                       "num_heads_q must be divisible by num_heads_k and group_size must equal their quotient"),
                return ge::GRAPH_FAILED);

    if (info.isArch35) {
        OP_CHECK_IF(info.queryHeadNum > INDEXER_HEAD_NUM_MAX || info.topk < 1 ||
                        info.topk > INDEXER_TOPK_MAX_ARCH35 || info.cmpRatio < 1 ||
                        info.cmpRatio > INDEXER_CMP_RATIO_MAX,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName,
                                                           "num_heads_q, topk and cmp_ratio",
                                                           (std::to_string(info.queryHeadNum) + ", " + std::to_string(info.topk) + ", " +
                                                            std::to_string(info.cmpRatio))
                                                               .c_str(),
                                                           "Ascend 950 requires num_heads_q in [1,64], topk in [1,8192] and cmp_ratio in [1,128]"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(info.queryHeadNum != INDEXER_HEAD_NUM_MAX || info.topk < 1 ||
                    info.topk > INDEXER_TOPK_MAX_ARCH22 || !IsPowerOfTwo(info.cmpRatio) ||
                    info.cmpRatio > INDEXER_CMP_RATIO_MAX || info.keyLayout != IndexerLayout::PA_BBND ||
                    info.outputIdxOffset.present || info.returnValue != 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "A2/A3 feature parameters",
                                                         "A2/A3 requires num_heads_q=64, topk in [1,2048], power-of-two cmp_ratio in [1,128], "
                                                         "layout_k=PA_BBND, no output_idx_offset and return_value=0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const IndexerCheckerInfo &info)
{
    const bool layoutMatched =
        (info.queryLayout == IndexerLayout::BSND &&
         (info.keyLayout == IndexerLayout::BSND || info.keyLayout == IndexerLayout::PA_BBND)) ||
        (info.queryLayout == IndexerLayout::TND &&
         (info.keyLayout == IndexerLayout::TND || info.keyLayout == IndexerLayout::PA_BBND));
    OP_CHECK_IF(!layoutMatched,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(info.opName, "layout_q and layout_k",
                                                       (LayoutToString(info.queryLayout) + ", " + LayoutToString(info.keyLayout)).c_str(),
                                                       "BSND query supports BSND/PA_BBND key and TND query supports TND/PA_BBND key"),
                return ge::GRAPH_FAILED);

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
    } else if (!info.key.shape.empty()) {
        keyShape = {info.key.shape[0], info.blockSize, info.keyHeadNum, info.headDim};
    }

    if (CheckShape(info.query, "q", queryShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.key, "k", keyShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.weights, "w", weightShape, info) != ge::GRAPH_SUCCESS ||
        CheckShape(info.sparseIndices, "sparse_indices", outputShape, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const std::vector<int64_t> sparseValuesShape =
        info.returnValue == 1 ? outputShape : std::vector<int64_t>{0};
    if (CheckShape(info.sparseValues, "sparse_values", sparseValuesShape, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckOutputOffsetConsistency(info);
}

ge::graphStatus CommonChecker::CheckOutputOffset(const IndexerCheckerInfo &info) const
{
    if (!info.outputIdxOffset.present) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckDtype(info.outputIdxOffset, "output_idx_offset", {ge::DT_INT32}, info) != ge::GRAPH_SUCCESS ||
        CheckFormat(info.outputIdxOffset, "output_idx_offset", info) != ge::GRAPH_SUCCESS ||
        CheckNoZeroDim(info.outputIdxOffset, "output_idx_offset", info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int32_t *data = info.outputIdxOffset.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t elementCount = ShapeSize(info.outputIdxOffset.shape);
    for (int64_t i = 0; i < elementCount; ++i) {
        OP_CHECK_IF(data[i] <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "output_idx_offset",
                                                          std::to_string(data[i]).c_str(),
                                                          "every offset must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckOutputOffsetConsistency(const IndexerCheckerInfo &info) const
{
    if (!info.outputIdxOffset.present) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<int64_t> expectedShape = info.queryLayout == IndexerLayout::BSND ?
                                                   std::vector<int64_t>{info.batchSize, info.querySeqLen, info.keyHeadNum} :
                                                   std::vector<int64_t>{info.querySeqLen, info.keyHeadNum};
    if (CheckShape(info.outputIdxOffset, "output_idx_offset", expectedShape, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    constexpr int64_t int32Max = std::numeric_limits<int32_t>::max();
    OP_CHECK_IF(info.keySeqLen <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "output_idx_offset",
                                                         "key sequence length must be greater than 0"),
                return ge::GRAPH_FAILED);
    const int64_t maxIndex = info.keySeqLen - 1;
    const int32_t *data = info.outputIdxOffset.tensor->GetData<int32_t>();
    if (data == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t elementCount = ShapeSize(info.outputIdxOffset.shape);
    for (int64_t i = 0; i < elementCount; ++i) {
        OP_CHECK_IF(maxIndex > int32Max - static_cast<int64_t>(data[i]),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(info.opName, "output_idx_offset",
                                                          std::to_string(data[i]).c_str(),
                                                          "the offset must not make sparse_indices exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2
} // namespace optiling
