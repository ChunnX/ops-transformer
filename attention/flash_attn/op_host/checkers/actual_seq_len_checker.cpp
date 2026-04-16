/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file actual_seq_len_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "actual_seq_len_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// single para
ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQDim(const FaTilingInfo &faInfo)
{
    // 校验query的actualSeqLengths的维度
    auto &actualSeqLengthsQTensor = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    uint32_t batchSize = faInfo.bSize;
    FaLayout qLayout = faInfo.qLayout;
    if (qLayout == FaLayout::TND || qLayout == FaLayout::NTD) {
        // query的layout为TND/NTD时，actualSeqLengthsQ的长度为query的batch值
        OP_CHECK_IF((actualSeqLengthsQDimNum != batchSize),
            OP_LOGE(faInfo.opName,
                    "The size(%u) of actualSeqLengthQ is not equal to the batchSize(%u) of query. "
                    "The size of actualSeqLengthQ must be equal to the batchSize of query when "
                    "the layout of query is TND or NTD.", actualSeqLengthsQDimNum, batchSize),
            return ge::GRAPH_FAILED);
    } else {
        // query为非TND/NTD，actualSeqLengthsQ的长度为1或大于等于query的batch值
        OP_CHECK_IF((actualSeqLengthsQDimNum != DIM_NUM_1 && actualSeqLengthsQDimNum < batchSize),
            OP_LOGE(faInfo.opName,
                    "The size(%u) of actualSeqLengthsQ should be greater than or equal to "
                    "the batchSize(%u) of query or equal to 1. The size of actualSeqLengthsQ should be "
                    "greater than or equal to "
                    "the batchSize of query or equal to 1.", actualSeqLengthsQDimNum, batchSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQData(const FaTilingInfo &faInfo)
{
    // 校验query的actualSeqLengthData的数值约束
    auto &actualSeqLengthsQTensor = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    uint32_t batchSize = faInfo.bSize;
    FaLayout qLayout = faInfo.qLayout;
    if (qLayout == FaLayout::TND || qLayout == FaLayout::NTD) {
        // query的layout为TND/NTD时，其值应递增，且为非负数
        for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            int64_t curSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx];
            // 其值应为递增
            if (bIdx != 0U) {
                int64_t lastSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx - 1];
                OP_CHECK_IF((curSeqLengthData < lastSeqLengthData),
                    OP_LOGE(faInfo.opName,
                            "actualSeqLengthsQ[%u](%ld) < actualSeqLengthQ[%u](%ld). "
                            "actualSeqLengthsQ must be increasing when the layout of query is "
                            "TND or NTD.", bIdx, curSeqLengthData, bIdx - 1, lastSeqLengthData),
                    return ge::GRAPH_FAILED);
            }
            // curSeqLengthData应为非负数
            OP_CHECK_IF(curSeqLengthData < 0,
                OP_LOGE(faInfo.opName,
                        "actualSeqLengthsQ[%u](%ld) is less than 0.", bIdx, curSeqLengthData),
                return ge::GRAPH_FAILED);
        }
    } else {
        // query的layout为非TND/NTD，其值应不大于Q_S，且为非负数
        int64_t sOfQuery = static_cast<int64_t>(faInfo.s1Size);
        uint32_t actualSeqLengthsSize = std::min(actualSeqLengthsQDimNum, batchSize);
        for (uint32_t i = 0; i < actualSeqLengthsSize; i++) {
            int64_t curSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[i];
            // curSeqLengthData应不大于Q_S
            OP_CHECK_IF(curSeqLengthData > sOfQuery,
                OP_LOGE(faInfo.opName,
                        "actualSeqLengthsQ[%u](%ld) is larger than Q_S(%ld). The elements of actualSeqLengthsQ should not "
                        "be larger than Q_S when "
                        "the layout of query is not TND/NTD.", i, curSeqLengthData, sOfQuery),
                return ge::GRAPH_FAILED);
            // curSeqLengthData应为非负数
            OP_CHECK_IF(curSeqLengthData < 0,
                OP_LOGE(faInfo.opName,
                        "actualSeqLengthsQ[%u](%ld) is less than 0.", i, curSeqLengthData),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvDim(const FaTilingInfo &faInfo)
{
    // 校验key/value的actualSeqLengths的维度
    auto &actualSeqLengthsKvTensor = faInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    uint32_t batchSize = faInfo.bSize;
    FaLayout kvLayout = faInfo.kvLayout;
    if (kvLayout == FaLayout::TND || kvLayout == FaLayout::NTD) {
        // key/value的layout为TND/NTD时，actualSeqLengthsKv的长度为batchSize
        OP_CHECK_IF((actualSeqLengthsKvDimNum != batchSize),
            OP_LOGE(faInfo.opName,
                    "The size(%u) of actualSeqLengthsKv is not equal to the batchSize(%u). "
                    "The size of actualSeqLengthsKv must be equal to the batchSize when "
                    "the layout of key/value is TND or NTD.", actualSeqLengthsKvDimNum, batchSize),
            return ge::GRAPH_FAILED);
    } else {
        // key/value的layout为非TND/NTD，actualSeqLengthsKv的长度为1或大于等于batchSize
        OP_CHECK_IF((actualSeqLengthsKvDimNum != DIM_NUM_1 && actualSeqLengthsKvDimNum < batchSize),
            OP_LOGE(faInfo.opName,
                    "The size(%u) of actualSeqLengthsKv should be greater than or equal to "
                    "the batchSize(%u) or equal to 1. The size of actualSeqLengthsKv should be "
                    "greater than or equal to "
                    "the batchSize or equal to 1.", actualSeqLengthsKvDimNum, batchSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvData(const FaTilingInfo &faInfo)
{
    // 校验key/value的actualSeqLengthData的数值约束
    auto &actualSeqLengthsKvTensor = faInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    uint32_t batchSize = faInfo.bSize;
    FaLayout kvLayout = faInfo.kvLayout;
    if (kvLayout == FaLayout::TND || kvLayout == FaLayout::NTD) {
        // key/value的layout为TND或NTD时，非page attention场景时，其值应递增，且为非负数
        for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            int64_t curSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx];
            // 非page attention场景时，其值应为递增
            if (bIdx != 0U) {
                int64_t lastSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx - 1];
                OP_CHECK_IF((!faInfo.pageAttentionFlag && curSeqLengthData < lastSeqLengthData),
                    OP_LOGE(faInfo.opName,
                            "actualSeqLengthsKv[%u](%ld) < actualSeqLengthsKv[%u](%ld). "
                            "actualSeqLengthsKv must be increasing when the layout of key/value is "
                            "TND or NTD.", bIdx, curSeqLengthData, bIdx - 1, lastSeqLengthData),
                    return ge::GRAPH_FAILED);
            }
            // curSeqLengthData应为非负数
            OP_CHECK_IF((curSeqLengthData < 0),
                OP_LOGE(faInfo.opName,
                        "actualSeqLengthsKv[%u](%ld) is less than 0.", bIdx, curSeqLengthData),
                return ge::GRAPH_FAILED);
        }
    } else {
        // key/value的layout为非TND/NTD，其值应不大于KV_S，且为非负数
        int64_t sOfKeyValue = static_cast<int64_t>(faInfo.s2Size);
        uint32_t actualSeqLengthsSize = std::min(actualSeqLengthsKvDimNum, batchSize);
        for (uint32_t i = 0; i < actualSeqLengthsSize; i++) {
            int64_t curSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[i];
            // curSeqLengthData应不大于KV_S
            OP_CHECK_IF(curSeqLengthData > sOfKeyValue,
                OP_LOGE(faInfo.opName,
                        "actualSeqLengthsKv[%u](%ld) is larger than KV_S(%ld).", i, curSeqLengthData, sOfKeyValue),
                return ge::GRAPH_FAILED);
            // curSeqLengthData应为非负数
            OP_CHECK_IF(curSeqLengthData < 0,
                OP_LOGE(faInfo.opName,
                    "actualSeqLengthsKv[%u](%ld) is less than 0.", i, curSeqLengthData),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQTNDLastData(const FaTilingInfo &faInfo)
{
    // 校验query的输入为TND/NTD时，actualSeqLengthQ的最后一个元素与T相等
    auto &actualSeqLengthsQTensor = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
    auto &queryShape = faInfo.opParamInfo.query.shape->GetStorageShape();
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    int64_t actualSeqLengthsQLastData = actualSeqLengthsQTensor->GetData<int64_t>()[actualSeqLengthsQDimNum - 1];
    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(actualSeqLengthsQLastData != queryShape.GetDim(DIM_NUM_0),
            OP_LOGE(faInfo.opName,
                    "The last element(%ld) of actualSeqLengthsQ is not equal to the T(%ld) of query. "
                    "The last element of actualSeqLengthsQ must be equal to the T of query when "
                    "the layout of query is TND.", actualSeqLengthsQLastData, queryShape.GetDim(DIM_NUM_0)),
            return ge::GRAPH_FAILED);
    }
    if (faInfo.qLayout == FaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsQLastData != queryShape.GetDim(DIM_NUM_1),
            OP_LOGE(faInfo.opName,
                    "The last element(%ld) of actualSeqLengthsQ is not equal to the T(%ld) of query. "
                    "The last element of actualSeqLengthsQ must be equal to the T of query when "
                    "the layout of query is NTD.", actualSeqLengthsQLastData, queryShape.GetDim(DIM_NUM_1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvTNDLastData(const FaTilingInfo &faInfo)
{
    // 校验key/value的输入为TND/NTD时，actualSeqLengthsKv的最后一个元素与T相等
    auto &actualSeqLengthsKvTensor = faInfo.opParamInfo.actualSeqLengths.tensor;
    auto &keyShape = faInfo.opParamInfo.key.shape->GetStorageShape();
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    if (faInfo.pageAttentionFlag) {
        // 若使能page attention，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    int64_t actualSeqLengthsKvLastData = actualSeqLengthsKvTensor->GetData<int64_t>()[actualSeqLengthsKvDimNum - 1];
    if (faInfo.kvLayout == FaLayout::TND) {
        OP_CHECK_IF(actualSeqLengthsKvLastData != keyShape.GetDim(DIM_NUM_0),
            OP_LOGE(faInfo.opName,
                    "The last element(%ld) of actualSeqLengthsKv is not equal to the T(%ld) of "
                    "key/value. The last element of actualSeqLengthsKv must be equal to the T of key/value "
                    "when the layout of key/value is TND.", actualSeqLengthsKvLastData, keyShape.GetDim(DIM_NUM_0)),
            return ge::GRAPH_FAILED);
    }
    if (faInfo.kvLayout == FaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsKvLastData != keyShape.GetDim(DIM_NUM_1),
            OP_LOGE(faInfo.opName,
                    "The last element(%ld) of actualSeqLengthsKv is not equal to the T(%ld) of "
                    "key/value. The last element of actualSeqLengthsKv must be equal to the T of key/value "
                    "when the layout of key/value is NTD.", actualSeqLengthsKvLastData, keyShape.GetDim(DIM_NUM_1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus ActualSeqLenChecker::CheckExistenceActualSeqLenQ(const FaTilingInfo &faInfo)
{
    // 校验actualSeqLenQ的存在性
    FaLayout qLayout = faInfo.qLayout;
    // query的Layout为TND/NTD时，actualSeqLengthsQ必须传入
    auto &actualSeqLengthsQTensor = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (qLayout == FaLayout::TND || qLayout == FaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsQTensor == nullptr,
            OP_LOGE(faInfo.opName,
                    "actualSeqLengthsQ does not exist. "
                    "actualSeqLengthsQ must exist when the layout of query is TND or NTD."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckExistenceActualSeqLenKv(const FaTilingInfo &faInfo)
{
    FaLayout kvLayout = faInfo.kvLayout;
    // key、value的layout为TND/NTD时，actualSeqLengthsKv必须传入
    if (kvLayout == FaLayout::TND || kvLayout == FaLayout::NTD) {
        OP_CHECK_IF(faInfo.opParamInfo.actualSeqLengths.tensor == nullptr,
            OP_LOGE(faInfo.opName,
                    "actualSeqLengthsKv does not exist. "
                    "actualSeqLengthsKv must exist when the layout of key and value is TND or NTD."),
            return ge::GRAPH_FAILED);
    }
    // PagedAttention场景下，必须传入actualSeqLengthsKv
    if (faInfo.pageAttentionFlag) {
        OP_CHECK_IF(faInfo.opParamInfo.actualSeqLengths.tensor == nullptr,
            OP_LOGE(faInfo.opName,
                    "actualSeqLengthsKv does not exist. "
                    "actualSeqLengthsKv must exist when page attention is enabled."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus ActualSeqLenChecker::CheckFeatureAlibi(const FaTilingInfo &faInfo)
{
    // 使能alibi pse时，query和key每个batch的seqlength需要相等
    int64_t actualSeqLengthsQData = 0;
    int64_t actualSeqLengthsKvData = 0;
    uint32_t batchSize = faInfo.bSize;
    if (faInfo.enableAlibiPse) {
        for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            actualSeqLengthsQData = GetActualSeqLengthsQData(faInfo, bIdx);
            actualSeqLengthsKvData = GetActualSeqLengthsKvData(faInfo, bIdx);
            OP_CHECK_IF((actualSeqLengthsQData != actualSeqLengthsKvData),
                OP_LOGE(faInfo.opName,
                        "actualSeqLengthsQData(%ld) and actualSeqLengthsKvData(%ld) are "
                        "different when batch = %u. actualSeqLengthsQData and "
                        "actualSeqLengthsKvData must be equal in each batch when "
                        "pseType is 2 or 3.", actualSeqLengthsQData, actualSeqLengthsKvData, bIdx),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeatureIFAMLA(const FaTilingInfo &faInfo)
{
    auto &actualSeqLengthsQTensor = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
    FaLayout qLayout = faInfo.qLayout;
    // IFAMLA全量化场景，仅query的layout为TND/NTD时支持传入actualSeqLengthsQ
    bool enableIFAMLA = (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA && actualSeqLengthsQTensor != nullptr) {
        OP_CHECK_IF((qLayout != FaLayout::TND) && (qLayout != FaLayout::NTD),
            OP_LOGE(faInfo.opName,
                    "actualSeqLengthsQ cannot be configured in IFA MLA and non-TND/NTD scenarios."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// general
int64_t ActualSeqLenChecker::GetActualSeqLengthsQData(const FaTilingInfo &faInfo, uint32_t bIdx)
{
    // 获取bIdx对应batch的query的seqLength
    FaLayout qLayout = faInfo.qLayout;
    int64_t actualSeqLengthsQData = 0;
    auto &actualSeqLengthsQTensor = faInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (faInfo.isAccumQSeq) {
        // actualSeqLengths为累加形式
        if (bIdx == 0U) {
            actualSeqLengthsQData = actualSeqLengthsQTensor->GetData<int64_t>()[0];
        } else {
            actualSeqLengthsQData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx] -
                                    actualSeqLengthsQTensor->GetData<int64_t>()[bIdx - 1];
        }
    } else {
        // 非累加形式
        if (actualSeqLengthsQTensor != nullptr) {
            actualSeqLengthsQData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx];
        } else {
            actualSeqLengthsQData = faInfo.s1Size;
        }
    }
    return actualSeqLengthsQData;
}

int64_t ActualSeqLenChecker::GetActualSeqLengthsKvData(const FaTilingInfo &faInfo, uint32_t bIdx)
{
    // 获取bIdx对应batch的key和value的seqLength
    FaLayout kvLayout = faInfo.kvLayout;
    int64_t actualSeqLengthsKvData = 0;
    auto &actualSeqLengthsKvTensor = faInfo.opParamInfo.actualSeqLengths.tensor;
    if (faInfo.isAccumKVSeq) {
        // actualSeqLengths为累加形式
        if (bIdx == 0U) {
            actualSeqLengthsKvData = actualSeqLengthsKvTensor->GetData<int64_t>()[0];
        } else {
            actualSeqLengthsKvData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx] -
                                     actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx - 1];
        }
    } else {
        // 非累加形式
        if (actualSeqLengthsKvTensor != nullptr) {
            actualSeqLengthsKvData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx];
        } else {
            actualSeqLengthsKvData = faInfo.s2Size;
        }
    }
    return actualSeqLengthsKvData;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (faInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckActualSeqLenQDim(faInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenQData(faInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvDim(faInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvData(faInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenQTNDLastData(faInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvTNDLastData(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (faInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckExistenceActualSeqLenQ(faInfo) ||
        ge::GRAPH_SUCCESS != CheckExistenceActualSeqLenKv(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (faInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    
    if (ge::GRAPH_SUCCESS != CheckFeatureAlibi(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureIFAMLA(faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (faInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
