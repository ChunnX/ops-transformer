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
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "common_checker.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// ============================================================================
// Layout — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSingleParaLayout(const FaTilingInfo &faInfo)
{
    const std::vector<FaLayout> supportedQLayouts = {FaLayout::BNSD, FaLayout::BSND, FaLayout::TND};
    const std::vector<FaLayout> supportedKvLayouts = {FaLayout::BNSD,    FaLayout::BSND,    FaLayout::TND,
                                                      FaLayout::PA_BBND, FaLayout::PA_BNBD, FaLayout::PA_NZ};
    const std::vector<FaLayout> supportedOutLayouts = {FaLayout::BNSD, FaLayout::BSND, FaLayout::TND};

    OP_CHECK_IF(std::find(supportedQLayouts.begin(), supportedQLayouts.end(), faInfo.qLayout) ==
                    supportedQLayouts.end(),
                OP_LOGE(faInfo.opName, "layout_q only supports BNSD/BSND/TND, but got %s",
                        LayoutToSerialString(faInfo.qLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) ==
                    supportedKvLayouts.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    faInfo.opName, "layout_kv", LayoutToSerialString(faInfo.kvLayout).c_str(),
                    "The value of layout_kv(layout of K/V) can only be BNSD/BSND/TND/PA_BBND/PA_BNBD/PA_NZ"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
                    supportedOutLayouts.end(),
                OP_LOGE(faInfo.opName, "layout_out only supports BNSD/BSND/TND, but got %s",
                        LayoutToSerialString(faInfo.outLayout).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Attr — SinglePara
// ============================================================================

ge::graphStatus CommonChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (CheckSingleParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus CommonChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.query.desc == nullptr || faInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE(faInfo.opName, "Query input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.key.desc == nullptr || faInfo.opParamInfo.key.shape == nullptr,
                OP_LOGE(faInfo.opName, "Key input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.value.desc == nullptr || faInfo.opParamInfo.value.shape == nullptr,
                OP_LOGE(faInfo.opName, "Value input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.attnOut.desc == nullptr || faInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE(faInfo.opName, "AttentionOut is null pointer!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Dtype
// ============================================================================

ge::graphStatus CommonChecker::CheckNonQuantDataType(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeConsistency(const FaTilingInfo &faInfo)
{
    const gert::CompileTimeTensorDesc *queryDesc = faInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc *keyDesc = faInfo.opParamInfo.key.desc;
    const gert::CompileTimeTensorDesc *valueDesc = faInfo.opParamInfo.value.desc;
    const gert::CompileTimeTensorDesc *attnOutDesc = faInfo.opParamInfo.attnOut.desc;

    ge::DataType queryDtype = queryDesc->GetDataType();

    if (keyDesc != nullptr) {
        OP_CHECK_IF(keyDesc->GetDataType() != queryDtype,
                    OP_LOGE(faInfo.opName, "key dtype(%s) should be consistent with query dtype(%s).",
                            DataTypeToSerialString(keyDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (valueDesc != nullptr) {
        OP_CHECK_IF(valueDesc->GetDataType() != queryDtype,
                    OP_LOGE(faInfo.opName, "value dtype(%s) should be consistent with query dtype(%s).",
                            DataTypeToSerialString(valueDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (attnOutDesc != nullptr) {
        OP_CHECK_IF(attnOutDesc->GetDataType() != queryDtype,
                    OP_LOGE(faInfo.opName, "attentionOut dtype(%s) should be consistent with query dtype(%s).",
                            DataTypeToSerialString(attnOutDesc->GetDataType()).c_str(),
                            DataTypeToSerialString(queryDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// HeadNum
// ============================================================================

ge::graphStatus CommonChecker::CheckNonQuantHeadNum(const FaTilingInfo &faInfo)
{
    if ((faInfo.n1Size < 0) || (faInfo.n2Size < 0)) {
        OP_LOGE(faInfo.opName, "numHeads(%ld) or numKeyValueHeads(%ld) is negative!", faInfo.n1Size, faInfo.n2Size);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(faInfo.n1Size < faInfo.n2Size,
                OP_LOGE(faInfo.opName, "numHeads(%ld) should be greater than or equal to numKeyValueHeads(%ld)!",
                        faInfo.n1Size, faInfo.n2Size),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.n1Size % faInfo.n2Size != 0,
                OP_LOGE(faInfo.opName, "numHeads(%ld) should be an integer multiple of numKeyValueHeads(%ld)!",
                        faInfo.n1Size, faInfo.n2Size),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Axis
// ============================================================================

ge::graphStatus CommonChecker::CheckAxis(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.bSize >= B_LIMIT || faInfo.bSize <= 0,
                OP_LOGE(faInfo.opName, "The axis B only support (0, %u), the current is %ld.", B_LIMIT, faInfo.bSize),
                return ge::GRAPH_FAILED);

    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(faInfo.qTSize <= 0,
                    OP_LOGE(faInfo.opName, "The axis Q_T must be greater than 0, the current is %ld.", faInfo.qTSize),
                    return ge::GRAPH_FAILED);
    }
    if (faInfo.kvLayout == FaLayout::TND) {
        OP_CHECK_IF(faInfo.kTSize <= 0,
                    OP_LOGE(faInfo.opName, "The axis KV_T must be greater than 0, the current is %ld.", faInfo.kTSize),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(faInfo.n1Size <= 0,
                OP_LOGE(faInfo.opName, "The axis Q_N must be greater than 0, the current is %ld.", faInfo.n1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.n2Size <= 0,
                OP_LOGE(faInfo.opName, "The axis KV_N must be greater than 0, the current is %ld.", faInfo.n2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s1Size <= 0,
                OP_LOGE(faInfo.opName, "The axis Q_S must be greater than 0, the current is %ld.", faInfo.s1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s2Size <= 0,
                OP_LOGE(faInfo.opName, "The axis KV_S must be greater than 0, the current is %ld.", faInfo.s2Size),
                return ge::GRAPH_FAILED);

    const std::vector<int64_t> supportedHeadDims = {64, 128, 256};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.qkHeadDim, supportedHeadDims),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(faInfo.opName, "axis D of query and key",
                                                       std::to_string(faInfo.qkHeadDim).c_str(),
                                                       "The value of axis D of query and key can only be 64/128/256"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(faInfo.vHeadDim, supportedHeadDims),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis D of value",
                                                      std::to_string(faInfo.vHeadDim).c_str(),
                                                      "The value of axis D of value can only be 64/128/256"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Layout — MultiPara constraint table
// ============================================================================

struct LayoutConstraintConfig {
    std::vector<FaLayout> supportedKvLayouts;
    std::vector<FaLayout> supportedOutLayouts;
};

static const std::map<FaLayout, LayoutConstraintConfig> LAYOUT_CONSTRAINT_TABLE = {
    {FaLayout::BNSD,
     {{FaLayout::BNSD, FaLayout::PA_BBND, FaLayout::PA_BNBD, FaLayout::PA_NZ}, {FaLayout::BNSD, FaLayout::BSND}}},
    {FaLayout::BSND, {{FaLayout::BSND, FaLayout::PA_BBND, FaLayout::PA_BNBD, FaLayout::PA_NZ}, {FaLayout::BSND}}},
    {FaLayout::TND, {{FaLayout::TND, FaLayout::PA_BBND, FaLayout::PA_BNBD, FaLayout::PA_NZ}, {FaLayout::TND}}},
};

ge::graphStatus CommonChecker::CheckMultiParaLayout(const FaTilingInfo &faInfo)
{
    auto it = LAYOUT_CONSTRAINT_TABLE.find(faInfo.qLayout);
    OP_CHECK_IF(it == LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE(faInfo.opName, "layout_q %s is not supported", LayoutToSerialString(faInfo.qLayout).c_str()),
                return ge::GRAPH_FAILED);

    const auto &config = it->second;
    const std::string qLayoutStr = LayoutToSerialString(faInfo.qLayout);

    OP_CHECK_IF(std::find(config.supportedKvLayouts.begin(), config.supportedKvLayouts.end(), faInfo.kvLayout) ==
                    config.supportedKvLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_kv must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.kvLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(config.supportedOutLayouts.begin(), config.supportedOutLayouts.end(), faInfo.outLayout) ==
                    config.supportedOutLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_out must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.outLayout).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Shape Compare
// ============================================================================

void CommonChecker::SetFaShapeCompare(const FaTilingInfo &faInfo)
{
    queryShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.query.shape->GetStorageShape(),
                                                            faInfo.qLayout, QUERY_NAME, faInfo.opName);
    keyShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.key.shape->GetStorageShape(),
                                                          faInfo.kvLayout, KEY_NAME, faInfo.opName);
    valueShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.value.shape->GetStorageShape(),
                                                            faInfo.kvLayout, VALUE_NAME, faInfo.opName);
    attnOutShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.attnOut.shape->GetStorageShape(),
                                                              faInfo.outLayout, ATTN_OUT_NAME, faInfo.opName);
}

ge::graphStatus CommonChecker::CheckQueryShape(const FaTilingInfo &faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus CommonChecker::CheckKVShapeForContinuous(const FaTilingInfo &faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s2Size);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.kTSize);

    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKVShapeForPageAttention(const FaTilingInfo &faInfo) const
{
    ge::DataType kvDtype = faInfo.opParamInfo.key.desc->GetDataType();
    uint32_t kvBlockElemNum = 32 / FABaseChecker::GetTypeSize(kvDtype);

    if (faInfo.blockSize % kvBlockElemNum != 0) {
        OP_LOGE(faInfo.opName, "block_size(%d) must be divisible by %u (32 / sizeof(kv_dtype)).", faInfo.blockSize,
                kvBlockElemNum);
        return ge::GRAPH_FAILED;
    }

    FaTilingShapeCompareParam shapeParams;
    shapeParams.Bn = static_cast<int64_t>(faInfo.totalBlockNum);
    shapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    shapeParams.Bs = static_cast<int64_t>(faInfo.blockSize);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.D0 = static_cast<int64_t>(kvBlockElemNum);

    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKVShape(const FaTilingInfo &faInfo) const
{
    if (faInfo.kvLayout == FaLayout::BNSD || faInfo.kvLayout == FaLayout::BSND || faInfo.kvLayout == FaLayout::TND) {
        return CheckKVShapeForContinuous(faInfo);
    }

    if (faInfo.kvLayout == FaLayout::PA_BBND || faInfo.kvLayout == FaLayout::PA_BNBD ||
        faInfo.kvLayout == FaLayout::PA_NZ) {
        if (faInfo.pageAttentionFlag) {
            return CheckKVShapeForPageAttention(faInfo);
        }
        OP_LOGE(faInfo.opName, "kv_layout %s requires PagedAttention enabled (block_table must be provided).",
                LayoutToSerialString(faInfo.kvLayout).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGE(faInfo.opName, "kv_layout %s is not supported.", LayoutToSerialString(faInfo.kvLayout).c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus CommonChecker::CheckAttnOutShape(const FaTilingInfo &faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    if (attnOutShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckShapeConsistency(const FaTilingInfo &faInfo)
{
    SetFaShapeCompare(faInfo);
    if (ge::GRAPH_SUCCESS != CheckQueryShape(faInfo) || ge::GRAPH_SUCCESS != CheckKVShape(faInfo) ||
        ge::GRAPH_SUCCESS != CheckAttnOutShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — combined
// ============================================================================

ge::graphStatus CommonChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (CheckMultiParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonQuantDataType(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAxis(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonQuantHeadNum(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
