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
 * \file shape_checker.cpp
 * \brief Checker for shape parameters (文档约束: 公共参数组 + Paged Attention参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "shape_checker.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// ============================================================================
// 公共参数组 - 存在性校验
// ============================================================================

ge::graphStatus ShapeChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.query.desc == nullptr || faInfo.opParamInfo.query.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Query input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.key.desc == nullptr || faInfo.opParamInfo.key.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Key input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.value.desc == nullptr || faInfo.opParamInfo.value.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "Value input is null pointer!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.attnOut.desc == nullptr || faInfo.opParamInfo.attnOut.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName, "AttentionOut is null pointer!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// 公共参数组 - 单参数校验
// ============================================================================

ge::graphStatus ShapeChecker::CheckNonQuantDataType(const FaTilingInfo &faInfo)
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

// ============================================================================
// 公共参数组 - 一致性校验
// ============================================================================

ge::graphStatus ShapeChecker::CheckDtypeConsistency(const FaTilingInfo &faInfo)
{
    const gert::CompileTimeTensorDesc *queryDesc = faInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc *keyDesc = faInfo.opParamInfo.key.desc;
    const gert::CompileTimeTensorDesc *valueDesc = faInfo.opParamInfo.value.desc;
    const gert::CompileTimeTensorDesc *attnOutDesc = faInfo.opParamInfo.attnOut.desc;

    ge::DataType queryDtype = queryDesc->GetDataType();

    if (keyDesc != nullptr) {
        OP_CHECK_IF(keyDesc->GetDataType() != queryDtype,
                    OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                                "key dtype(%s) should be consistent with query dtype(%s).",
                                                DataTypeToSerialString(keyDesc->GetDataType()).c_str(),
                                                DataTypeToSerialString(queryDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (valueDesc != nullptr) {
        OP_CHECK_IF(valueDesc->GetDataType() != queryDtype,
                    OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                                "value dtype(%s) should be consistent with query dtype(%s).",
                                                DataTypeToSerialString(valueDesc->GetDataType()).c_str(),
                                                DataTypeToSerialString(queryDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (attnOutDesc != nullptr) {
        OP_CHECK_IF(attnOutDesc->GetDataType() != queryDtype,
                    OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                                "attentionOut dtype(%s) should be consistent with query dtype(%s).",
                                                DataTypeToSerialString(attnOutDesc->GetDataType()).c_str(),
                                                DataTypeToSerialString(queryDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus ShapeChecker::CheckNonQuantHeadNum(const FaTilingInfo &faInfo)
{
    if ((faInfo.n1Size < 0) || (faInfo.n2Size < 0)) {
        OP_LOGE(faInfo.opName, "numHeads(%ld) or numKeyValueHeads(%ld) is negative!", faInfo.n1Size, faInfo.n2Size);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((faInfo.n2Size != 0) && (faInfo.n1Size % faInfo.n2Size != 0),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                            "numHeads(%ld) should be an integer multiple of numKeyValueHeads(%ld)!",
                                            faInfo.n1Size, faInfo.n2Size),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// 公共参数组 - 特性交叉校验（轴校验）
// ============================================================================

ge::graphStatus ShapeChecker::CheckAxis(const FaTilingInfo &faInfo)
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

    OP_CHECK_IF(faInfo.qkHeadDim > D_LIMIT || faInfo.qkHeadDim < 16,
                OP_LOGE(faInfo.opName, "The axis D of query and key only support [16, %u], the current is %ld.",
                        D_LIMIT, faInfo.qkHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.vHeadDim > D_LIMIT || faInfo.vHeadDim < 16,
                OP_LOGE(faInfo.opName, "The axis D of value only support [16, %u], the current is %ld.", D_LIMIT,
                        faInfo.vHeadDim),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        faInfo.qkHeadDim % 16 != 0,
        OP_LOGE(faInfo.opName, "The axis D of query and key must be 16 aligned, the current is %ld.", faInfo.qkHeadDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.vHeadDim % 16 != 0,
                OP_LOGE(faInfo.opName, "The axis D of value must be 16 aligned, the current is %ld.", faInfo.vHeadDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// 公共参数组 - 多参数校验
// ============================================================================

ge::graphStatus ShapeChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
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

// ============================================================================
// Shape Consistency - 使用 FaTilingShapeCompare
// ============================================================================

void ShapeChecker::SetFaShapeCompare(const FaTilingInfo &faInfo)
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

ge::graphStatus ShapeChecker::CheckQueryShape(const FaTilingInfo &faInfo) const
{
    FaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus ShapeChecker::CheckKVShapeForContinuous(const FaTilingInfo &faInfo) const
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

ge::graphStatus ShapeChecker::CheckKVShapeForPageAttention(const FaTilingInfo &faInfo) const
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

ge::graphStatus ShapeChecker::CheckKVShape(const FaTilingInfo &faInfo) const
{
    // 连续存储 layout
    if (faInfo.kvLayout == FaLayout::BNSD || faInfo.kvLayout == FaLayout::BSND || faInfo.kvLayout == FaLayout::TND) {
        return CheckKVShapeForContinuous(faInfo);
    }

    // PA layout
    if (faInfo.kvLayout == FaLayout::PA_BBND || faInfo.kvLayout == FaLayout::PA_BNBD ||
        faInfo.kvLayout == FaLayout::PA_Nz) {
        return CheckKVShapeForPageAttention(faInfo);
    }

    OP_LOGE(faInfo.opName, "kv_layout %d is not supported.", static_cast<int>(faInfo.kvLayout));
    return ge::GRAPH_FAILED;
}

ge::graphStatus ShapeChecker::CheckAttnOutShape(const FaTilingInfo &faInfo) const
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


ge::graphStatus ShapeChecker::CheckShapeConsistency(const FaTilingInfo &faInfo)
{
    SetFaShapeCompare(faInfo);
    if (ge::GRAPH_SUCCESS != CheckQueryShape(faInfo) || ge::GRAPH_SUCCESS != CheckKVShape(faInfo) ||
        ge::GRAPH_SUCCESS != CheckAttnOutShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling