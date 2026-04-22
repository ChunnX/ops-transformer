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
 * \brief
 */

#include <cmath>
#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "shape_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus ShapeChecker::CheckPAKeyValueDim3(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;

    uint32_t keyBlockNum = keyShape->GetStorageShape().GetDim(0);
    uint32_t keyBlockSize = keyShape->GetStorageShape().GetDim(1);
    uint32_t keyH = keyShape->GetStorageShape().GetDim(2);

    uint32_t valueBlockNum = valueShape->GetStorageShape().GetDim(0);
    uint32_t valueBlockSize = valueShape->GetStorageShape().GetDim(1);
    uint32_t valueH = valueShape->GetStorageShape().GetDim(2);

    OP_CHECK_IF((keyBlockNum != valueBlockNum) || (keyBlockSize != valueBlockSize) || (keyH != valueH),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                            "The dim num of key and value are inconsistent when PA enable. key "
                                            "shape [%u,%u,%u], value shape [%u,%u,%u].",
                                            keyBlockNum, keyBlockSize, keyH, valueBlockNum, valueBlockSize, valueH),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckPAKeyValueDim4(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;

    uint32_t keyBlockNum = keyShape->GetStorageShape().GetDim(0);
    uint32_t keyHeadNum = keyShape->GetStorageShape().GetDim(1);
    uint32_t keyBlockSize = keyShape->GetStorageShape().GetDim(2);
    uint32_t keyHeadDim = keyShape->GetStorageShape().GetDim(3);

    uint32_t valueBlockNum = valueShape->GetStorageShape().GetDim(0);
    uint32_t valueHeadNum = valueShape->GetStorageShape().GetDim(1);
    uint32_t valueBlockSize = valueShape->GetStorageShape().GetDim(2);
    uint32_t valueHeadDim = valueShape->GetStorageShape().GetDim(3);

    OP_CHECK_IF((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) || (keyBlockSize != valueBlockSize) ||
                    (keyHeadDim != valueHeadDim),
                OPS_REPORT_VECTOR_INNER_ERR(
                    faInfo.opName,
                    "the dim num of key and value are inconsistent when PA enable. key shape [%u,%u,%u,%u],"
                    " value shape [%u,%u,%u,%u].",
                    keyBlockNum, keyHeadNum, keyBlockSize, keyHeadDim, valueBlockNum, valueHeadNum, valueBlockSize,
                    valueHeadDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckPAKeyValueDim5(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;

    uint32_t keyBlockNum = keyShape->GetStorageShape().GetDim(0);
    uint32_t keyHeadNum = keyShape->GetStorageShape().GetDim(1);
    uint32_t keyD1 = keyShape->GetStorageShape().GetDim(2);
    uint32_t keyBlockSize = keyShape->GetStorageShape().GetDim(3);
    uint32_t keyD0 = keyShape->GetStorageShape().GetDim(4);

    uint32_t valueBlockNum = valueShape->GetStorageShape().GetDim(0);
    uint32_t valueHeadNum = valueShape->GetStorageShape().GetDim(1);
    uint32_t valueD1 = valueShape->GetStorageShape().GetDim(2);
    uint32_t valueBlockSize = valueShape->GetStorageShape().GetDim(3);
    uint32_t valueD0 = valueShape->GetStorageShape().GetDim(4);

    OP_CHECK_IF((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) || (keyD1 != valueD1) ||
                    (keyBlockSize != valueBlockSize) || (keyD0 != valueD0),
                OPS_REPORT_VECTOR_INNER_ERR(
                    faInfo.opName,
                    "The dim num of key and value are inconsistent when PA enable. key shape [%u, %u, %u, %u, %u],"
                    " value shape [%u, %u, %u, %u, %u].",
                    keyBlockNum, keyHeadNum, keyD1, keyBlockSize, keyD0, valueBlockNum, valueHeadNum, valueD1,
                    valueBlockSize, valueD0),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyD0 != 16 || valueD0 != 16,
                OP_LOGE(faInfo.opName,
                        "The D0 of key is %u, the D0 of value is %u, "
                        "they both should be 16",
                        keyD0, valueD0),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckPAKeyValue(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE(faInfo.opName,
                        "The dim num of key should be consistent with the dim num of value,"
                        "but current dim num of key is %u, dim num of value is %u.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);

    if (keyDimNum == 3) {
        return CheckPAKeyValueDim3(faInfo);
    } else if (keyDimNum == 4) {
        return CheckPAKeyValueDim4(faInfo);
    } else if (keyDimNum == 5) {
        return CheckPAKeyValueDim5(faInfo);
    } else {
        OP_LOGE(faInfo.opName, "The dim num of key only support [3, 4, 5] in PA mode, but current is %u.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus ShapeChecker::CheckAxis(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.bSize >= B_LIMIT || faInfo.bSize <= 0,
                OP_LOGE(faInfo.opName, "The axis B only support (0, %u), the current is %u.", B_LIMIT, faInfo.bSize),
                return ge::GRAPH_FAILED);
    // qTSize and kTSize are only valid for TND/NTD layout
    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(faInfo.qTSize <= 0,
                    OP_LOGE(faInfo.opName, "The axis Q_T must be greater than 0, the current is %u.", faInfo.qTSize),
                    return ge::GRAPH_FAILED);
    }
    if (faInfo.kvLayout == FaLayout::TND) {
        OP_CHECK_IF(faInfo.kTSize <= 0,
                    OP_LOGE(faInfo.opName, "The axis KV_T must be greater than 0, the current is %u.", faInfo.kTSize),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(faInfo.n1Size <= 0,
                OP_LOGE(faInfo.opName, "The axis Q_N must be greater than 0, the current is %ld.", faInfo.n1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.n2Size <= 0,
                OP_LOGE(faInfo.opName, "The axis KV_N must be greater than 0, the current is %ld.", faInfo.n2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s1Size <= 0,
                OP_LOGE(faInfo.opName, "The axis Q_S must be greater than 0, the current is %u.", faInfo.s1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s2Size <= 0,
                OP_LOGE(faInfo.opName, "The axis KV_S must be greater than 0, the current is %ld.", faInfo.s2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.qkHeadDim > D_LIMIT || faInfo.qkHeadDim < 16,
                OP_LOGE(faInfo.opName, "The axis D of query and key only support [16, %u], the current is %u.", D_LIMIT,
                        faInfo.qkHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.vHeadDim > D_LIMIT || faInfo.vHeadDim < 16,
                OP_LOGE(faInfo.opName, "The axis D of value only support [16, %u], the current is %u.", D_LIMIT,
                        faInfo.vHeadDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        faInfo.qkHeadDim % 16 != 0,
        OP_LOGE(faInfo.opName, "The axis D of query and key must be 16 aligned, the current is %u.", faInfo.qkHeadDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.vHeadDim % 16 != 0,
                OP_LOGE(faInfo.opName, "The axis D of value must be 16 aligned, the current is %u.", faInfo.vHeadDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckQueryOutConsistencySemantic(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *attentionOutShape = faInfo.opParamInfo.attnOut.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    size_t dimNumQ = faInfo.opParamInfo.query.shape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(faInfo.bSize != attentionOutShape->GetStorageShape().GetDim(0),
                OP_LOGE(faInfo.opName, "Batch size mismatch between query(%u) and attentionOut(%ld).", faInfo.bSize,
                        attentionOutShape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.n1Size !=
                    attentionOutShape->GetStorageShape().GetDim((faInfo.outLayout == FaLayout::BNSD) ? 1 : 2),
                OP_LOGE(faInfo.opName, "Head number mismatch."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.s1Size !=
                    attentionOutShape->GetStorageShape().GetDim((faInfo.outLayout == FaLayout::BNSD) ? 2 : 1),
                OP_LOGE(faInfo.opName, "Sequence length mismatch."), return ge::GRAPH_FAILED);

    int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(dimNumQ - 1);
    int64_t tmpOutDim = attentionOutShape->GetStorageShape().GetDim(dimNumOut - 1);
    OP_CHECK_IF(tmpOutDim != tmpValueDim,
                OPS_REPORT_VECTOR_INNER_ERR(
                    faInfo.opName, "The D dimension of attentionOut(%ld) should be consistent with D of value(%ld).",
                    tmpOutDim, tmpValueDim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckQueryOutConsistencyIndex(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *queryShape = faInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = faInfo.opParamInfo.attnOut.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    size_t dimNumQ = queryShape->GetStorageShape().GetDimNum();

    for (uint32_t queryDimIdx = 0; queryDimIdx < dimNumQ; ++queryDimIdx) {
        int64_t tmpQueryDim = queryShape->GetStorageShape().GetDim(queryDimIdx);
        int64_t tmpOutDim = attentionOutShape->GetStorageShape().GetDim(queryDimIdx);

        if (queryDimIdx == dimNumQ - 1) {
            int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(queryDimIdx);
            OP_CHECK_IF(tmpOutDim != tmpValueDim,
                        OPS_REPORT_VECTOR_INNER_ERR(
                            faInfo.opName,
                            "The D dimension of attentionOut(%ld) should be consistent with D of value(%ld).",
                            tmpOutDim, tmpValueDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                tmpQueryDim != tmpOutDim,
                OPS_REPORT_VECTOR_INNER_ERR(
                    faInfo.opName,
                    "The dimension %u of query(%ld) should be consistent with dimension %u of attentionOut(%ld).",
                    queryDimIdx, tmpQueryDim, queryDimIdx, tmpOutDim),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckQueryOutConsistency(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *queryShape = faInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = faInfo.opParamInfo.attnOut.shape;
    size_t dimNumQ = queryShape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(dimNumQ != dimNumOut,
                OP_LOGE(faInfo.opName,
                        "The dim num of query should be consistent with the dim num of attentionOut,"
                        "but current dim num of query is %zu, dim num of attentionOut is %zu.",
                        dimNumQ, dimNumOut),
                return ge::GRAPH_FAILED);

    if (faInfo.qLayout != faInfo.outLayout) {
        return CheckQueryOutConsistencySemantic(faInfo);
    }
    return CheckQueryOutConsistencyIndex(faInfo);
}

ge::graphStatus ShapeChecker::CheckNonQuantDataType(const FaTilingInfo &faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.value.desc, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.attnOut.desc, ATTEN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

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

ge::graphStatus ShapeChecker::CheckSoftmaxScale(const FaTilingInfo &faInfo)
{
    auto &softmaxScale = faInfo.opParamInfo.softmaxScale;
    if (softmaxScale != nullptr) {
        double scaleValue = *softmaxScale;
        const double epsilon = 1e-9;
        OP_CHECK_IF(std::fabs(scaleValue) < epsilon, OP_LOGE(faInfo.opName, "softmax_scale cannot be zero."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckDeterministic(const FaTilingInfo &faInfo)
{
    auto &deterministic = faInfo.opParamInfo.deterministic;
    if (deterministic != nullptr) {
        int64_t deterministicValue = *deterministic;
        OP_CHECK_IF(deterministicValue != 0 && deterministicValue != 1,
                    OP_LOGE(faInfo.opName, "deterministic(%ld) only supports 0 or 1.", deterministicValue),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckShapeLayoutQuery(const FaTilingInfo &faInfo)
{
    uint32_t queryDimNum = faInfo.opParamInfo.query.shape->GetStorageShape().GetDimNum();

    switch (faInfo.qLayout) {
        case FaLayout::BNSD:
            OP_CHECK_IF(
                queryDimNum != 4,
                OP_LOGE(faInfo.opName, "query dim num must be 4 when layout_q is BNSD, but got %u.", queryDimNum),
                return ge::GRAPH_FAILED);
            break;
        case FaLayout::BSND:
            OP_CHECK_IF(
                queryDimNum != 4,
                OP_LOGE(faInfo.opName, "query dim num must be 4 when layout_q is BSND, but got %u.", queryDimNum),
                return ge::GRAPH_FAILED);
            break;
        case FaLayout::TND:
            OP_CHECK_IF(
                queryDimNum != 3,
                OP_LOGE(faInfo.opName, "query dim num must be 3 when layout_q is TND, but got %u.", queryDimNum),
                return ge::GRAPH_FAILED);
            break;
        default:
            OP_LOGE(faInfo.opName, "layout_q %d is not supported for shape check.", static_cast<int>(faInfo.qLayout));
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckShapeLayoutKV(const FaTilingInfo &faInfo)
{
    uint32_t keyDimNum = faInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    uint32_t valueDimNum = faInfo.opParamInfo.value.shape->GetStorageShape().GetDimNum();

    switch (faInfo.kvLayout) {
        case FaLayout::BNSD:
        case FaLayout::BSND:
            OP_CHECK_IF(keyDimNum != 4 || valueDimNum != 4,
                        OP_LOGE(faInfo.opName,
                                "key/value dim num must be 4 when layout_kv is BNSD/BSND, but got key:%u, value:%u.",
                                keyDimNum, valueDimNum),
                        return ge::GRAPH_FAILED);
            break;
        case FaLayout::TND:
            OP_CHECK_IF(keyDimNum != 3 || valueDimNum != 3,
                        OP_LOGE(faInfo.opName,
                                "key/value dim num must be 3 when layout_kv is TND, but got key:%u, value:%u.",
                                keyDimNum, valueDimNum),
                        return ge::GRAPH_FAILED);
            break;
        case FaLayout::PA_BBND:
        case FaLayout::PA_BNBD:
            OP_CHECK_IF(
                keyDimNum != 4 || valueDimNum != 4,
                OP_LOGE(faInfo.opName,
                        "key/value dim num must be 4 when layout_kv is PA_BBND/PA_BNBD, but got key:%u, value:%u.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);
            break;
        case FaLayout::PA_Nz:
            OP_CHECK_IF(keyDimNum != 5 || valueDimNum != 5,
                        OP_LOGE(faInfo.opName,
                                "key/value dim num must be 5 when layout_kv is PA_Nz, but got key:%u, value:%u.",
                                keyDimNum, valueDimNum),
                        return ge::GRAPH_FAILED);
            break;
        default:
            OP_LOGE(faInfo.opName, "layout_kv %d is not supported for shape check.", static_cast<int>(faInfo.kvLayout));
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckShapeLayout(const FaTilingInfo &faInfo)
{
    if (CheckShapeLayoutQuery(faInfo) != ge::GRAPH_SUCCESS || CheckShapeLayoutKV(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
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

ge::graphStatus ShapeChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    (void)faInfo;
    return ge::GRAPH_SUCCESS;
}

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

ge::graphStatus ShapeChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    if (faInfo.pageAttentionFlag) {
        if (CheckPAKeyValue(faInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (CheckNonQuantDataType(faInfo) != ge::GRAPH_SUCCESS || CheckAxis(faInfo) != ge::GRAPH_SUCCESS ||
        CheckQueryOutConsistency(faInfo) != ge::GRAPH_SUCCESS || CheckNonQuantHeadNum(faInfo) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistency(faInfo) != ge::GRAPH_SUCCESS || CheckSoftmaxScale(faInfo) != ge::GRAPH_SUCCESS ||
        CheckDeterministic(faInfo) != ge::GRAPH_SUCCESS || CheckShapeLayout(faInfo) != ge::GRAPH_SUCCESS ||
        CheckShapeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void ShapeChecker::SetFaShapeCompare(const FaTilingInfo &faInfo)
{
    queryShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.query.shape->GetStorageShape(),
                                                            faInfo.qLayout, QUERY_NAME, faInfo.opName);
    keyShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.key.shape->GetStorageShape(),
                                                          faInfo.kvLayout, KEY_NAME, faInfo.opName);
    valueShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.value.shape->GetStorageShape(),
                                                            faInfo.kvLayout, VALUE_NAME, faInfo.opName);
    attnOutShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.attnOut.shape->GetStorageShape(),
                                                              faInfo.outLayout, ATTEN_OUT_NAME, faInfo.opName);
    if (faInfo.softmaxLseFlag) {
        lseOutShapeCmp_ = std::make_shared<FaTilingShapeCompare>(faInfo.opParamInfo.lseOut.shape->GetStorageShape(),
                                                                 faInfo.outLayout, SOFTMAX_LSE_NAME, faInfo.opName);
    }
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

ge::graphStatus ShapeChecker::CheckKVShape(const FaTilingInfo &faInfo) const
{
    FaTilingShapeCompareParam keyShapeParams;
    keyShapeParams.B = static_cast<int64_t>(faInfo.bSize);
    keyShapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    keyShapeParams.S = static_cast<int64_t>(faInfo.s2Size);
    keyShapeParams.D = static_cast<int64_t>(faInfo.qkHeadDim);
    keyShapeParams.T = static_cast<int64_t>(faInfo.kTSize);
    keyShapeParams.Bn = static_cast<int64_t>(faInfo.totalBlockNum);
    keyShapeParams.Bs = static_cast<int64_t>(faInfo.blockSize);
    keyShapeParams.D0 = 16;

    if (keyShapeCmp_->CompareShape(keyShapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FaTilingShapeCompareParam valueShapeParams = keyShapeParams;
    valueShapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    if (valueShapeCmp_->CompareShape(valueShapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
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

ge::graphStatus ShapeChecker::CheckLseOutShape(const FaTilingInfo &faInfo) const
{
    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    if (lseOutShapeCmp_ == nullptr) {
        OP_LOGE(faInfo.opName, "lseOutShapeCmp is nullptr when softmaxLseFlag is true.");
        return ge::GRAPH_FAILED;
    }

    FaTilingShapeCompareParam shapeParams;
    if (faInfo.outLayout == FaLayout::TND) {
        shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
        shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
        shapeParams.CONST = 1;
    } else {
        shapeParams.B = static_cast<int64_t>(faInfo.bSize);
        shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
        shapeParams.S1 = static_cast<int64_t>(faInfo.s1Size);
        shapeParams.CONST = 1;
    }
    return lseOutShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus ShapeChecker::CheckShapeConsistency(const FaTilingInfo &faInfo)
{
    SetFaShapeCompare(faInfo);
    if (ge::GRAPH_SUCCESS != CheckQueryShape(faInfo) || ge::GRAPH_SUCCESS != CheckKVShape(faInfo) ||
        ge::GRAPH_SUCCESS != CheckAttnOutShape(faInfo) || ge::GRAPH_SUCCESS != CheckLseOutShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
