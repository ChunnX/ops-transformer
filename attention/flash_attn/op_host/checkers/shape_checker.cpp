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

ge::graphStatus ShapeChecker::CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                               std::map<std::string, std::vector<ge::DataType>> dataMap)
{
    if (desc != nullptr) {
        const auto &it = dataMap.find(name);
        OP_CHECK_IF(it == dataMap.end(),
                    OP_LOGE("FA", "%s datatype support list should be specify in map", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) ==
                        expectDtypeList.end(),
                    OP_LOGE("FA", "%s datatype not supported", name.c_str()), return ge::GRAPH_FAILED);
    }

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
                        "but current dim num of key is %zu, dim num of value is %zu.",
                        keyDimNum, valueDimNum),
                return ge::GRAPH_FAILED);

    if (keyDimNum == 3) {
        uint32_t keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        uint32_t keyBlockSize = keyShape->GetStorageShape().GetDim(1);
        uint32_t keyH = keyShape->GetStorageShape().GetDim(2);

        uint32_t valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        uint32_t valueBlockSize = valueShape->GetStorageShape().GetDim(1);
        uint32_t valueH = valueShape->GetStorageShape().GetDim(2);

        OP_CHECK_IF((keyBlockNum != valueBlockNum) || (keyBlockSize != valueBlockSize) || (keyH != valueH),
                    OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                                "The dim num of key and value are inconsistent when PA enable. key "
                                                "shape [%ld,%ld,%ld], value shape [%ld,%ld,%ld].",
                                                keyBlockNum, keyBlockSize, keyH, valueBlockNum, valueBlockSize, valueH),
                    return ge::GRAPH_FAILED);
    } else if (keyDimNum == 4) {
        uint32_t keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        uint32_t keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        uint32_t keyBlockSize = keyShape->GetStorageShape().GetDim(2);
        uint32_t keyHeadDim = keyShape->GetStorageShape().GetDim(3);

        uint32_t valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        uint32_t valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        uint32_t valueBlockSize = valueShape->GetStorageShape().GetDim(2);
        uint32_t valueHeadDim = valueShape->GetStorageShape().GetDim(3);

        OP_CHECK_IF((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) ||
                        (keyBlockSize != valueBlockSize) || (keyHeadDim != valueHeadDim),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        faInfo.opName,
                        "the dim num of key and value are inconsistent when PA enable. key shape [%ld,%ld,%ld,%ld],"
                        " value shape [%ld,%ld,%ld,%ld].",
                        keyBlockNum, keyHeadNum, keyBlockSize, keyHeadDim, valueBlockNum, valueHeadNum, valueBlockSize,
                        valueHeadDim),
                    return ge::GRAPH_FAILED);
    } else if (keyDimNum == 5) {
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

        OP_CHECK_IF(
            (keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) || (keyD1 != valueD1) ||
                (keyBlockSize != valueBlockSize) || (keyD0 != valueD0),
            OPS_REPORT_VECTOR_INNER_ERR(
                faInfo.opName,
                "The dim num of key and value are inconsistent when PA enable. key shape [%ld, %ld, %ld, %ld, %ld],"
                " value shape [%ld, %ld, %ld, %ld, %ld].",
                keyBlockNum, keyHeadNum, keyD1, keyBlockSize, keyD0, valueBlockNum, valueHeadNum, valueD1,
                valueBlockSize, valueD0),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(keyD0 != 16 || valueD0 != 16,
                    OP_LOGE(faInfo.opName,
                            "The D0 of key is %u, the D0 of value is %u, "
                            "they both should be 16",
                            keyD0, valueD0),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(faInfo.opName, "The dim num of key only support [3, 4, 5] in PA mode, but current is %zu.", keyDimNum);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckAxis(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.bSize > B_LIMIT || faInfo.bSize <= 0,
                OP_LOGE(faInfo.opName, "The axis B only support (0, %u], the current is %u.", B_LIMIT, faInfo.bSize),
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
                OP_LOGE(faInfo.opName, "The axis Q_N must be greater than 0, the current is %u.", faInfo.n1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.n2Size <= 0,
                OP_LOGE(faInfo.opName, "The axis KV_N must be greater than 0, the current is %u.", faInfo.n2Size),
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

ge::graphStatus ShapeChecker::CheckQueryOutConsistency(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *queryShape = faInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = faInfo.opParamInfo.attnOut.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    size_t dimNumQ = queryShape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(dimNumQ != dimNumOut,
                OP_LOGE(faInfo.opName,
                        "The dim num of query should be consistent with the dim num of attentionOut,"
                        "but current dim num of query is %zu, dim num of attentionOut is %zu.",
                        dimNumQ, dimNumOut),
                return ge::GRAPH_FAILED);

    // When layout_q != layout_out, skip shape dimension consistency check at index level
    // because different layouts have different axis orders (e.g., BNSD vs BSND)
    if (faInfo.qLayout != faInfo.outLayout) {
        // Only check semantic dimensions: B, N, S, D should match
        // B: batch size
        OP_CHECK_IF(faInfo.bSize != attentionOutShape->GetStorageShape().GetDim(0),
                    OP_LOGE(faInfo.opName, "Batch size mismatch between query(%u) and attentionOut(%ld).", faInfo.bSize,
                            attentionOutShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);
        // N: number of heads
        OP_CHECK_IF(faInfo.n1Size !=
                        attentionOutShape->GetStorageShape().GetDim((faInfo.outLayout == FaLayout::BNSD) ? 1 : 2),
                    OP_LOGE(faInfo.opName, "Head number mismatch."), return ge::GRAPH_FAILED);
        // S: sequence length
        OP_CHECK_IF(faInfo.s1Size !=
                        attentionOutShape->GetStorageShape().GetDim((faInfo.outLayout == FaLayout::BNSD) ? 2 : 1),
                    OP_LOGE(faInfo.opName, "Sequence length mismatch."), return ge::GRAPH_FAILED);
        // D: head dimension - should match value's D
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(dimNumQ - 1);
        int64_t tmpOutDim = attentionOutShape->GetStorageShape().GetDim(dimNumOut - 1);
        OP_CHECK_IF(tmpOutDim != tmpValueDim,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        faInfo.opName,
                        "The D dimension of attentionOut(%ld) should be consistent with D of value(%ld).", tmpOutDim,
                        tmpValueDim),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // Same layout: check dimension by index
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

ge::graphStatus ShapeChecker::CheckNonQuantDataType(const FaTilingInfo &faInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> NO_QUANT_MAP = {
        {"query", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value", {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut", {ge::DT_FLOAT16, ge::DT_BF16}},
    };
    if (ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.query.desc, "query", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.key.desc, "key", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.value.desc, "value", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(faInfo.opParamInfo.attnOut.desc, "attentionOut", NO_QUANT_MAP)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckNonQuantHeadNum(const FaTilingInfo &faInfo)
{
    if ((faInfo.n1Size < 0) || (faInfo.n2Size < 0)) {
        OP_LOGE(faInfo.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", faInfo.n1Size, faInfo.n2Size);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((faInfo.n2Size != 0) && (faInfo.n1Size % faInfo.n2Size != 0),
                OPS_REPORT_VECTOR_INNER_ERR(faInfo.opName,
                                            "numHeads(%d) should be an integer multiple of numKeyValueHeads(%d)!",
                                            faInfo.n1Size, faInfo.n2Size),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
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
        CheckQueryOutConsistency(faInfo) != ge::GRAPH_SUCCESS || CheckNonQuantHeadNum(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
