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
 * \file paged_attention_checker.cpp
 * \brief Implementation of PagedAttentionChecker for Paged Attention参数组.
 */

#include "paged_attention_checker.h"
#include "log/log.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;

ge::graphStatus PagedAttentionChecker::CheckOriBlockTable(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.oriBlockTable;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, ORI_BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, ORI_BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, ORI_BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_TWO};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, ORI_BLOCK_TABLE_NAME);
}

ge::graphStatus PagedAttentionChecker::CheckCmpBlockTable(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.cmpBlockTable;
    if (t.tensor == nullptr)
        return ge::GRAPH_SUCCESS;
    if (CheckShapeSizeNonZero(t.tensor, CMP_BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeSupport(t.desc, CMP_BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, CMP_BLOCK_TABLE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_TWO};
    return CheckDimNumSupport(&t.tensor->GetShape(), expectDims, CMP_BLOCK_TABLE_NAME);
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    if (CheckOriBlockTable(info) != ge::GRAPH_SUCCESS || CheckCmpBlockTable(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    if (info.kvLayout == QSMLALayout::PA_BBND) {
        OP_CHECK_IF(info.opParamInfo.oriBlockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "ori_block_table", "ori_block_table must not be empty when layout_kv is PA_BBND"),
                    return ge::GRAPH_FAILED);
        if (info.opParamInfo.cmpKv.tensor != nullptr) {
            OP_CHECK_IF(info.opParamInfo.cmpBlockTable.tensor == nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            info.opName, "cmp_block_table",
                            "cmp_block_table must not be empty when layout_kv is PA_BBND and cmp_kv is provided"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(info.opParamInfo.cmpBlockTable.tensor != nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            info.opName, "cmp_block_table", "cmp_block_table is not supported when cmp_kv is None"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(info.opParamInfo.oriBlockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "ori_block_table", "ori_block_table should be null when layout_kv is not PA_BBND"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(info.opParamInfo.cmpBlockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        info.opName, "cmp_block_table", "cmp_block_table should be null when layout_kv is not PA_BBND"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    if (info.opParamInfo.oriBlockTable.tensor != nullptr) {
        OP_CHECK_IF(info.opParamInfo.oriBlockTable.tensor->GetStorageShape().GetDim(DIM_0) !=
                        static_cast<int64_t>(info.bSize),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        info.opName, "ori_block_table",
                        GetShapeStr(info.opParamInfo.oriBlockTable.tensor->GetStorageShape()).c_str(),
                        ("ori_block_table B should equal q B(" + to_string(info.bSize) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (info.opParamInfo.cmpBlockTable.tensor != nullptr) {
        OP_CHECK_IF(info.opParamInfo.cmpBlockTable.tensor->GetStorageShape().GetDim(DIM_0) !=
                        static_cast<int64_t>(info.bSize),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        info.opName, "cmp_block_table",
                        GetShapeStr(info.opParamInfo.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                        ("cmp_block_table B should equal q B(" + to_string(info.bSize) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
