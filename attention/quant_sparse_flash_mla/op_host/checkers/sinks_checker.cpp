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
 * \file sinks_checker.cpp
 * \brief Implementation of SinksChecker for Sinks参数组.
 */

#include "sinks_checker.h"
#include "log/log.h"

namespace optiling {

using std::string;
using std::to_string;
using std::vector;

ge::graphStatus SinksChecker::CheckSinglePara(const QSMLATilingInfo &info) const
{
    const auto &t = info.opParamInfo.sinks;
    OP_CHECK_IF(t.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "sinks", "sinks is required but got nullptr"),
                return ge::GRAPH_FAILED);

    if (CheckDtypeSupport(t.desc, SINKS_NAME) != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(t.desc, SINKS_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const vector<size_t> expectDims = {DIM_NUM_ONE};
    if (CheckDimNumSupport(&t.tensor->GetShape(), expectDims, SINKS_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinksChecker::CheckParaExistence(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.opParamInfo.sinks.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "sinks", "sinks is required in current version"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinksChecker::CheckCrossFeature(const QSMLATilingInfo &info) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinksChecker::CheckMultiParaConsistency(const QSMLATilingInfo &info) const
{
    OP_CHECK_IF(info.opParamInfo.sinks.tensor->GetShapeSize() != static_cast<int64_t>(info.n1Size),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    info.opName, "sinks", to_string(info.opParamInfo.sinks.tensor->GetShapeSize()).c_str(),
                    ("sinks shapeSize should equal q_head_num(" + to_string(info.n1Size) + ")").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
