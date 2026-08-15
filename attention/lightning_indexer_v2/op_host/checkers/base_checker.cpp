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
 * \file base_checker.cpp
 * \brief Common helper implementation for LightningIndexerV2 family checkers
 */

#include <sstream>
#include "log/log.h"
#include "base_checker.h"

namespace optiling {
namespace lightning_indexer_v2 {

ge::graphStatus LIBaseChecker::CheckRequiredTensor(const CheckerTensor &tensor, const char *name,
                                                   const IndexerCheckerInfo &info) const
{
    OP_CHECK_IF(!tensor.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "The tensor desc and shape cannot be null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIBaseChecker::CheckOptionalTensor(const CheckerTensor &tensor, const char *name,
                                                   const IndexerCheckerInfo &info) const
{
    OP_CHECK_IF(tensor.present && tensor.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "The tensor desc cannot be null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIBaseChecker::CheckDtype(const CheckerTensor &tensor, const char *name,
                                          std::initializer_list<ge::DataType> expected,
                                          const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(tensor.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "The tensor desc cannot be null"),
                return ge::GRAPH_FAILED);
    auto dtype = tensor.desc->GetDataType();
    std::string expectedDtypes;
    for (auto expectedDtype : expected) {
        if (!expectedDtypes.empty()) {
            expectedDtypes += ", ";
        }
        expectedDtypes += DataTypeToString(expectedDtype);
    }
    OP_CHECK_IF(std::find(expected.begin(), expected.end(), dtype) == expected.end(),
                OP_LOGE_FOR_INVALID_DTYPE(info.opName, name, DataTypeToString(dtype).c_str(), expectedDtypes.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIBaseChecker::CheckFormat(const CheckerTensor &tensor, const char *name,
                                           const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(tensor.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "The tensor desc cannot be null"),
                return ge::GRAPH_FAILED);
    const auto format = tensor.desc->GetOriginFormat();
    const std::string actualFormat = std::to_string(static_cast<int32_t>(format));
    OP_CHECK_IF(format != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(info.opName, name, actualFormat.c_str(), "ND"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIBaseChecker::CheckRank(const CheckerTensor &tensor, const char *name, size_t expected,
                                         const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!tensor.shapeAvailable || tensor.shape.size() != expected,
                OP_LOGE_FOR_INVALID_SHAPEDIM(info.opName, name, std::to_string(tensor.shape.size()).c_str(),
                                             std::to_string(expected).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIBaseChecker::CheckShape(const CheckerTensor &tensor, const char *name,
                                          const std::vector<int64_t> &expected,
                                          const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!tensor.shapeAvailable || tensor.shape != expected,
                OP_LOGE_FOR_INVALID_SHAPE(info.opName, name, ShapeToString(tensor.shape).c_str(),
                                          ShapeToString(expected).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIBaseChecker::CheckNoZeroDim(const CheckerTensor &tensor, const char *name,
                                              const IndexerCheckerInfo &info) const
{
    if (!tensor.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(std::any_of(tensor.shape.begin(), tensor.shape.end(), [](int64_t dim) { return dim <= 0; }),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, name, ShapeToString(tensor.shape).c_str(),
                                                      "The tensor cannot contain a dimension less than or equal to 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

std::string LIBaseChecker::DataTypeToString(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return "FLOAT";
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BFLOAT16";
        case ge::DT_INT8:
            return "INT8";
        case ge::DT_INT32:
            return "INT32";
        case ge::DT_FLOAT8_E4M3FN:
            return "FLOAT8_E4M3FN";
        case ge::DT_FLOAT8_E8M0:
            return "FLOAT8_E8M0";
        case ge::DT_HIFLOAT8:
            return "HIFLOAT8";
        case ge::DT_FLOAT4_E2M1:
            return "FLOAT4_E2M1";
        default:
            return std::to_string(static_cast<int32_t>(dtype));
    }
}

std::string LIBaseChecker::ShapeToString(const std::vector<int64_t> &shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << shape[i];
    }
    stream << "]";
    return stream.str();
}

std::string LIBaseChecker::LayoutToString(IndexerLayout layout)
{
    switch (layout) {
        case IndexerLayout::BSND:
            return "BSND";
        case IndexerLayout::TND:
            return "TND";
        case IndexerLayout::PA_BBND:
            return "PA_BBND";
        default:
            return "UNKNOWN";
    }
}

} // namespace lightning_indexer_v2
} // namespace optiling
