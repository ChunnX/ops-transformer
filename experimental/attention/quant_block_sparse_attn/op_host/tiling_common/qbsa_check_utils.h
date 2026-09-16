/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_CHECK_UTILS_H_
#define QBSA_CHECK_UTILS_H_

#include <array>
#include <cstdint>
#include <string>
#include "qbsa_tiling_context.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling::qbsa_check {
constexpr const char *kCheckOpName = "QuantBlockSparseAttn";
template <size_t N>
std::string ShapeToString(const std::array<int64_t, N> &shape)
{
    std::string result = "[";
    for (size_t i = 0U; i < N; ++i) {
        if (i != 0U) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

template <size_t N>
bool IsShapeEqual(const gert::Shape &actualShape, const std::array<int64_t, N> &expectedShape)
{
    if (actualShape.GetDimNum() != N) {
        return false;
    }
    for (size_t i = 0U; i < N; ++i) {
        if (actualShape.GetDim(i) != expectedShape[i]) {
            return false;
        }
    }
    return true;
}

template <size_t N>
ge::graphStatus CheckInputShape(const QBSARequiredParaInfo &input, const char *inputName,
                                const std::array<int64_t, N> &expectedShape)
{
    if (input.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kCheckOpName, inputName, "nullptr", "Shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &actualShape = input.shape->GetStorageShape();
    if (!IsShapeEqual(actualShape, expectedShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kCheckOpName, inputName, Ops::Base::ToString(actualShape),
                                              "must be " + ShapeToString(expectedShape) + " for this feature");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputDtype(const QBSARequiredParaInfo &input, const char *inputName,
                                       ge::DataType expectedType, const char *expectedTypeName)
{
    if (input.desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kCheckOpName, inputName, "nullptr", "Desc is nullptr");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType actualType = input.desc->GetDataType();
    if (actualType != expectedType) {
        OP_LOGE_FOR_INVALID_DTYPE(kCheckOpName, inputName, ge::TypeUtils::DataTypeToSerialString(actualType),
                                  expectedTypeName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckKVDtypeConsistency(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    if (opParamInfo.key.desc == nullptr || opParamInfo.value.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const ge::DataType keyType = opParamInfo.key.desc->GetDataType();
    const ge::DataType valueType = opParamInfo.value.desc->GetDataType();
    if (keyType != valueType) {
        OP_LOGE_FOR_INVALID_DTYPE(
            kCheckOpName, "key/value",
            ge::TypeUtils::DataTypeToSerialString(keyType) + "/" + ge::TypeUtils::DataTypeToSerialString(valueType),
            "key and value must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckStrideDim(const gert::Stride *stride, const char *inputName, size_t dim, uint64_t expected,
                                      bool hasViewStride)
{
    if (!hasViewStride) {
        return ge::GRAPH_SUCCESS;
    }
    if (stride == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kCheckOpName, inputName, "nullptr",
                                              "This input requires explicit view strides");
        return ge::GRAPH_FAILED;
    }
    if (stride->GetDimNum() <= dim) {
        const std::string argName = std::string(inputName) + " stride";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kCheckOpName, argName.c_str(),
                                                 std::to_string(stride->GetDimNum()) + "D",
                                                 "at least " + std::to_string(dim + 1U) + " stride dimensions");
        return ge::GRAPH_FAILED;
    }
    const uint64_t actual = stride->GetStride(dim);
    if (actual != expected) {
        const std::string argName = std::string(inputName) + " stride[" + std::to_string(dim) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kCheckOpName, argName.c_str(), std::to_string(actual),
                                              "Must be " + std::to_string(expected));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling::qbsa_check
#endif // QBSA_CHECK_UTILS_H_
