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
 * \file qsmla_tiling_utils.cpp
 * \brief Free utility functions for QSMLA: DataType/Layout serialization and shape string.
 */

#include "qsmla_tiling_utils.h"
#include "log/log.h"

namespace optiling {

std::string QSMLADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    }
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantSparseFlashMla", "datatype",
                                             ("datatype " + std::to_string(type) + " not supported").c_str());
    return "UNDEFINED";
}

std::string QSMLALayoutToSerialString(QSMLALayout layout)
{
    switch (layout) {
        case QSMLALayout::BSND:
            return "BSND";
        case QSMLALayout::TND:
            return "TND";
        case QSMLALayout::PA_BBND:
            return "PA_BBND";
        default:
            return "UNKNOWN";
    }
}

std::string QSMLAGetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

} // namespace optiling
