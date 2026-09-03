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
 * \file plugin_util.h
 * \brief Common utilities for framework plugins without protobuf dependency.
 */

#ifndef OPS_TRANSFORMER_COMMON_INCLUDE_FRAMEWORK_PLUGIN_UTIL_H
#define OPS_TRANSFORMER_COMMON_INCLUDE_FRAMEWORK_PLUGIN_UTIL_H

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "graph/ascend_string.h"
#include "graph/graph.h"
#include "log/log.h"
#include "register/register.h"

namespace domi {

inline bool StrToFloat(const std::string& str, float& value)
{
    if (str.empty()) {
        return false;
    }
    errno = 0;
    char* end_ptr = nullptr;
    const float parsed = std::strtof(str.c_str(), &end_ptr);
    if (end_ptr == str.c_str() || *end_ptr != '\0' || errno == ERANGE) {
        return false;
    }
    value = parsed;
    return true;
}

template <typename T>
inline std::string GetOpName(const T& op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        return "None";
    }
    return op_ascend_name.GetString();
}

template <typename T>
inline ge::Tensor Vec2Tensor(std::vector<T>& vals, const std::vector<int64_t>& dims, ge::DataType dtype,
                             ge::Format format = ge::FORMAT_ND)
{
    ge::Shape shape(dims);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
    return tensor;
}

template <typename T>
inline ge::Tensor CreateScalar(T val, ge::DataType dtype, ge::Format format = ge::FORMAT_ND)
{
    std::vector<int64_t> dims_scalar = {};
    ge::Shape shape(dims_scalar);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(&val), sizeof(T));
    return tensor;
}

} // namespace domi

#endif // OPS_TRANSFORMER_COMMON_INCLUDE_FRAMEWORK_PLUGIN_UTIL_H
