/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file onnx_plugin_utils.h
 * \brief Protobuf-free helpers shared by ONNX plugin implementations.
 */

#ifndef MATH_COMMON_ONNX_PLUGIN_UTILS_H
#define MATH_COMMON_ONNX_PLUGIN_UTILS_H

#include <cstdint>
#include <string>
#include <vector>

#include "graph/operator.h"
#include "log/log.h"

namespace domi {
template <typename T>
inline std::string GetOpName(const T &op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        return "None";
    }
    return op_ascend_name.GetString();
}

template <typename T>
inline ge::Tensor Vec2Tensor(std::vector<T> &vals, const std::vector<int64_t> &dims, ge::DataType dtype,
                             ge::Format format = ge::FORMAT_ND)
{
    ge::Shape shape(dims);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t *>(vals.data()), vals.size() * sizeof(T));
    return tensor;
}

template <typename T>
inline ge::Tensor CreateScalar(T val, ge::DataType dtype, ge::Format format = ge::FORMAT_ND)
{
    std::vector<int64_t> dims_scalar = {};
    ge::Shape shape(dims_scalar);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t *>(&val), sizeof(T));
    return tensor;
}
} // namespace domi

#endif // MATH_COMMON_ONNX_PLUGIN_UTILS_H
