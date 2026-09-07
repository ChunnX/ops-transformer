/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_ONNX_ATTRIBUTE_UTILS_H
#define OPS_TRANSFORMER_ONNX_ATTRIBUTE_UTILS_H

#include <cstdint>
#include <exception>
#include <limits>
#include <string>

#include "graph/operator.h"
#include "register/register.h"
#include "nlohmann/json.hpp"

namespace domi {
// Read the JSON attributes supplied by the ONNX operator-based parser without
// depending on generated protobuf classes. Keep the caller's default if absent.
inline Status ParseOnnxIntAttribute(const ge::Operator& op_src, const char* name, int& value)
{
    ge::AscendString attributes;
    if (op_src.GetAttr("attribute", attributes) != ge::GRAPH_SUCCESS) {
        return SUCCESS;
    }
    if (attributes.GetString() == nullptr) {
        return FAILED;
    }
    try {
        const auto root = nlohmann::json::parse(attributes.GetString());
        if (!root.is_object()) {
            return FAILED;
        }
        const auto attrs = root.find("attribute");
        if (attrs == root.end()) {
            return SUCCESS;
        }
        if (!attrs->is_array()) {
            return FAILED;
        }
        for (const auto& attr : *attrs) {
            if (!attr.is_object()) {
                return FAILED;
            }
            const auto attr_name = attr.find("name");
            const auto type = attr.find("type");
            // ONNX AttributeProto::INT is 2; JSON may use its enum name.
            if (attr_name == attr.end() || *attr_name != name || type == attr.end() ||
                !(*type == 2 || *type == "INT")) {
                continue;
            }
            const auto integer = attr.find("i");
            int64_t parsed = 0; // An omitted protobuf integer field defaults to zero.
            if (integer != attr.end()) {
                if (integer->is_string()) {
                    const auto text = integer->get<std::string>();
                    const auto first_digit = !text.empty() && text[0] == '-' ? 1U : 0U;
                    if (text.size() == first_digit || text.find_first_not_of("0123456789", first_digit) !=
                        std::string::npos) {
                        return FAILED;
                    }
                    parsed = std::stoll(text);
                } else if (integer->is_number_unsigned()) {
                    const auto number = integer->get<uint64_t>();
                    if (number > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                        return FAILED;
                    }
                    parsed = static_cast<int64_t>(number);
                } else if (integer->is_number_integer()) {
                    parsed = integer->get<int64_t>();
                } else {
                    return FAILED;
                }
            }
            if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
                return FAILED;
            }
            value = static_cast<int>(parsed);
        }
    } catch (const std::exception&) {
        return FAILED;
    }
    return SUCCESS;
}
} // namespace domi

#endif // OPS_TRANSFORMER_ONNX_ATTRIBUTE_UTILS_H
