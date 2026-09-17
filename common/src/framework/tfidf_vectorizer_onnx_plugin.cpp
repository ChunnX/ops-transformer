/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "graph/operator.h"
#include "register/register.h"

namespace domi {
// ONNX AttributeProto type values used by this plugin.
constexpr int ATTRIBUTE_TYPE_INT = 2;
constexpr int ATTRIBUTE_TYPE_STRING = 3;
constexpr int ATTRIBUTE_TYPE_FLOATS = 6;
constexpr int ATTRIBUTE_TYPE_INTS = 7;
constexpr int ATTRIBUTE_TYPE_STRINGS = 8;

static bool GetInt64(const nlohmann::json &value, int64_t &parsed_value)
{
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() &&
         value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
        return false;
    }
    parsed_value = value.get<int64_t>();
    return true;
}

static Status ParseParamsTfIdfVectorizer(const ge::Operator &op_src, ge::Operator &op_dest)
{
    ge::AscendString attributes;
    // ONNX repeated AttributeProto messages are exposed as JSON in "attribute".
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
        if (!root.contains("attribute")) {
            return SUCCESS;
        }
        const auto &attrs = root.at("attribute");
        if (!attrs.is_array()) {
            return FAILED;
        }
        for (const auto &attr : attrs) {
            if (!attr.is_object()) {
                return FAILED;
            }
            const std::string name = attr.value("name", "");
            const int type = attr.value("type", 0);
            if (type == ATTRIBUTE_TYPE_INT) {
                // An omitted protobuf integer field has the default value zero.
                int64_t parsed_value = 0;
                if (attr.contains("i") && !GetInt64(attr.at("i"), parsed_value)) {
                    return FAILED;
                }
                if (name == "max_gram_length" || name == "max_skip_count" || name == "min_gram_length") {
                    op_dest.SetAttr(name.c_str(), parsed_value);
                }
            } else if (name == "mode" && type == ATTRIBUTE_TYPE_STRING) {
                // An omitted protobuf string field is empty.
                if (!attr.contains("s")) {
                    op_dest.SetAttr(name.c_str(), std::string());
                } else if (!attr.at("s").is_string()) {
                    return FAILED;
                } else {
                    op_dest.SetAttr(name.c_str(), attr.at("s").get<std::string>());
                }
            } else if (name == "weights" && type == ATTRIBUTE_TYPE_FLOATS) {
                if (!attr.contains("floats")) {
                    op_dest.SetAttr(name.c_str(), std::vector<float>{});
                } else if (!attr.at("floats").is_array()) {
                    return FAILED;
                } else {
                    op_dest.SetAttr(name.c_str(), attr.at("floats").get<std::vector<float>>());
                }
            } else if (type == ATTRIBUTE_TYPE_INTS) {
                if (name == "ngram_counts" || name == "ngram_indexes" || name == "pool_int64s") {
                    if (!attr.contains("ints")) {
                        op_dest.SetAttr(name.c_str(), std::vector<int64_t>{});
                    } else if (!attr.at("ints").is_array()) {
                        return FAILED;
                    } else {
                        std::vector<int64_t> values;
                        values.reserve(attr.at("ints").size());
                        for (const auto &element : attr.at("ints")) {
                            int64_t parsed_value = 0;
                            if (!GetInt64(element, parsed_value)) {
                                return FAILED;
                            }
                            values.push_back(parsed_value);
                        }
                        op_dest.SetAttr(name.c_str(), values);
                    }
                }
            } else if (name == "pool_strings" && type == ATTRIBUTE_TYPE_STRINGS) {
                if (!attr.contains("strings")) {
                    op_dest.SetAttr(name.c_str(), std::vector<std::string>{});
                } else if (!attr.at("strings").is_array()) {
                    return FAILED;
                } else {
                    op_dest.SetAttr(name.c_str(), attr.at("strings").get<std::vector<std::string>>());
                }
            }
        }
    } catch (const nlohmann::json::exception &) {
        return FAILED;
    }
    return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("TfIdfVectorizer")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::9::TfIdfVectorizer"), ge::AscendString("ai.onnx::10::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::11::TfIdfVectorizer"), ge::AscendString("ai.onnx::12::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::13::TfIdfVectorizer"), ge::AscendString("ai.onnx::14::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::15::TfIdfVectorizer"), ge::AscendString("ai.onnx::16::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::17::TfIdfVectorizer"), ge::AscendString("ai.onnx::18::TfIdfVectorizer")})
    .ParseParamsByOperatorFn(ParseParamsTfIdfVectorizer)
    .ImplyType(ImplyType::TVM);
} // namespace domi
