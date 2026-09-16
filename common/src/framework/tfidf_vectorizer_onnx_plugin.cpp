/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    int64_t max_gram_length = 0, max_skip_count = 0, min_gram_length = 0;
    std::string mode;
    std::vector<float> weights;
    std::vector<int64_t> ngram_counts, ngram_indexes, pool_int64s;
    std::vector<std::string> pool_strings;
    bool has_max_gram_length = false, has_max_skip_count = false, has_min_gram_length = false, has_mode = false;
    bool has_weights = false, has_ngram_counts = false, has_ngram_indexes = false, has_pool_int64s = false;
    bool has_pool_strings = false;
    ge::AscendString attributes;
    // ONNX repeated AttributeProto messages are exposed as JSON in "attribute".
    if (op_src.GetAttr("attribute", attributes) == ge::GRAPH_SUCCESS) {
        if (attributes.GetString() == nullptr) {
            return FAILED;
        }
        try {
            const auto root = nlohmann::json::parse(attributes.GetString());
            if (!root.is_object() || (root.contains("attribute") && !root.at("attribute").is_array())) {
                return FAILED;
            }
            for (const auto &attr : root.value("attribute", nlohmann::json::array())) {
                if (!attr.is_object()) {
                    return FAILED;
                }
                const auto name = attr.value("name", "");
                const auto type = attr.value("type", 0);
                if ((name == "max_gram_length" || name == "max_skip_count" || name == "min_gram_length") && type == 2) {
                    int64_t value = 0;
                    if (attr.contains("i") && !GetInt64(attr.at("i"), value)) {
                        return FAILED;
                    }
                    if (name == "max_gram_length") {
                        max_gram_length = value;
                        has_max_gram_length = true;
                    } else if (name == "max_skip_count") {
                        max_skip_count = value;
                        has_max_skip_count = true;
                    } else {
                        min_gram_length = value;
                        has_min_gram_length = true;
                    }
                } else if (name == "mode" && type == 3) {
                    if (!attr.contains("s")) {
                        mode.clear();
                        has_mode = true;
                    } else if (!attr.at("s").is_string()) {
                        return FAILED;
                    } else {
                        mode = attr.at("s").get<std::string>();
                        has_mode = true;
                    }
                } else if (name == "weights" && type == 6) {
                    if (!attr.contains("floats")) {
                        weights.clear();
                        has_weights = true;
                    } else if (!attr.at("floats").is_array()) {
                        return FAILED;
                    } else {
                        weights = attr.at("floats").get<std::vector<float>>();
                        has_weights = true;
                    }
                } else if ((name == "ngram_counts" || name == "ngram_indexes" || name == "pool_int64s") && type == 7) {
                    if (!attr.contains("ints")) {
                        if (name == "ngram_counts") {
                            ngram_counts.clear();
                            has_ngram_counts = true;
                        } else if (name == "ngram_indexes") {
                            ngram_indexes.clear();
                            has_ngram_indexes = true;
                        } else {
                            pool_int64s.clear();
                            has_pool_int64s = true;
                        }
                    } else if (!attr.at("ints").is_array()) {
                        return FAILED;
                    } else {
                        std::vector<int64_t> value;
                        for (const auto &element : attr.at("ints")) {
                            int64_t integer = 0;
                            if (!GetInt64(element, integer)) {
                                return FAILED;
                            }
                            value.push_back(integer);
                        }
                        if (name == "ngram_counts") {
                            ngram_counts = value;
                            has_ngram_counts = true;
                        } else if (name == "ngram_indexes") {
                            ngram_indexes = value;
                            has_ngram_indexes = true;
                        } else {
                            pool_int64s = value;
                            has_pool_int64s = true;
                        }
                    }
                } else if (name == "pool_strings" && type == 8) {
                    if (!attr.contains("strings")) {
                        pool_strings.clear();
                        has_pool_strings = true;
                    } else if (!attr.at("strings").is_array()) {
                        return FAILED;
                    } else {
                        pool_strings = attr.at("strings").get<std::vector<std::string>>();
                        has_pool_strings = true;
                    }
                }
            }
        } catch (const nlohmann::json::exception &) {
            return FAILED;
        }
    }

    if (has_max_gram_length)
        op_dest.SetAttr("max_gram_length", max_gram_length);
    if (has_max_skip_count)
        op_dest.SetAttr("max_skip_count", max_skip_count);
    if (has_min_gram_length)
        op_dest.SetAttr("min_gram_length", min_gram_length);
    if (has_mode)
        op_dest.SetAttr("mode", mode);
    if (has_weights)
        op_dest.SetAttr("weights", weights);
    if (has_ngram_counts)
        op_dest.SetAttr("ngram_counts", ngram_counts);
    if (has_ngram_indexes)
        op_dest.SetAttr("ngram_indexes", ngram_indexes);
    if (has_pool_int64s)
        op_dest.SetAttr("pool_int64s", pool_int64s);
    if (has_pool_strings)
        op_dest.SetAttr("pool_strings", pool_strings);

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
