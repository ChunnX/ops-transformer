/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_scaled_masked_softmax_onnx_plugin.cpp
 * \brief onnx plugin for npu custom operator npu_scaled_masked_softmax
 */

#include <cstdint>
#include <limits>

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

static Status ParseParamsNPUScaledMaskedSoftmax(const ge::Operator &op_src, ge::Operator &op_dest)
{
    float scale = 1.0;
    bool fixed_triu_mask = false;
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
                if (attr.value("name", "") == "scale" && attr.value("type", 0) == 1) {
                    if (!attr.contains("f")) {
                        scale = 0.0F;
                    } else if (!attr.at("f").is_number()) {
                        return FAILED;
                    } else {
                        scale = attr.at("f").get<float>();
                    }
                } else if (attr.value("name", "") == "fixed_triu_mask" && attr.value("type", 0) == 2) {
                    if (!attr.contains("i")) {
                        fixed_triu_mask = false;
                    } else {
                        int64_t integer_value = 0;
                        if (!GetInt64(attr.at("i"), integer_value)) {
                            return FAILED;
                        }
                        fixed_triu_mask = integer_value == 1;
                    }
                }
            }
        } catch (const nlohmann::json::exception &) {
            return FAILED;
        }
    }
    op_dest.SetAttr("scale", scale);
    op_dest.SetAttr("fixed_triu_mask", fixed_triu_mask);

    return SUCCESS;
}

// register npu_scaled_masked_softmax op info to GE
REGISTER_CUSTOM_OP("ScaledMaskedSoftmax")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::11::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::12::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::13::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::14::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::15::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::16::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::17::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::18::NPUScaledMaskedSoftmax")})
    .ParseParamsByOperatorFn(ParseParamsNPUScaledMaskedSoftmax)
    .ImplyType(ImplyType::TVM);
} // namespace domi
