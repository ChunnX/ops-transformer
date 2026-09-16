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
 * \file npu_masked_softmax_with_relposbias_onnx_plugin.cpp
 * \brief
 */

#include <cstdint>
#include <limits>

#include "nlohmann/json.hpp"
#include "graph/operator.h"
#include "register/register.h"
#include "graph/graph.h"
#include "stub_ops.h"
#include "log/log.h"
#include "op_transformer_proto_extend.h"

namespace domi {

static std::string GetOpName(const ge::Operator &op)
{
    ge::AscendString name;
    if (op.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
        return "None";
    }
    return name.GetString();
}

constexpr int OUTPUT_INDEX = 1;

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
// x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode float scale_value=1.0, int inner_precision_mode=0
static Status ParseParamsNpuMaskedSoftmaxWithRelPosBias(const ge::Operator &op_src, ge::Operator &op_dest)
{
    op_dest.DynamicInputRegister("x", static_cast<uint32_t>(op_src.GetInputsSize()));
    op_dest.DynamicOutputRegister("y", static_cast<uint32_t>(op_src.GetOutputsSize()));

    float scale_value = 0.0F;
    int64_t inner_precision_mode = 1;
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
                if (attr.value("name", "") == "inner_precision_mode" && attr.value("type", 0) == 2) {
                    if (!attr.contains("i")) {
                        inner_precision_mode = 0;
                    } else {
                        if (!GetInt64(attr.at("i"), inner_precision_mode)) {
                            return FAILED;
                        }
                    }
                } else if (attr.value("name", "") == "scale_value" && attr.value("type", 0) == 1) {
                    if (!attr.contains("f")) {
                        scale_value = 0.0F;
                    } else if (!attr.at("f").is_number()) {
                        return FAILED;
                    } else {
                        scale_value = attr.at("f").get<float>();
                    }
                }
            }
        } catch (const nlohmann::json::exception &) {
            return FAILED;
        }
    }
    ge::AscendString name;
    if (op_src.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
        return FAILED;
    }
    op_dest.SetAttr("name", name.GetString());
    op_dest.SetAttr("inner_precision_mode", inner_precision_mode);
    op_dest.SetAttr("scale_value", scale_value);
    op_dest.SetAttr("original_type", "npu::1::NPUMaskedSoftmaxWithRelPosBias");
    return SUCCESS;
}

static Status ParseOpToGraphNpuMaskedSoftmaxWithRelPosBias(const ge::Operator &op, ge::Graph &graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);

    int inner_precision_mode = 0;
    if (op.GetAttr("inner_precision_mode", inner_precision_mode) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get inner_precision_mode from op failed");
        return FAILED;
    }
    float scale_value = 1.0f;
    if (op.GetAttr("scale_value", scale_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get scale_value from op failed");
        return FAILED;
    }
    auto masked_softmax_with_relposbias =
        ge::op::MaskedSoftmaxWithRelPosBias((ori_name + "_MaskedSoftmaxWithRelPosBias").c_str())
            .set_input_x(data0)
            .set_input_atten_mask(data1)
            .set_input_relative_pos_bias(data2)
            .set_attr_scale_value(scale_value)
            .set_attr_inner_precision_mode(inner_precision_mode);

    std::vector<ge::Operator> inputs{data0, data1, data2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(masked_softmax_with_relposbias, std::vector<std::size_t>{OUTPUT_INDEX});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_masked_softmax_with_rel_pos_bias op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::11::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::12::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::13::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::14::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::15::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::16::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::17::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::18::NPUMaskedSoftmaxWithRelPosBias"),
                   ge::AscendString("ai.onnx::19::NPUMaskedSoftmaxWithRelPosBias")})
    .ParseParamsByOperatorFn(ParseParamsNpuMaskedSoftmaxWithRelPosBias)
    .ParseOpToGraphFn(ParseOpToGraphNpuMaskedSoftmaxWithRelPosBias)
    .ImplyType(ImplyType::TVM);
} // namespace domi
