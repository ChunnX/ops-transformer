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
 * \file npu_incre_flash_attention_onnx_plugin.cpp
 * \brief
 */
#include <cerrno>
#include <cstdlib>
#include <string>

#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "stub_ops.h"
#include "attention/incre_flash_attention/op_graph/incre_flash_attention_proto.h"

namespace domi {
namespace {
static std::string GetOpName(const ge::Operator &op)
{
    ge::AscendString op_name;
    if (op.GetName(op_name) != ge::GRAPH_SUCCESS) {
        return "None";
    }
    return op_name.GetString();
}

static bool StrToFloat(const std::string &str, float &value)
{
    if (str.empty()) {
        return false;
    }
    errno = 0;
    char *end_ptr = nullptr;
    const float parsed = std::strtof(str.c_str(), &end_ptr);
    if (end_ptr == str.c_str() || *end_ptr != '\0' || errno == ERANGE) {
        return false;
    }
    value = parsed;
    return true;
}
} // namespace

constexpr int REQUIRED_ATTR = 2;
constexpr int OUTPUT_INDEX = 0;
constexpr int POS_ACTUAL_SEQ_LENGTHS = 1;
constexpr int POS_ATTEN_MASK = 2;
constexpr int POS_PSE_SHIFT = 3;
constexpr int REQUIRED_INPUT_SIZE = 4; // query/pse_shift/atten_mask/actual_seq_lengths
constexpr int GROUP_SIZE = 2;
static Status ParseParamsIncreFlashAttention(const ge::Operator &op_src, ge::Operator &op_dest)
{
    using json = nlohmann::json;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    if (op_src.GetAttr("input", input_names) != ge::GRAPH_SUCCESS ||
        op_src.GetAttr("output", output_names) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op_dest).c_str(), "get input/output from op failed.");
        return FAILED;
    }

    int input_size = static_cast<int>(input_names.size());
    int dynamicSize = input_size - REQUIRED_INPUT_SIZE;
    int output_size = static_cast<int>(output_names.size());
    op_dest.DynamicInputRegister("x", static_cast<uint32_t>(input_size));
    op_dest.DynamicOutputRegister("y", static_cast<uint32_t>(output_size));

    if (dynamicSize % GROUP_SIZE != 0) {
        OP_LOGE(GetOpName(op_dest).c_str(), "dynamic size error.");
        return FAILED;
    }
    dynamicSize /= GROUP_SIZE;

    int num_heads = 0;
    std::string input_layout = "";
    float scale_value = 1.0f;
    int required_attr_num = 0;
    ge::AscendString attrs_string;
    if (op_src.GetAttr("attribute", attrs_string) == ge::GRAPH_SUCCESS) {
        try {
            json node_attrs = json::parse(attrs_string.GetString());
            if (node_attrs.contains("attribute") && node_attrs["attribute"].is_array()) {
                for (json &attr : node_attrs["attribute"]) {
                    std::string attr_name = attr.value("name", "");
                    if (attr_name == "num_heads" && attr.contains("i")) {
                        num_heads = attr["i"].get<int>();
                        ++required_attr_num;
                    } else if (attr_name == "input_layout" && attr.contains("s")) {
                        input_layout = attr["s"].get<std::string>();
                        ++required_attr_num;
                    } else if (attr_name == "scale_value" && attr.contains("f")) {
                        std::string scale_value_str = attr["f"].get<std::string>();
                        if (!StrToFloat(scale_value_str, scale_value)) {
                            OP_LOGE(GetOpName(op_dest).c_str(), "invalid scale_value value: %s",
                                    scale_value_str.c_str());
                            return FAILED;
                        }
                    }
                }
            }
        } catch (const nlohmann::json::exception &e) {
            OP_LOGE(GetOpName(op_dest).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(op_dest).c_str(), "get unknown exception, please check onnx attributes.");
            return FAILED;
        }
    }

    if (required_attr_num != REQUIRED_ATTR) {
        OP_LOGE(GetOpName(op_dest).c_str(), "attr num two is required.");
        return FAILED;
    }

    op_dest.SetAttr("name", GetOpName(op_src));
    op_dest.SetAttr("num_heads", num_heads);
    op_dest.SetAttr("input_layout", input_layout);
    op_dest.SetAttr("scale_value", scale_value);
    op_dest.SetAttr("dynamic_size", dynamicSize);
    op_dest.SetAttr("original_type", "npu::1::NPUIncreFlashAttention");
    return SUCCESS;
}

static Status GetNpuIncreFlashAttentionAttrByName(const ge::Operator &op, int &num_heads, std::string &input_layout,
                                                  float &scale_value)
{
    if (op.GetAttr("num_heads", num_heads) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get num_heads from op failed");
        return FAILED;
    }
    if (op.GetAttr("input_layout", input_layout) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get input_layout from op failed");
        return FAILED;
    }
    if (op.GetAttr("scale_value", scale_value) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get scale_value from op failed");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseOpToGraphNpuIncreFlashAttention(const ge::Operator &op, ge::Graph &graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int dynamicSize = 0;
    op.GetAttr("dynamic_size", dynamicSize);
    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

    int num_heads = 0;
    string input_layout = "BSH";
    float scale_value = 1.0f;
    Status ret = GetNpuIncreFlashAttentionAttrByName(op, num_heads, input_layout, scale_value);
    if (ret != SUCCESS) {
        return FAILED;
    }

    auto incre_flash_attention = ge::op::IncreFlashAttention((ori_name + "_IncreFlashAttention").c_str())
                                     .set_input_query(data0)
                                     .create_dynamic_input_key(dynamicSize)
                                     .create_dynamic_input_value(dynamicSize);

    std::vector<ge::Operator> inputs{data0};

    int index = 1;
    for (int i = 0; i < dynamicSize; ++i) {
        auto datak = ge::op::Data((ori_name + "_datak" + to_string(i)).c_str()).set_attr_index(index);
        incre_flash_attention.set_dynamic_input_key(index++, datak);
        inputs.emplace_back(datak);
    }
    for (int i = 0; i < dynamicSize; ++i) {
        auto datav = ge::op::Data((ori_name + "_datav" + to_string(i)).c_str()).set_attr_index(index);
        incre_flash_attention.set_dynamic_input_value(index++, datav);
        inputs.emplace_back(datav);
    }

    auto data_pd = ge::op::Data((ori_name + "_data_pd").c_str()).set_attr_index(index++);
    auto data_am = ge::op::Data((ori_name + "_data_am").c_str()).set_attr_index(index++);
    auto data_acq = ge::op::Data((ori_name + "_data_acq").c_str()).set_attr_index(index);

    incre_flash_attention.set_input_pse_shift(data_pd)
        .set_input_atten_mask(data_am)
        .set_input_actual_seq_lengths(data_acq)
        .set_attr_num_heads(num_heads)
        .set_attr_scale_value(scale_value)
        .set_attr_input_layout(input_layout);

    inputs.emplace_back(data_pd);
    inputs.emplace_back(data_am);
    inputs.emplace_back(data_acq);

    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(incre_flash_attention, std::vector<std::size_t>{OUTPUT_INDEX});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_incre_flash_attention op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::11::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::12::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::13::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::14::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::15::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::16::NPUIncreFlashAttention"),
                   ge::AscendString("ai.onnx::19::NPUIncreFlashAttention")})
    .ParseParamsByOperatorFn(ParseParamsIncreFlashAttention)
    .ParseOpToGraphFn(ParseOpToGraphNpuIncreFlashAttention)
    .ImplyType(ImplyType::TVM);
} // namespace domi
