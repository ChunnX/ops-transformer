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

#include "nlohmann/json.hpp"
#include "graph/operator.h"
#include "graph/graph.h"
#include "register/register.h"
#include "stub_ops.h"
#include "onnx_plugin_utils.h"
#include "op_transformer_proto_extend.h"

namespace domi {
constexpr int REQUIRED_ATTR = 1;
constexpr int ONE = 1;
constexpr int INDEX_TWO = 2;
constexpr int INDEX_THREE = 3;
constexpr int ACL_FLOAT16 = 1;
// ONNX AttributeProto type values used by this plugin.
constexpr int ATTRIBUTE_TYPE_FLOAT = 1;
constexpr int ATTRIBUTE_TYPE_INT = 2;

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

static Status ParseParamsNpuFusedAttentionScore(const ge::Operator &op_src, ge::Operator &op_dest)
{
    op_dest.DynamicInputRegister("x", static_cast<uint32_t>(op_src.GetInputsSize()));
    op_dest.DynamicOutputRegister("y", static_cast<uint32_t>(op_src.GetOutputsSize()));

    int required_attr_num = 0;
    float scale = 0.0F;
    float keep_prob = 1.0F;
    bool query_transpose = false;
    bool key_transpose = false;
    bool bmm_score_transpose_a = false;
    bool bmm_score_transpose_b = false;
    ge::AscendString attributes;
    // ONNX repeated AttributeProto messages are exposed as JSON in "attribute".
    if (op_src.GetAttr("attribute", attributes) == ge::GRAPH_SUCCESS) {
        if (attributes.GetString() == nullptr) {
            return FAILED;
        }
        try {
            const auto root = nlohmann::json::parse(attributes.GetString());
            if (!root.is_object()) {
                return FAILED;
            }
            if (root.contains("attribute")) {
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
                    // Other types were ignored by the old parser.
                    if (name == "scale" && type == ATTRIBUTE_TYPE_FLOAT) {
                        // An omitted protobuf float field has the default value zero.
                        if (!attr.contains("f")) {
                            scale = 0.0F;
                        } else if (!attr.at("f").is_number()) {
                            return FAILED;
                        } else {
                            scale = attr.at("f").get<float>();
                        }
                        ++required_attr_num;
                    } else if (name == "keep_prob" && type == ATTRIBUTE_TYPE_FLOAT) {
                        if (!attr.contains("f")) {
                            keep_prob = 0.0F;
                        } else if (!attr.at("f").is_number()) {
                            return FAILED;
                        } else {
                            keep_prob = attr.at("f").get<float>();
                        }
                    } else if (type == ATTRIBUTE_TYPE_INT) {
                        // An omitted protobuf integer field has the default value zero.
                        int64_t parsed_value = 0;
                        if (attr.contains("i") && !GetInt64(attr.at("i"), parsed_value)) {
                            return FAILED;
                        }
                        const bool transpose = (parsed_value == 1);
                        if (name == "query_transpose") {
                            query_transpose = transpose;
                        } else if (name == "key_transpose") {
                            key_transpose = transpose;
                        } else if (name == "bmm_score_transpose_a") {
                            bmm_score_transpose_a = transpose;
                        } else if (name == "bmm_score_transpose_b") {
                            bmm_score_transpose_b = transpose;
                        }
                    }
                }
            }
        } catch (const nlohmann::json::exception &) {
            return FAILED;
        }
    }

    if (required_attr_num != REQUIRED_ATTR) {
        OP_LOGE(GetOpName(op_dest).c_str(), "attr scale is required.");
        return FAILED;
    }
    ge::AscendString name;
    if (op_src.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
        return FAILED;
    }
    op_dest.SetAttr("name", name.GetString());
    op_dest.SetAttr("scale", scale);
    op_dest.SetAttr("keep_prob", keep_prob);
    op_dest.SetAttr("query_transpose", query_transpose);
    op_dest.SetAttr("key_transpose", key_transpose);
    op_dest.SetAttr("bmm_score_transpose_a", bmm_score_transpose_a);
    op_dest.SetAttr("bmm_score_transpose_b", bmm_score_transpose_b);
    op_dest.SetAttr("original_type", "npu::1::NPUFusedAttentionScore");
    return SUCCESS;
}

namespace {
static Status GetAttrFromPre3(const ge::Operator &op, float &scale, float &keep_prob, bool &query_transpose)
{
    if (op.GetAttr("scale", scale) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get scale from op failed");
        return FAILED;
    }
    if (op.GetAttr("keep_prob", keep_prob) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_prob from op failed");
        return FAILED;
    }
    if (op.GetAttr("query_transpose", query_transpose) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get query_transpose from op failed");
        return FAILED;
    }
    return SUCCESS;
}

static Status GetAttrFromLast3(const ge::Operator &op, bool &key_transpose, bool &bmm_score_transpose_a,
                               bool &bmm_score_transpose_b)
{
    if (op.GetAttr("key_transpose", key_transpose) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get key_transpose from op failed");
        return FAILED;
    }
    if (op.GetAttr("bmm_score_transpose_a", bmm_score_transpose_a) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get bmm_score_transpose_a from op failed");
        return FAILED;
    }
    if (op.GetAttr("bmm_score_transpose_b", bmm_score_transpose_b) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get bmm_score_transpose_b from op failed");
        return FAILED;
    }
    return SUCCESS;
}
} // namespace

static Status ParseOpToGraphNpuFusedAttentionScore(const ge::Operator &op, ge::Graph &graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
    auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);

    float scale = 0;
    float keep_prob = 0;
    bool query_transpose = false;
    if (GetAttrFromPre3(op, scale, keep_prob, query_transpose) != SUCCESS) {
        return FAILED;
    }
    bool key_transpose = false;
    bool bmm_score_transpose_a = false;
    bool bmm_score_transpose_b = false;
    if (GetAttrFromLast3(op, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b) != SUCCESS) {
        return FAILED;
    }

    // create const input tensor "drop_mask" which is filled with the scalar value 1 for inferencing
    // deop_mask.size = {query_size[0], query_size[1], query_size[2], query_size[2]}
    ge::Tensor saclar_one = CreateScalar(ONE, ge::DT_UINT8);
    auto const_one = ge::op::Const((ori_name + "_Const_one").c_str()).set_attr_value(saclar_one);
    std::vector<int64_t> dims = op.GetInputDesc(0).GetShape().GetDims();
    dims[INDEX_THREE] = dims[INDEX_TWO];
    auto tensor_dims = Vec2Tensor(dims, {4}, ge::DT_INT64);
    auto const_dims = ge::op::Const((ori_name + "_Const_dims").c_str()).set_attr_value(tensor_dims);
    auto drop_mask =
        ge::op::Fill((ori_name + "_Fill_ones").c_str()).set_input_dims(const_dims).set_input_value(const_one);
    // set drop_mask["dtype"] to fp16 for inferencing
    auto cast_drop_mask =
        ge::op::Cast((ori_name + "_Cast_drop_mask").c_str()).set_input_x(drop_mask).set_attr_dst_type(ACL_FLOAT16);

    ge::Tensor tensor_scale = CreateScalar(scale, ge::DT_FLOAT);
    auto const_scale = ge::op::Const((ori_name + "_Const_scale").c_str()).set_attr_value(tensor_scale);
    auto cast_const_scale =
        ge::op::Cast((ori_name + "_Cast_const_scale").c_str()).set_input_x(const_scale).set_attr_dst_type(ACL_FLOAT16);

    auto AttentionScore = ge::op::AttentionScore((ori_name + "_AttentionScore").c_str())
                              .set_input_query(data0)
                              .set_input_key(data1)
                              .set_input_value(data2)
                              .set_input_padding_mask(data3)
                              .set_input_scale(cast_const_scale)
                              .set_input_drop_mask(cast_drop_mask)
                              .set_attr_keep_prob(keep_prob)
                              .set_attr_query_transpose(query_transpose)
                              .set_attr_key_transpose(key_transpose)
                              .set_attr_bmm_score_transpose_a(bmm_score_transpose_a)
                              .set_attr_bmm_score_transpose_b(bmm_score_transpose_b)
                              .set_attr_softmax_axes({-1});

    std::vector<ge::Operator> inputs{data0, data1, data2, data3};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(AttentionScore, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register npu_fused_attention_score op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::11::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::12::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::13::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::14::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::15::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::16::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::17::NPUFusedAttentionScore"),
                   ge::AscendString("ai.onnx::18::NPUFusedAttentionScore")})
    .ParseParamsByOperatorFn(ParseParamsNpuFusedAttentionScore)
    .ParseOpToGraphFn(ParseOpToGraphNpuFusedAttentionScore)
    .ImplyType(ImplyType::TVM);
} // namespace domi
