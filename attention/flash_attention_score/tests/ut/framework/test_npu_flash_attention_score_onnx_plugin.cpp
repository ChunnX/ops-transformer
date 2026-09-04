/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../../framework/npu_flash_attention_score_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string &name)
{
    return ge::Operator(name, "TestOp");
}

ge::Operator CreateSourceOperator(const std::vector<std::string> &inputs, const std::vector<std::string> &outputs,
                                  const std::string &attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("input", inputs);
    op_src.SetAttr("output", outputs);
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxNpuFlashAttentionScorePluginTest, ParsesAttrsAndDynamicIo)
{
    const std::vector<std::string> inputs = {"query",        "key",        "value",  "real_shift",        "drop_mask",
                                             "padding_mask", "atten_mask", "prefix", "actual_seq_lengths"};
    const std::vector<std::string> outputs = {"attention_out"};
    const std::string attrs = R"({"attribute":[
        {"name":"head_num","type":2,"i":10},
        {"name":"input_layout","type":3,"s":"BSH"},
        {"name":"scale","type":1,"f":"0.25"},
        {"name":"keep_prob","type":1,"f":"0.9"},
        {"name":"pre_tockens","type":2,"i":2},
        {"name":"next_tockens","type":2,"i":3},
        {"name":"inner_precise","type":2,"i":1},
        {"name":"sparse_mode","type":2,"i":2}
    ]})";
    ge::Operator op_src = CreateSourceOperator(inputs, outputs, attrs);
    ge::Operator op_dest = CreateOperator("partitioned_call");

    EXPECT_EQ(domi::ParseParamsFlashAttention(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetInputsSize(), inputs.size());
    EXPECT_EQ(op_dest.GetOutputsSize(), outputs.size());

    int head_num = 0;
    std::string input_layout;
    float scale = 0.0F;
    float keep_prob = 0.0F;
    int pre_tockens = 0;
    int next_tockens = 0;
    int inner_precise = 0;
    int sparse_mode = 0;
    EXPECT_EQ(op_dest.GetAttr("head_num", head_num), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("input_layout", input_layout), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("scale", scale), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("keep_prob", keep_prob), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("pre_tockens", pre_tockens), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("next_tockens", next_tockens), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("inner_precise", inner_precise), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("sparse_mode", sparse_mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(head_num, 10);
    EXPECT_EQ(input_layout, "BSH");
    EXPECT_FLOAT_EQ(scale, 0.25F);
    EXPECT_FLOAT_EQ(keep_prob, 0.9F);
    EXPECT_EQ(pre_tockens, 2);
    EXPECT_EQ(next_tockens, 3);
    EXPECT_EQ(inner_precise, 1);
    EXPECT_EQ(sparse_mode, 2);
}

TEST(OnnxNpuFlashAttentionScorePluginTest, MissingRequiredAttrFails)
{
    const std::vector<std::string> inputs = {"query", "key", "value"};
    const std::vector<std::string> outputs = {"attention_out"};
    const std::string attrs = R"({"attribute":[{"name":"scale","type":1,"f":"0.25"}]})";
    ge::Operator op_src = CreateSourceOperator(inputs, outputs, attrs);
    ge::Operator op_dest = CreateOperator("partitioned_call");

    EXPECT_NE(domi::ParseParamsFlashAttention(op_src, op_dest), domi::SUCCESS);
}
