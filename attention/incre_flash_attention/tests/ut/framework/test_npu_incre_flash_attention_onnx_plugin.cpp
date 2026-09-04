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

#include "../../../framework/npu_incre_flash_attention_onnx_plugin.cpp"

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

TEST(OnnxNpuIncreFlashAttentionPluginTest, ParsesAttrsAndDynamicSize)
{
    const std::vector<std::string> inputs = {"query",     "key_0",      "value_0",
                                             "pse_shift", "atten_mask", "actual_seq_lengths"};
    const std::vector<std::string> outputs = {"attention_out"};
    const std::string attrs = R"({"attribute":[
        {"name":"num_heads","type":2,"i":2},
        {"name":"input_layout","type":3,"s":"BSH"},
        {"name":"scale_value","type":1,"f":"0.5"}
    ]})";
    ge::Operator op_src = CreateSourceOperator(inputs, outputs, attrs);
    ge::Operator op_dest = CreateOperator("partitioned_call");

    EXPECT_EQ(domi::ParseParamsIncreFlashAttention(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetInputsSize(), inputs.size());
    EXPECT_EQ(op_dest.GetOutputsSize(), outputs.size());

    int num_heads = 0;
    std::string input_layout;
    float scale_value = 0.0F;
    int dynamic_size = 0;
    EXPECT_EQ(op_dest.GetAttr("num_heads", num_heads), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("input_layout", input_layout), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("scale_value", scale_value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("dynamic_size", dynamic_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_heads, 2);
    EXPECT_EQ(input_layout, "BSH");
    EXPECT_FLOAT_EQ(scale_value, 0.5F);
    EXPECT_EQ(dynamic_size, 1);
}

TEST(OnnxNpuIncreFlashAttentionPluginTest, OddDynamicInputFails)
{
    const std::vector<std::string> inputs = {"query", "key_0", "value_0", "pse_shift", "atten_mask"};
    const std::vector<std::string> outputs = {"attention_out"};
    const std::string attrs = R"({"attribute":[
        {"name":"num_heads","type":2,"i":2},
        {"name":"input_layout","type":3,"s":"BSH"}
    ]})";
    ge::Operator op_src = CreateSourceOperator(inputs, outputs, attrs);
    ge::Operator op_dest = CreateOperator("partitioned_call");

    EXPECT_NE(domi::ParseParamsIncreFlashAttention(op_src, op_dest), domi::SUCCESS);
}
