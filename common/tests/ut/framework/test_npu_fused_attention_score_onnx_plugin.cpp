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

#include "../../../src/framework/npu_fused_attention_score_onnx_plugin.cpp"

TEST(OnnxFusedAttentionScorePluginTest, PreservesDynamicInputAndOutputCounts)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.DynamicInputRegister("input", 3);
    src.DynamicOutputRegister("output", 1);
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"scale","type":1,"f":1.0}]})"));
    ASSERT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::SUCCESS);
    EXPECT_EQ(dest.GetInputsSize(), src.GetInputsSize());
    EXPECT_EQ(dest.GetOutputsSize(), src.GetOutputsSize());
}

TEST(OnnxFusedAttentionScorePluginTest, MissingAttributeArrayUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString("{}"));
    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
}

class OnnxFusedAttentionScorePluginTestInvalidJson : public testing::TestWithParam<const char *> {};

TEST_P(OnnxFusedAttentionScorePluginTestInvalidJson, RejectsWithoutChangingExistingAttribute)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("sentinel", int64_t{42});
    src.SetAttr("attribute", ge::AscendString(GetParam()));
    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
    int64_t sentinel = 0;
    ASSERT_EQ(dest.GetAttr("sentinel", sentinel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sentinel, 42);
}

INSTANTIATE_TEST_SUITE_P(InvalidEnvelope, OnnxFusedAttentionScorePluginTestInvalidJson,
                         testing::Values("[]", "null", R"({"attribute":null})", R"({"attribute":[null]})",
                                         R"({"attribute":[{"name":1,"type":2}]})"));

TEST(OnnxFusedAttentionScorePluginTest, ParsesRequiredAndOptionalAttributes)
{
    ge::Operator src("original_name", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"scale","type":1,"f":0.125},{"name":"keep_prob","type":1,"f":0.75},{"name":"query_transpose","type":2,"i":1},{"name":"key_transpose","type":2,"i":0}]})"));

    ASSERT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::SUCCESS);
    float scale = 0.0F;
    float keep_prob = 0.0F;
    bool query_transpose = false;
    bool key_transpose = true;
    std::string name;
    ASSERT_EQ(dest.GetAttr("scale", scale), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("keep_prob", keep_prob), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("query_transpose", query_transpose), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("key_transpose", key_transpose), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 0.125F);
    EXPECT_FLOAT_EQ(keep_prob, 0.75F);
    EXPECT_TRUE(query_transpose);
    EXPECT_FALSE(key_transpose);
    EXPECT_EQ(name, "original_name");
}

TEST(OnnxFusedAttentionScorePluginTest, RequiresExactlyOneScale)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"keep_prob","type":1,"f":0.5}]})"));

    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
}

TEST(OnnxFusedAttentionScorePluginTest, RejectsOutOfRangeInteger)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"scale","type":1,"f":1.0},{"name":"query_transpose","type":2,"i":18446744073709551615}]})"));

    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
}

TEST(OnnxFusedAttentionScorePluginTest, ParsesAllTransposeFlagsAndDefaultKeepProb)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"scale","type":1,"f":2.0},{"name":"query_transpose","type":2,"i":0},{"name":"key_transpose","type":2,"i":1},{"name":"bmm_score_transpose_a","type":2,"i":1},{"name":"bmm_score_transpose_b","type":2,"i":1}]})"));

    ASSERT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::SUCCESS);
    float keep_prob = 0.0F;
    bool key_transpose = false;
    bool transpose_a = false;
    bool transpose_b = false;
    ASSERT_EQ(dest.GetAttr("keep_prob", keep_prob), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("key_transpose", key_transpose), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("bmm_score_transpose_a", transpose_a), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("bmm_score_transpose_b", transpose_b), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(keep_prob, 1.0F);
    EXPECT_TRUE(key_transpose);
    EXPECT_TRUE(transpose_a);
    EXPECT_TRUE(transpose_b);
}

TEST(OnnxFusedAttentionScorePluginTest, RejectsDuplicateScaleAndMalformedEnvelope)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(R"({"attribute":[{"name":"scale","type":1,"f":1.0},{"name":"scale","type":1,"f":2.0}]})"));
    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":{}})"));
    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
}
