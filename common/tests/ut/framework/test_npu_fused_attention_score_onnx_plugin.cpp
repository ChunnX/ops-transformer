/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../src/framework/npu_fused_attention_score_onnx_plugin.cpp"

TEST(OnnxFusedAttentionScorePluginTest, ParsesRequiredAndOptionalAttributes)
{
    ge::Operator src("original_name", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"scale","type":1,"f":0.125},{"name":"keep_prob","type":1,"f":0.75},)"
            R"({"name":"query_transpose","type":2,"i":1},{"name":"key_transpose","type":2,"i":0},)"
            R"({"name":"bmm_score_transpose_a","type":2,"i":1},{"name":"bmm_score_transpose_b","type":2,"i":1}]})"));

    ASSERT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::SUCCESS);
    float scale = 0.0F;
    float keep_prob = 0.0F;
    bool query_transpose = false;
    bool key_transpose = true;
    bool bmm_score_transpose_a = false;
    bool bmm_score_transpose_b = false;
    std::string name;
    ASSERT_EQ(dest.GetAttr("scale", scale), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("keep_prob", keep_prob), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("query_transpose", query_transpose), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("key_transpose", key_transpose), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("bmm_score_transpose_a", bmm_score_transpose_a), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("bmm_score_transpose_b", bmm_score_transpose_b), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 0.125F);
    EXPECT_FLOAT_EQ(keep_prob, 0.75F);
    EXPECT_TRUE(query_transpose);
    EXPECT_FALSE(key_transpose);
    EXPECT_TRUE(bmm_score_transpose_a);
    EXPECT_TRUE(bmm_score_transpose_b);
    EXPECT_EQ(name, "original_name");
}

TEST(OnnxFusedAttentionScorePluginTest, MissingScaleIsRejected)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"keep_prob","type":1,"f":0.75}]})"));

    EXPECT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::FAILED);
}

TEST(OnnxFusedAttentionScorePluginTest, PreservesDynamicInputAndOutputCounts)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.DynamicInputRegister("input", 4);
    src.DynamicOutputRegister("output", 1);
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"scale","type":1,"f":1.0}]})"));

    ASSERT_EQ(domi::ParseParamsNpuFusedAttentionScore(src, dest), domi::SUCCESS);
    EXPECT_EQ(dest.GetInputsSize(), src.GetInputsSize());
    EXPECT_EQ(dest.GetOutputsSize(), src.GetOutputsSize());
}

class OnnxFusedAttentionScoreInvalidJsonTest : public testing::TestWithParam<const char *> {};

TEST_P(OnnxFusedAttentionScoreInvalidJsonTest, RejectsWithoutChangingExistingAttributes)
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

INSTANTIATE_TEST_SUITE_P(
    InvalidInput, OnnxFusedAttentionScoreInvalidJsonTest,
    testing::Values("not-json", "[]", "null", R"({"attribute":null})", R"({"attribute":{}})", R"({"attribute":[null]})",
                    R"({"attribute":[{"name":1,"type":2}]})", R"({"attribute":[{"name":"scale","type":1,"f":"bad"}]})",
                    R"({"attribute":[{"name":"query_transpose","type":2,"i":18446744073709551615}]})"));
