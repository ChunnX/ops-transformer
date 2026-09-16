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

#include "../../../src/framework/npu_scaled_masked_softmax_onnx_plugin.cpp"

TEST(OnnxScaledMaskedSoftmaxPluginTest, MissingAttributeArrayUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString("{}"));
    EXPECT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::SUCCESS);
}

class OnnxScaledMaskedSoftmaxPluginTestInvalidJson : public testing::TestWithParam<const char *> {};

TEST_P(OnnxScaledMaskedSoftmaxPluginTestInvalidJson, RejectsWithoutChangingExistingAttribute)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("sentinel", int64_t{42});
    src.SetAttr("attribute", ge::AscendString(GetParam()));
    EXPECT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::FAILED);
    int64_t sentinel = 0;
    ASSERT_EQ(dest.GetAttr("sentinel", sentinel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sentinel, 42);
}

INSTANTIATE_TEST_SUITE_P(InvalidEnvelope, OnnxScaledMaskedSoftmaxPluginTestInvalidJson,
                         testing::Values("[]", "null", R"({"attribute":null})", R"({"attribute":[null]})",
                                         R"({"attribute":[{"name":1,"type":2}]})"));

TEST(OnnxScaledMaskedSoftmaxPluginTest, ParsesAttributes)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(
                    R"({"attribute":[{"name":"scale","type":1,"f":0.25},{"name":"fixed_triu_mask","type":2,"i":1}]})"));

    ASSERT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::SUCCESS);
    float scale = 0.0F;
    bool fixed_triu_mask = false;
    ASSERT_EQ(dest.GetAttr("scale", scale), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("fixed_triu_mask", fixed_triu_mask), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 0.25F);
    EXPECT_TRUE(fixed_triu_mask);
}

TEST(OnnxScaledMaskedSoftmaxPluginTest, UsesDefaultsWithoutAttributes)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");

    ASSERT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::SUCCESS);
    float scale = 0.0F;
    bool fixed_triu_mask = true;
    ASSERT_EQ(dest.GetAttr("scale", scale), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("fixed_triu_mask", fixed_triu_mask), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 1.0F);
    EXPECT_FALSE(fixed_triu_mask);
}

TEST(OnnxScaledMaskedSoftmaxPluginTest, RejectsMalformedJson)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString("not-json"));

    EXPECT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::FAILED);
}

TEST(OnnxScaledMaskedSoftmaxPluginTest, HandlesMissingFieldsAndNonOneMaskValue)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(R"({"attribute":[{"name":"scale","type":1},{"name":"fixed_triu_mask","type":2,"i":2}]})"));
    ASSERT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::SUCCESS);
    float scale = 1.0F;
    bool fixed_triu_mask = true;
    ASSERT_EQ(dest.GetAttr("scale", scale), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("fixed_triu_mask", fixed_triu_mask), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale, 0.0F);
    EXPECT_FALSE(fixed_triu_mask);
}

TEST(OnnxScaledMaskedSoftmaxPluginTest, RejectsInvalidAttributeValues)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"scale","type":1,"f":false}]})"));
    EXPECT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::FAILED);
    src.SetAttr("attribute",
                ge::AscendString(R"({"attribute":[{"name":"fixed_triu_mask","type":2,"i":18446744073709551615}]})"));
    EXPECT_EQ(domi::ParseParamsNPUScaledMaskedSoftmax(src, dest), domi::FAILED);
}
