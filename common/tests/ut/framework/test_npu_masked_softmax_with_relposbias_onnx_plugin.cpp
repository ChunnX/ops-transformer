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

#include "../../../src/framework/npu_masked_softmax_with_relposbias_onnx_plugin.cpp"

TEST(OnnxMaskedSoftmaxWithRelPosBiasPluginTest, PreservesDynamicInputAndOutputCounts)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.DynamicInputRegister("input", 3);
    src.DynamicOutputRegister("output", 1);
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"scale","type":1,"f":1.0}]})"));
    ASSERT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::SUCCESS);
    EXPECT_EQ(dest.GetInputsSize(), src.GetInputsSize());
    EXPECT_EQ(dest.GetOutputsSize(), src.GetOutputsSize());
}

TEST(OnnxMaskedSoftmaxWithRelPosBiasPluginTest, MissingAttributeArrayUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString("{}"));
    EXPECT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::SUCCESS);
}

class OnnxMaskedSoftmaxWithRelPosBiasPluginTestInvalidJson : public testing::TestWithParam<const char *> {};

TEST_P(OnnxMaskedSoftmaxWithRelPosBiasPluginTestInvalidJson, RejectsWithoutChangingExistingAttribute)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("sentinel", int64_t{42});
    src.SetAttr("attribute", ge::AscendString(GetParam()));
    EXPECT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::FAILED);
    int64_t sentinel = 0;
    ASSERT_EQ(dest.GetAttr("sentinel", sentinel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sentinel, 42);
}

INSTANTIATE_TEST_SUITE_P(InvalidEnvelope, OnnxMaskedSoftmaxWithRelPosBiasPluginTestInvalidJson,
                         testing::Values("[]", "null", R"({"attribute":null})", R"({"attribute":[null]})",
                                         R"({"attribute":[{"name":1,"type":2}]})"));

TEST(OnnxMaskedSoftmaxWithRelPosBiasPluginTest, ParsesAttributesAndName)
{
    ge::Operator src("original_name", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"scale_value","type":1,"f":0.5},{"name":"inner_precision_mode","type":2,"i":3}]})"));

    ASSERT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::SUCCESS);
    float scale_value = 0.0F;
    int64_t inner_precision_mode = 0;
    std::string name;
    ASSERT_EQ(dest.GetAttr("scale_value", scale_value), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("inner_precision_mode", inner_precision_mode), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale_value, 0.5F);
    EXPECT_EQ(inner_precision_mode, 3);
    EXPECT_EQ(name, "original_name");
}

TEST(OnnxMaskedSoftmaxWithRelPosBiasPluginTest, RejectsInvalidAttributeEnvelope)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":{}})"));

    EXPECT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::FAILED);
}

TEST(OnnxMaskedSoftmaxWithRelPosBiasPluginTest, UsesDefaultsWithoutAttributes)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    ASSERT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::SUCCESS);
    float scale_value = 1.0F;
    int64_t inner_precision_mode = 0;
    ASSERT_EQ(dest.GetAttr("scale_value", scale_value), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("inner_precision_mode", inner_precision_mode), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(scale_value, 0.0F);
    EXPECT_EQ(inner_precision_mode, 1);
}

TEST(OnnxMaskedSoftmaxWithRelPosBiasPluginTest, RejectsOutOfRangeAndWrongValueType)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(R"({"attribute":[{"name":"inner_precision_mode","type":2,"i":18446744073709551615}]})"));
    EXPECT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::FAILED);
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"scale_value","type":1,"f":"bad"}]})"));
    EXPECT_EQ(domi::ParseParamsNpuMaskedSoftmaxWithRelPosBias(src, dest), domi::FAILED);
}
