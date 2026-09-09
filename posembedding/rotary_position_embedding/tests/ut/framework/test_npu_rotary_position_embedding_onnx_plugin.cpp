/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "../../../framework/npu_rotary_position_embedding_onnx_plugin.cpp"

TEST(OnnxRotaryPositionEmbeddingPluginTest, MissingAttributeEnvelope)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, EmptyAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, MissingAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, ParsesJsonValue0)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":0}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, ParsesJsonValue1)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":1}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 1);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, ParsesJsonValue8)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":8}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 8);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, UnrelatedAttributeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"unrelated","type":2,"i":5}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, WrongOnnxTypeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":1,"i":7}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, MissingIntegerFieldDefaultsToZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, ParsesNegativeOne)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":-1}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, -1);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, MalformedJsonFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"(not-json)"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, WrongRootTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"([])"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, WrongArrayTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":{}})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, WrongEntryTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[4]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, StringIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":"2"}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, FloatIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":2.5}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, OutOfRangeIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":18446744073709551615}]})"));
    dest.SetAttr("mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, DuplicateRequiredAttributeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":1},{"name":"mode","type":2,"i":2}]})"));
    dest.SetAttr("mode", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{9});
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, UnrelatedAttributeDoesNotHideRequiredAttribute)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(R"({"attribute":[{"name":"other","type":2,"i":3},{"name":"mode","type":2,"i":2}]})"));
    dest.SetAttr("mode", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{2});
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, PreservesInt64Value)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":2147483648}]})"));
    dest.SetAttr("mode", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{2147483648});
}

TEST(OnnxRotaryPositionEmbeddingPluginTest, MalformedEntryAfterValueDoesNotChangeDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":2,"i":2},4]})"));
    dest.SetAttr("mode", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuRotaryPositionEmbedding(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{9});
}
