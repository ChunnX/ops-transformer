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

#include "../../../src/framework/tfidf_vectorizer_onnx_plugin.cpp"

TEST(OnnxTfIdfVectorizerPluginTest, MissingAttributeSetsNothing)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");

    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
    int64_t max_gram_length = -1;
    EXPECT_NE(dest.GetAttr("max_gram_length", max_gram_length), ge::GRAPH_SUCCESS);
}

TEST(OnnxTfIdfVectorizerPluginTest, ParsesAllAttributeKinds)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"max_gram_length","type":2,"i":2},{"name":"max_skip_count","type":2,"i":3},)"
            R"({"name":"min_gram_length","type":2,"i":1},{"name":"mode","type":3,"s":"TF"},)"
            R"({"name":"weights","type":6,"floats":[1.5,2.5]},{"name":"ngram_counts","type":7,"ints":[2,3]},)"
            R"({"name":"ngram_indexes","type":7,"ints":[0,1]},{"name":"pool_int64s","type":7,"ints":[5,6]},)"
            R"({"name":"pool_strings","type":8,"strings":["a","b"]}]})"));

    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
    int64_t max_gram_length = 0;
    int64_t max_skip_count = 0;
    int64_t min_gram_length = 0;
    std::string mode;
    std::vector<float> weights;
    std::vector<int64_t> ngram_counts;
    std::vector<int64_t> ngram_indexes;
    std::vector<int64_t> pool_int64s;
    std::vector<std::string> pool_strings;
    ASSERT_EQ(dest.GetAttr("max_gram_length", max_gram_length), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("max_skip_count", max_skip_count), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("min_gram_length", min_gram_length), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("weights", weights), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("ngram_counts", ngram_counts), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("ngram_indexes", ngram_indexes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("pool_int64s", pool_int64s), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("pool_strings", pool_strings), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_gram_length, 2);
    EXPECT_EQ(max_skip_count, 3);
    EXPECT_EQ(min_gram_length, 1);
    EXPECT_EQ(mode, "TF");
    EXPECT_EQ(weights, (std::vector<float>{1.5F, 2.5F}));
    EXPECT_EQ(ngram_counts, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(ngram_indexes, (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(pool_int64s, (std::vector<int64_t>{5, 6}));
    EXPECT_EQ(pool_strings, (std::vector<std::string>{"a", "b"}));
}

TEST(OnnxTfIdfVectorizerPluginTest, PreservesInt64ArrayBoundaries)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"pool_int64s","type":7,"ints":[-9223372036854775808,9223372036854775807]}]})"));

    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
    std::vector<int64_t> values;
    ASSERT_EQ(dest.GetAttr("pool_int64s", values), ge::GRAPH_SUCCESS);
    EXPECT_EQ(values, (std::vector<int64_t>{std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}));
}

TEST(OnnxTfIdfVectorizerPluginTest, OmittedRepeatedFieldsAreEmpty)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(R"({"attribute":[{"name":"weights","type":6},{"name":"ngram_counts","type":7}]})"));

    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
    std::vector<float> weights{1.0F};
    std::vector<int64_t> ngram_counts{1};
    ASSERT_EQ(dest.GetAttr("weights", weights), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("ngram_counts", ngram_counts), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(weights.empty());
    EXPECT_TRUE(ngram_counts.empty());
}

TEST(OnnxTfIdfVectorizerPluginTest, InvalidEnvelopePreservesExistingAttributes)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("mode", std::string("previous"));
    src.SetAttr("attribute", ge::AscendString("not-json"));

    EXPECT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
    std::string mode;
    ASSERT_EQ(dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(mode, "previous");
}

class OnnxTfIdfVectorizerInvalidValueTest : public testing::TestWithParam<const char *> {};

TEST_P(OnnxTfIdfVectorizerInvalidValueTest, RejectsInvalidAttributeValues)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    const std::string json = std::string(R"({"attribute":[)") + GetParam() + "]}";
    src.SetAttr("attribute", ge::AscendString(json.c_str()));

    EXPECT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
}

INSTANTIATE_TEST_SUITE_P(InvalidInput, OnnxTfIdfVectorizerInvalidValueTest,
                         testing::Values(R"({"name":"max_gram_length","type":2,"i":18446744073709551615})",
                                         R"({"name":"min_gram_length","type":2,"i":1.5})",
                                         R"({"name":"max_skip_count","type":2,"i":"2"})",
                                         R"({"name":"ngram_counts","type":7,"ints":[1,1.5]})",
                                         R"({"name":"ngram_indexes","type":7,"ints":[18446744073709551615]})",
                                         R"({"name":"pool_int64s","type":7,"ints":[true]})",
                                         R"({"name":"pool_int64s","type":7,"ints":["2"]})",
                                         R"({"name":"weights","type":6,"floats":"bad"})",
                                         R"({"name":"mode","type":3,"s":1})"));
