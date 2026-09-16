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

#include "../../../src/framework/tfidf_vectorizer_onnx_plugin.cpp"

class OnnxTfIdfIntegerValidationTest : public testing::TestWithParam<const char *> {};

TEST_P(OnnxTfIdfIntegerValidationTest, RejectsInvalidIntegerWithoutPartialWrites)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("mode", std::string("previous"));
    const std::string json = std::string(R"({"attribute":[{"name":"mode","type":3,"s":"TF"},)") + GetParam() + "]}";
    src.SetAttr("attribute", ge::AscendString(json.c_str()));
    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
    std::string mode;
    ASSERT_EQ(dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(mode, "previous");
}

INSTANTIATE_TEST_SUITE_P(IntegerValidation, OnnxTfIdfIntegerValidationTest,
                         testing::Values(R"({"name":"max_gram_length","type":2,"i":18446744073709551615})",
                                         R"({"name":"min_gram_length","type":2,"i":1.5})",
                                         R"({"name":"max_skip_count","type":2,"i":"2"})",
                                         R"({"name":"ngram_counts","type":7,"ints":[1,1.5]})",
                                         R"({"name":"ngram_indexes","type":7,"ints":[18446744073709551615]})",
                                         R"({"name":"pool_int64s","type":7,"ints":[true]})",
                                         R"({"name":"pool_int64s","type":7,"ints":["2"]})"));

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

TEST(OnnxTfIdfVectorizerPluginTest, MissingAttributeArrayUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString("{}"));
    EXPECT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
}

class OnnxTfIdfVectorizerPluginTestInvalidJson : public testing::TestWithParam<const char *> {};

TEST_P(OnnxTfIdfVectorizerPluginTestInvalidJson, RejectsWithoutChangingExistingAttribute)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("sentinel", int64_t{42});
    src.SetAttr("attribute", ge::AscendString(GetParam()));
    EXPECT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
    int64_t sentinel = 0;
    ASSERT_EQ(dest.GetAttr("sentinel", sentinel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sentinel, 42);
}

INSTANTIATE_TEST_SUITE_P(InvalidEnvelope, OnnxTfIdfVectorizerPluginTestInvalidJson,
                         testing::Values("[]", "null", R"({"attribute":null})", R"({"attribute":[null]})",
                                         R"({"attribute":[{"name":1,"type":2}]})"));

TEST(OnnxTfIdfVectorizerPluginTest, ParsesSupportedAttributeTypes)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"max_gram_length","type":2,"i":3},{"name":"max_skip_count","type":2,"i":1},{"name":"min_gram_length","type":2,"i":2},{"name":"mode","type":3,"s":"TFIDF"},{"name":"weights","type":6,"floats":[0.5,1.5]},{"name":"ngram_counts","type":7,"ints":[1,2]},{"name":"ngram_indexes","type":7,"ints":[3,4]},{"name":"pool_int64s","type":7,"ints":[5,6]},{"name":"pool_strings","type":8,"strings":["a","b"]}]})"));

    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
    int64_t max_gram_length = 0;
    std::string mode;
    std::vector<float> weights;
    std::vector<int64_t> ngram_counts;
    std::vector<std::string> pool_strings;
    ASSERT_EQ(dest.GetAttr("max_gram_length", max_gram_length), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("weights", weights), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("ngram_counts", ngram_counts), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("pool_strings", pool_strings), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_gram_length, 3);
    int64_t max_skip_count = 0;
    int64_t min_gram_length = 0;
    std::vector<int64_t> ngram_indexes;
    std::vector<int64_t> pool_int64s;
    ASSERT_EQ(dest.GetAttr("max_skip_count", max_skip_count), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("min_gram_length", min_gram_length), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("ngram_indexes", ngram_indexes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("pool_int64s", pool_int64s), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_skip_count, 1);
    EXPECT_EQ(min_gram_length, 2);
    EXPECT_EQ(ngram_indexes, (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(pool_int64s, (std::vector<int64_t>{5, 6}));
    EXPECT_EQ(mode, "TFIDF");
    EXPECT_EQ(weights, (std::vector<float>{0.5F, 1.5F}));
    EXPECT_EQ(ngram_counts, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(pool_strings, (std::vector<std::string>{"a", "b"}));
}

TEST(OnnxTfIdfVectorizerPluginTest, RejectsMalformedJsonWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("mode", std::string("previous"));
    src.SetAttr("attribute", ge::AscendString("not-json"));

    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
    std::string mode;
    ASSERT_EQ(dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    EXPECT_EQ(mode, "previous");
}

TEST(OnnxTfIdfVectorizerPluginTest, ParsesMissingValuesAsEmptyOrZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(
            R"({"attribute":[{"name":"max_skip_count","type":2},{"name":"mode","type":3},{"name":"weights","type":6},{"name":"ngram_indexes","type":7},{"name":"pool_strings","type":8}]})"));
    ASSERT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::SUCCESS);
    int64_t max_skip_count = 1;
    std::string mode = "set";
    std::vector<float> weights{1.0F};
    std::vector<int64_t> ngram_indexes{1};
    std::vector<std::string> pool_strings{"set"};
    ASSERT_EQ(dest.GetAttr("max_skip_count", max_skip_count), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("mode", mode), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("weights", weights), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("ngram_indexes", ngram_indexes), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dest.GetAttr("pool_strings", pool_strings), ge::GRAPH_SUCCESS);
    EXPECT_EQ(max_skip_count, 0);
    EXPECT_TRUE(mode.empty());
    EXPECT_TRUE(weights.empty());
    EXPECT_TRUE(ngram_indexes.empty());
    EXPECT_TRUE(pool_strings.empty());
}

TEST(OnnxTfIdfVectorizerPluginTest, RejectsWrongArrayAndScalarTypes)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"weights","type":6,"floats":1}]})"));
    EXPECT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"mode","type":3,"s":1}]})"));
    EXPECT_EQ(domi::ParseParamsTfIdfVectorizer(src, dest), domi::FAILED);
}
