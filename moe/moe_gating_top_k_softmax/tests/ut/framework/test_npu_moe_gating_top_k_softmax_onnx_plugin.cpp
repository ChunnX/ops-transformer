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

#include "../../../framework/npu_moe_gating_top_k_softmax_onnx_plugin.cpp"

namespace {
struct ParseCase {
    const char* name;
    const char* attributes;
    bool success;
    int64_t expected;
};

class OnnxMoeGatingTopKSoftmaxTest : public testing::TestWithParam<ParseCase> {};

TEST_P(OnnxMoeGatingTopKSoftmaxTest, PreservesAttributeSemantics)
{
    const auto& test = GetParam();
    ge::Operator source("source", "TestOp");
    ge::Operator destination("destination", "TestOp");
    if (test.attributes != nullptr) {
        source.SetAttr("attribute", ge::AscendString(test.attributes));
    }
    const auto status = domi::ParseParamsMoeGatingTopKSoftmax(source, destination);
    ASSERT_EQ(status, test.success ? domi::SUCCESS : domi::FAILED);
    int64_t actual = 0;
    if (test.success) {
        ASSERT_EQ(destination.GetAttr("k", actual), ge::GRAPH_SUCCESS);
        EXPECT_EQ(actual, test.expected);
    } else {
        EXPECT_NE(destination.GetAttr("k", actual), ge::GRAPH_SUCCESS);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Attributes, OnnxMoeGatingTopKSoftmaxTest,
    testing::Values(
        ParseCase{"MissingAttribute", nullptr, false, -1},
        ParseCase{"EmptyObject",
            R"json({})json", false, -1},
        ParseCase{"EmptyAttributes",
            R"json({"attribute":[]})json", false, -1},
        ParseCase{"Zero",
            R"json({"attribute":[{"name":"k","type":2,"i":0}]})json", true, 0},
        ParseCase{"One",
            R"json({"attribute":[{"name":"k","type":2,"i":1}]})json", true, 1},
        ParseCase{"Positive",
            R"json({"attribute":[{"name":"k","type":2,"i":8}]})json", true, 8},
        ParseCase{"Negative",
            R"json({"attribute":[{"name":"k","type":2,"i":-2}]})json", true, -2},
        ParseCase{"Sentinel",
            R"json({"attribute":[{"name":"k","type":2,"i":-1}]})json", false, -1},
        ParseCase{"IntMax",
            R"json({"attribute":[{"name":"k","type":2,"i":2147483647}]})json", true, 2147483647},
        ParseCase{"IntMin",
            R"json({"attribute":[{"name":"k","type":2,"i":-2147483648}]})json", true, -2147483648},
        ParseCase{"StringInteger",
            R"json({"attribute":[{"name":"k","type":2,"i":"8"}]})json", true, 8},
        ParseCase{"EnumName",
            R"json({"attribute":[{"name":"k","type":"INT","i":3}]})json", true, 3},
        ParseCase{"OmittedInteger",
            R"json({"attribute":[{"name":"k","type":2}]})json", true, 0},
        ParseCase{"WrongType",
            R"json({"attribute":[{"name":"k","type":1,"f":2}]})json", false, -1},
        ParseCase{"MissingType",
            R"json({"attribute":[{"name":"k","i":2}]})json", false, -1},
        ParseCase{"Unrelated",
            R"json({"attribute":[{"name":"unrelated","type":2,"i":7}]})json", false, -1},
        ParseCase{"LastWins",
            R"json({"attribute":[{"name":"k","type":2,"i":2},{"name":"k","type":2,"i":4}]})json", true, 4},
        ParseCase{"IgnoreLaterWrongType",
            R"json({"attribute":[{"name":"k","type":2,"i":2},{"name":"k","type":1,"f":4}]})json", true, 2},
        ParseCase{"Overflow",
            R"json({"attribute":[{"name":"k","type":2,"i":2147483648}]})json", false, 0},
        ParseCase{"Underflow",
            R"json({"attribute":[{"name":"k","type":2,"i":-2147483649}]})json", false, 0},
        ParseCase{"UnsignedOverflow",
            R"json({"attribute":[{"name":"k","type":2,"i":18446744073709551615}]})json", false, 0},
        ParseCase{"StringOverflow",
            R"json({"attribute":[{"name":"k","type":2,"i":"9223372036854775808"}]})json", false, 0},
        ParseCase{"Float",
            R"json({"attribute":[{"name":"k","type":2,"i":1.5}]})json", false, 0},
        ParseCase{"Boolean",
            R"json({"attribute":[{"name":"k","type":2,"i":true}]})json", false, 0},
        ParseCase{"Null",
            R"json({"attribute":[{"name":"k","type":2,"i":null}]})json", false, 0},
        ParseCase{"InvalidString",
            R"json({"attribute":[{"name":"k","type":2,"i":"3x"}]})json", false, 0},
        ParseCase{"EmptyString",
            R"json({"attribute":[{"name":"k","type":2,"i":""}]})json", false, 0},
        ParseCase{"WhitespaceString",
            R"json({"attribute":[{"name":"k","type":2,"i":" 3"}]})json", false, 0},
        ParseCase{"InvalidJson",
            R"json({)json", false, 0},
        ParseCase{"WrongRoot",
            R"json([])json", false, 0},
        ParseCase{"NullRoot",
            R"json(null)json", false, 0},
        ParseCase{"WrongArray",
            R"json({"attribute":{}})json", false, 0},
        ParseCase{"InvalidElement",
            R"json({"attribute":[null]})json", false, 0}),
    [](const testing::TestParamInfo<ParseCase>& info) { return info.param.name; });
} // namespace
