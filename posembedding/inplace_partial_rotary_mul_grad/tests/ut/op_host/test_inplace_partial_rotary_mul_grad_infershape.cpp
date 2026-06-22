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
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class InplacePartialRotaryMulGrad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplacePartialRotaryMulGrad InferShape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InplacePartialRotaryMulGrad InferShape Test TearDown" << std::endl;
    }
};

TEST_F(InplacePartialRotaryMulGrad, infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplacePartialRotaryMulGrad",
        {
            // input info: dy, cos, sin  (x optional, not provided)
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info: dx, dcos, dsin
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attrs
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}, {1, 64, 1, 64}, {1, 64, 1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InplacePartialRotaryMulGrad, infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{1, 32, 4, 128}, {1, 32, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 4, 128}, {1, 1, 1, 128}, {1, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InplacePartialRotaryMulGrad, infer_shape_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 64, 8, 64}, {2, 64, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 8, 64}, {2, 1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 8, 64}, {2, 1, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 64, 8, 64}, {2, 1, 8, 64}, {2, 1, 8, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InplacePartialRotaryMulGrad, infer_shape_mixed_dtype)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{1, 32, 4, 64}, {1, 32, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 4, 64}, {1, 32, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 32, 4, 64}, {1, 32, 4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 4, 64}, {1, 32, 4, 64}, {1, 32, 4, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InplacePartialRotaryMulGrad, infer_shape_full_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 128})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 64, 8, 128}, {1, 1, 1, 128}, {1, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
