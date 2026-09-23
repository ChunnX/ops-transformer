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

class KvCacheTurboQuantInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- KvCacheTurboQuant InferShape UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- KvCacheTurboQuant InferShape UT TearDown ---" << std::endl;
    }
};

// mse_bits=3: idx 输出 [T, H, 48]
TEST_F(KvCacheTurboQuantInferShapeTest, infershape_mse3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvCacheTurboQuant",
        {{{{2048, 8, 128}, {2048, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{2048, 8, 48}, {2048, 8, 16}, {2048, 8}, {2048, 8}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// mse_bits=2: idx 输出 [T, H, 32]
TEST_F(KvCacheTurboQuantInferShapeTest, infershape_mse2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvCacheTurboQuant",
        {{{{64, 8, 128}, {64, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 8, 32}, {64, 8, 16}, {64, 8}, {64, 8}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// mse_bits=4: idx 输出 [T, H, 64]
TEST_F(KvCacheTurboQuantInferShapeTest, infershape_mse4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvCacheTurboQuant",
        {{{{1, 32, 128}, {1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 64}, {1, 32, 8}, {1, 32}, {1, 32}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// head_dim != 128 -> fail
TEST_F(KvCacheTurboQuantInferShapeTest, infershape_reject_head_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvCacheTurboQuant",
        {{{{1, 8, 64}, {1, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// num_kv_heads 越界 -> fail
TEST_F(KvCacheTurboQuantInferShapeTest, infershape_reject_kv_heads)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvCacheTurboQuant",
        {{{{1, 64, 128}, {1, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// qjl_dim 非 8 对齐 -> fail
TEST_F(KvCacheTurboQuantInferShapeTest, infershape_reject_qjl_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvCacheTurboQuant",
        {{{{1, 8, 128}, {1, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{100, 128}, {100, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
