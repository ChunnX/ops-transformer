/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

using namespace std;

namespace {
const gert::InfershapeContextPara::TensorDescription kOptionalInput = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};

std::vector<gert::InfershapeContextPara::OpAttr> CommonAttrs()
{
    return {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    };
}
}

class KvRmsNormRopeCache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvRmsNormRopeCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvRmsNormRopeCache Proto Test TearDown" << std::endl;
    }
};

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershapeA)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 64}, {64, 288, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 512}, {64, 288, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 288, 1, 64}, {64, 288, 1, 512}, {64, 1, 1, 64}, {64, 1, 1, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KvRmsNormRopeCache, KvRmsNormRopeCache_infershapeB)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192}, {64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 288, 1, 192}, {64, 288, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 288, 1, 128}, {64, 288, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs());

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {64, 288, 1, 192}, {64, 288, 1, 128}, {64, 1, 1, 192}, {64, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
