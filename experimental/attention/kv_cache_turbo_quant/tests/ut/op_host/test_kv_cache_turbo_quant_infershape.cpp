/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_kv_cache_turbo_quant_infershape.cpp
 * \brief KvCacheTurboQuant Shape 推导单元测试（op_host 层）
 *
 * 覆盖：
 *   - 合法路径：`[T, H, 128]` → `idx [T, H, 128·bits/8]`、`qjl [T, H, 16]`、`norm/gamma [T, H]`
 *     （mse_bits = 2/3/4 的字节数换算）
 *   - 非法路径：rank≠3、head_dim≠128、mse_bits ∉ {2,3,4} 时推导失败
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;

namespace {
constexpr int64_t kHeadDim = 128;

std::vector<gert::InfershapeContextPara::TensorDescription> KvInputs(int64_t tokens, int64_t heads,
                                                                   int64_t headDim = kHeadDim)
{
    return {
        {gert::StorageShape({tokens, heads, headDim}, {tokens, heads, headDim}), DT_BF16, FORMAT_ND},
        {gert::StorageShape({kHeadDim, kHeadDim}, {kHeadDim, kHeadDim}), DT_FLOAT, FORMAT_ND},
        {gert::StorageShape({kHeadDim, kHeadDim}, {kHeadDim, kHeadDim}), DT_FLOAT, FORMAT_ND},
    };
}

std::vector<gert::InfershapeContextPara::TensorDescription> OutputPlaceholders()
{
    return {
        {{{}, {}}, DT_UINT8, FORMAT_ND},
        {{{}, {}}, DT_UINT8, FORMAT_ND},
        {{{}, {}}, DT_BF16, FORMAT_ND},
        {{{}, {}}, DT_BF16, FORMAT_ND},
    };
}

gert::InfershapeContextPara MakePara(int64_t tokens, int64_t heads, int64_t mseBits, int64_t headDim = kHeadDim)
{
    return gert::InfershapeContextPara("KvCacheTurboQuant", KvInputs(tokens, heads, headDim),
                                       OutputPlaceholders(),
                                       {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mseBits)}});
}
} // namespace

class KvCacheTurboQuantInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvCacheTurboQuantInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvCacheTurboQuantInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(KvCacheTurboQuantInfershapeTest, PrefillBits3)
{
    auto para = MakePara(2048, 8, 3);
    const std::vector<std::vector<int64_t>> expect = {{2048, 8, 48}, {2048, 8, 16}, {2048, 8}, {2048, 8}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect);
}

TEST_F(KvCacheTurboQuantInfershapeTest, MseBits2)
{
    auto para = MakePara(2048, 8, 2);
    const std::vector<std::vector<int64_t>> expect = {{2048, 8, 32}, {2048, 8, 16}, {2048, 8}, {2048, 8}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect);
}

TEST_F(KvCacheTurboQuantInfershapeTest, MseBits4)
{
    auto para = MakePara(2048, 8, 4);
    const std::vector<std::vector<int64_t>> expect = {{2048, 8, 64}, {2048, 8, 16}, {2048, 8}, {2048, 8}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect);
}

TEST_F(KvCacheTurboQuantInfershapeTest, DecodeSingleToken)
{
    auto para = MakePara(1, 8, 3);
    const std::vector<std::vector<int64_t>> expect = {{1, 8, 48}, {1, 8, 16}, {1, 8}, {1, 8}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect);
}

TEST_F(KvCacheTurboQuantInfershapeTest, InvalidKvRank2)
{
    gert::InfershapeContextPara para(
        "KvCacheTurboQuant",
        {
            {gert::StorageShape({8, kHeadDim}, {8, kHeadDim}), DT_BF16, FORMAT_ND},
            {gert::StorageShape({kHeadDim, kHeadDim}, {kHeadDim, kHeadDim}), DT_FLOAT, FORMAT_ND},
            {gert::StorageShape({kHeadDim, kHeadDim}, {kHeadDim, kHeadDim}), DT_FLOAT, FORMAT_ND},
        },
        OutputPlaceholders(),
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}});
    ExecuteTestCase(para, GRAPH_FAILED, {});
}

TEST_F(KvCacheTurboQuantInfershapeTest, InvalidHeadDim64)
{
    auto para = MakePara(8, 8, 3, 64);
    ExecuteTestCase(para, GRAPH_FAILED, {});
}

TEST_F(KvCacheTurboQuantInfershapeTest, InvalidMseBits5)
{
    auto para = MakePara(8, 8, 5);
    ExecuteTestCase(para, GRAPH_FAILED, {});
}
