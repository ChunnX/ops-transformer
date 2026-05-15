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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/kv_rms_norm_rope_cache_tiling.h"

using namespace std;

namespace {
const gert::TilingContextPara::TensorDescription kOptionalInput = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};

std::vector<gert::TilingContextPara::OpAttr> CommonAttrs()
{
    return {
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
        {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    };
}
}

class KvRmsNormRopeCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
};

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576}, {64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
        },
        {
            {{{576, 128, 1, 64}, {576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512}, {576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64}, {64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512}, {64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs(),
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{32, 1, 15, 192}, {32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 64}, {32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 64}, {32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{480}, {480}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{125, 10, 1, 192}, {125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{125, 10, 1, 128}, {125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{32, 1, 15, 128}, {32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{125, 10, 1, 192}, {125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{125, 10, 1, 128}, {125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 192}, {32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 128}, {32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        CommonAttrs(),
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}
