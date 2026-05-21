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
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{32, 1, 15, 192},{32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 64},{32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 64},{32, 1, 15, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{480},{480}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{125, 10, 1, 192},{125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{125, 10, 1, 128},{125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{32, 1, 15, 128},{32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{125, 10, 1, 192},{125, 10, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{125, 10, 1, 128},{125, 10, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 192},{32, 1, 15, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 1, 15, 128},{32, 1, 15, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5010B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{9, 1, 14, 192},{9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 64},{9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 64},{9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9},{9}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{21, 16, 1, 192},{21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{21, 16, 1, 128},{21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{9, 1, 14, 128},{9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{21, 16, 1, 192},{21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{21, 16, 1, 128},{21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 192},{9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 128},{9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{30, 1, 10, 192},{30, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{30, 1, 10, 64},{30, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{30, 1, 10, 64},{30, 1, 10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{300},{300}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{42, 12, 1, 192},{42, 12, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{42, 12, 1, 128},{42, 12, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{30, 1, 10, 128},{30, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{42, 12, 1, 192},{42, 12, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{42, 12, 1, 128},{42, 12, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{30, 1, 10, 192},{30, 1, 10, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{30, 1, 10, 128},{30, 1, 10, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{38, 1, 3809, 576},{38, 1, 3809, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{228},{228}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{230, 745, 1, 64},{230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{230, 745, 1, 512},{230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{230, 745, 1, 64},{230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{230, 745, 1, 512},{230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 512},{38, 1, 3809, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_5000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{9, 1, 14, 192},{9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 64},{9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 64},{9, 1, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9},{9}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{21, 16, 1, 192},{21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{21, 16, 1, 128},{21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{9, 1, 14, 128},{9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{21, 16, 1, 192},{21, 16, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{21, 16, 1, 128},{21, 16, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 192},{9, 1, 14, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 1, 14, 128},{9, 1, 14, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4010B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{38, 1, 3809, 576},{38, 1, 3809, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{228},{228}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{230, 745, 1, 64},{230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{230, 745, 1, 512},{230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{230, 745, 1, 64},{230, 745, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{230, 745, 1, 512},{230, 745, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 512},{38, 1, 3809, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_4000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{38, 1, 3809, 192},{38, 1, 3809, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 64},{38, 1, 3809, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{228},{228}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{230, 745, 1, 192},{230, 745, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{230, 745, 1, 128},{230, 745, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{38, 1, 3809, 128},{38, 1, 3809, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{230, 745, 1, 192},{230, 745, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{230, 745, 1, 128},{230, 745, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 192},{38, 1, 3809, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{38, 1, 3809, 128},{38, 1, 3809, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BLK_NZ")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3010_SYMMETRIC_QUANT)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3010_ASYMMETRIC_QUANT)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3010, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 576},{64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 512},{64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 576},{64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 512},{64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_3000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2001A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 576},{64, 1, 7, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 64},{64, 1, 7, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 512},{64, 1, 7, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 64},{192, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 512},{192, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{192, 128, 1, 64},{192, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 512},{192, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_2000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1011B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1010B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5011, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000A)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1000B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_tiling_1001B)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 7, 192},{64, 1, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192},{192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7},{64, 7}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 192},{192, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 7, 192},{192, 128, 7, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 7, 128},{64, 1, 7, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend910B",
        48,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 5001, "");
}


TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_tilingB)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128},{128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            kOptionalInput,
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend950",
        64,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tilingA)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 576},{64, 1, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64},{64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512},{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput
        },
        {
            {{{576, 128, 1, 64},{576, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{576, 128, 1, 512},{576, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 512},{64, 1, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend950",
        64,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

TEST_F(KvRmsNormRopeCacheTiling, kv_rms_norm_rope_cache_d_full_load_is_output_kv_with_PA_tilingB)
{
    optiling::KvRmsNormRopeCacheCompileInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "KvRmsNormRopeCache",
        {
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128},{128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 64},{64, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1},{64, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{192, 128, 1, 64},{192, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            kOptionalInput,
            {{{128},{128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{192, 128, 1, 64},{192, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{192, 128, 1, 128},{192, 128, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{64, 1, 1, 192},{64, 1, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 1, 128},{64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA")},
            {"is_output_kv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        "Ascend950",
        64,
        196608);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10000, "");
}

