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
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class QuantLightningIndexerTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerTilingArch35 TearDown" << std::endl;
    }
};

// 0
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_0)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 128, 64, 128}, {2, 128, 64, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 512, 1, 128}, {2, 512, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 512, 1}, {2, 512, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 128, 1, 1024}, {2, 128, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    int64_t expectTilingKey = 205860;
    std::string expectTilingData =
        "1099511627778 274877906944 2199023255680 1024 274877906944 0 3 0 ";
    std::vector<size_t> expectWorkspaces = {17039360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 1
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_1)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 16, 128}, {2, 64, 16, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 2048}, {2, 64, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    int64_t expectTilingKey = 1090724386;
    std::string expectTilingData =
        "549755813890 68719476736 2199023255616 8796093024256 274877906960 137438953488 3 0 ";
    std::vector<size_t> expectWorkspaces = {17039360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// When query type is hifloat8, sparse_count must be 2048, but now sparse_count is 1024
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_2)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 16, 128}, {2, 64, 16, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 1024}, {2, 64, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 3
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_3)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 16, 128}, {2, 64, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 2048}, {2, 64, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4096)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    int64_t expectTilingKey = 1090724900;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {17039360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 4
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_4)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int64_t actualSeqKey[] = {256, 512};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{2, 64, 16, 128}, {2, 64, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{32, 16, 1, 128}, {32, 16, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 16}, {2, 64, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 16, 1}, {32, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 64, 1, 2048}, {2, 64, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-2)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// The commercial S1-base-8 path: host and kernel must share one flag and workspace layout.
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_s1_base8_fast_path)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int32_t actualSeqQuery[] = {1024};
    int32_t actualSeqKey[] = {5849};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{1024, 32, 128}, {1024, 32, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{5849, 128, 1, 128}, {5849, 128, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 32}, {1024, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 32}, {1024, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5849, 128, 1}, {5849, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{1, 640}, {1, 640}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{1024, 1, 2048}, {1024, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    uint64_t expectTilingKey = UINT64_MAX;
    std::string expectTilingData = "* * * * * * 4294967299 1 ";
    // The UT fake platform uses 64 AICs, max logical K=640*128 and a 16 MiB library workspace.
    std::vector<size_t> expectWorkspaces = {100663296};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Base-8 is independent of the exact PA table width and supports all four QLI dtype templates.
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_s1_base8_dtype_and_table)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    struct DTypeCase {
        const char *name;
        ge::DataType qkType;
        ge::DataType weightsType;
        ge::DataType scaleType;
        size_t expectedWorkspace;
    };
    const DTypeCase dtypeCases[] = {
        {"fp8_bf16", ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_FLOAT, 33554432},
        {"fp8_fp16", ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT16, 33554432},
        {"hif8_bf16", ge::DT_HIFLOAT8, ge::DT_BF16, ge::DT_FLOAT, 33554432},
        {"int8_fp16", ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, 50331648}
    };
    int32_t actualSeqQuery[] = {1024};
    int32_t actualSeqKey[] = {5849};

    for (const auto &dtypeCase : dtypeCases) {
        SCOPED_TRACE(dtypeCase.name);
        gert::TilingContextPara tilingContextPara(
            "QuantLightningIndexer",
            {
                {{{1024, 32, 128}, {1024, 32, 128}}, dtypeCase.qkType, ge::FORMAT_ND},
                {{{5849, 128, 1, 128}, {5849, 128, 1, 128}}, dtypeCase.qkType, ge::FORMAT_ND},
                {{{1024, 32}, {1024, 32}}, dtypeCase.weightsType, ge::FORMAT_ND},
                {{{1024, 32}, {1024, 32}}, dtypeCase.scaleType, ge::FORMAT_ND},
                {{{5849, 128, 1}, {5849, 128, 1}}, dtypeCase.scaleType, ge::FORMAT_ND},
                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
                {{{1, 128}, {1, 128}}, ge::DT_INT32, ge::FORMAT_ND}
            },
            {
                {{{1024, 1, 2048}, {1024, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}
            },
            {
                {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
            },
            &compileInfo, "Ascend950", 64, 262144, 16384);

        std::string expectTilingData = "* * * * * * 4294967299 1 ";
        std::vector<size_t> expectWorkspaces = {dtypeCase.expectedWorkspace};
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, expectTilingData, expectWorkspaces);
    }
}

// Base-8 is independent of mode/top and supports the validated inward-capacity G/block variants.
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_s1_base8_min_t_and_shape_guards)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int32_t actualSeqQuery[] = {64, 128, 192, 256, 320, 384, 448, 512, 576, 640};
    int32_t actualSeqKey[] = {2049, 2049, 2049, 2049, 2049, 2049, 2049, 2049, 2049, 2049};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{640, 16, 128}, {640, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{40, 64, 1, 128}, {40, 64, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{640, 16}, {640, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{640, 16}, {640, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{40, 64, 1}, {40, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{10, 40}, {10, 40}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{640, 1, 1024}, {640, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    // sparseMode=0 and enableS1Base8FastPath=1 share one packed word; direct capability is independently enabled.
    std::string expectTilingData = "* * * * * * 4294967296 1 ";
    // 16 MiB library workspace + 8 rows * aligned K(2560) * 2B score * 64 AICs.
    std::vector<size_t> expectWorkspaces = {19398656};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, expectTilingData, expectWorkspaces);
}

// Keep low-T INT8-like workloads on base-4 because CA shows no stable benefit below the threshold.
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_s1_base8_below_min_t_falls_back)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int32_t actualSeqQuery[] = {512};
    int32_t actualSeqKey[] = {2049};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{512, 16, 128}, {512, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{40, 64, 1, 128}, {40, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{512, 16}, {512, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 16}, {512, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{40, 64, 1}, {40, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{1, 40}, {1, 40}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{512, 1, 1024}, {512, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    // sparseMode=0, enableS1Base8FastPath=0, enableDirectIndicesFastPath=1.
    std::string expectTilingData = "* * * * * * 0 1 ";
    // 16 MiB library workspace + 4 rows * aligned K(2560) * 4B score * 64 AICs.
    std::vector<size_t> expectWorkspaces = {19398656};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, expectTilingData, expectWorkspaces);
}

// Direct capability covers multiple TND batches; the kernel requires an all-batch runtime match.
TEST_F(QuantLightningIndexerTilingArch35, QuantLightningIndexer_950_tiling_direct_indices_multibatch)
{
    struct QuantLightningIndexerCompileInfo {} compileInfo;
    int32_t actualSeqQuery[] = {256, 512};
    int32_t actualSeqKey[] = {200, 1000};
    gert::TilingContextPara tilingContextPara(
        "QuantLightningIndexer",
        {
            {{{512, 16, 128}, {512, 16, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{32, 64, 1, 128}, {32, 64, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{512, 16}, {512, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 16}, {512, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 64, 1}, {32, 64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQuery},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKey},
            {{{2, 32}, {2, 32}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{512, 1, 1024}, {512, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"key_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"key_dequant_scale_stride0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}
        },
        &compileInfo, "Ascend950", 64, 262144, 16384);

    std::string expectTilingData = "* * * * * * 0 1 ";
    std::vector<size_t> expectWorkspaces = {17825792};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, expectTilingData, expectWorkspaces);
}
