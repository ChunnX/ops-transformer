/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mhc_pre_split_post_res_tiling.cpp
 * \brief UT test cases for MhcPreSplitPostRes tiling function.
 */

#include <iostream>
#include <limits>
#include <gtest/gtest.h>
#include "../../../op_host/mhc_pre_split_post_res_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcPreSplitPostResTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreSplitPostResTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreSplitPostResTiling TearDown" << std::endl;
    }
};

// ==================== 功能用例 (Normal Test Cases) ====================

// ==================== BSND 格式（x 为四维）====================

// ---- MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE (D<=2560, dimLoop=1) ----
// 注意：新策略下 coresPerToken=2 仅在 dimLoop==1 且 totalTokens < coreNum/2 时启用

// BF16, T=B*S=1*4=4, N=4, D=100（不支持，D 不在白名单内）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_100_1_4_000000)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},               // x [B,S,N,D]
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi [N^2+N, N*D]
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // alpha [2]
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // bias [N^2+N]
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},  // h_post
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},  // h_res
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, T=B*S=32*4=128, N=4, D=2560, B=32, S=4
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_2560_32_4_000000)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// FP16, T=B*S=32*4=128, N=4, D=2560, B=32, S=4
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_fp16_128_4_2560_32_4_000000)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ---- MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE (D=5120, dimLoop=2, coresPerToken=1) ----

// BF16, T=B*S=1*4=4, N=4, D=5120, B=1, S=4
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_5120_1_4_000000)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// FP16, T=B*S=1*4=4, N=4, D=5120, B=1, S=4
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_fp16_4_4_5120_1_4_000000)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, T=128, N=4, D=8192（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_8192_32_4_000000)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 8192;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 32768

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// FP16, T=256, N=4, D=16400（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_fp16_256_4_16400_64_4_000000)
{
    int64_t B = 64;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 16400;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 65600

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== TND 格式（x 为三维）====================

// ---- MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE (D<=2560, dimLoop=1) ----
// 注意：新策略下 coresPerToken=2 仅在 dimLoop==1 且 totalTokens < coreNum/2 时启用

// BF16, T=1, N=4, D=100（不支持，D 不在白名单内）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_1_4_100_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},                     // x [T,N,D]
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // alpha
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // bias
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, T=512, N=4, D=2560
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_512_4_2560_000000)
{
    int64_t T = 512;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// FP16, T=512, N=4, D=2560
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_fp16_512_4_2560_000000)
{
    int64_t T = 512;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ---- MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE (D=5120, dimLoop=2, coresPerToken=1) ----

// BF16, T=1, N=4, D=5120
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_1_4_5120_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// FP16, T=1, N=4, D=5120
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_fp16_1_4_5120_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, T=512, N=4, D=8192（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_512_4_8192_000000)
{
    int64_t T = 512;
    int64_t N = 4;
    int64_t D = 8192;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 32768

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// FP16, T=512, N=4, D=16400（不支持，D 不在白名单内）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_fp16_512_4_16400_000000)
{
    int64_t T = 512;
    int64_t N = 4;
    int64_t D = 16400;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 65600

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== 异常用例 (Exception Test Cases) ====================

// ==================== 空 Tensor ====================

// x 为空（shape 维度为 0）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_emptyX_000000)
{
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // x 为空
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4, 4}, {4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// phi 为空
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_emptyPhi_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 64;
    int64_t phiRows = N * N + N;  // 20

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi 为空
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// alpha 为空
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_emptyAlpha_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // alpha 为空
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// bias 为空
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_emptyBias_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // bias 为空
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== Dim 维度数量异常 ====================

// x 维度不为 3 或 4（使用 2 维）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorXDimNum_000000)
{
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{N, D}, {N, D}}, ge::DT_BF16, ge::FORMAT_ND},  // x 为 2 维（错误）
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4, 4}, {4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// phi 维度不为 2（使用 1 维）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorPhiDimNum_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 64;
    int64_t phiRows = N * N + N;  // 20

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi 为 1 维（错误）
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// alpha 维度不为 1（使用 2 维）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorAlphaDimNum_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // alpha 为 2 维（错误）
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// bias 维度不为 1（使用 2 维）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorBiasDimNum_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows, 1}, {phiRows, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},  // bias 为 2 维（错误）
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== Shape 维度大小异常 ====================

// T < MHC_MIN_T（T=0，TND 格式）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorTMin_000000)
{
    int64_t T = 0;  // T < MHC_MIN_T (1)
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// T 无上限（T=512*1024+1，TND 格式，D 取白名单值）→ 放行
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_TLarge_000000)
{
    int64_t T = 512 * 1024 + 1;  // T 不再有上限
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// N 不等于 4（N=8，TND 格式）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorN_000000)
{
    int64_t T = 1;
    int64_t N = 8;  // N 不等于 4（错误）
    int64_t D = 64;
    int64_t phiRows = N * N + N;  // 72（与合法N=4不一致）
    int64_t phiCols = N * D;      // 512

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// D 低于下界 16（D=8，TND 格式）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorDMin_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 8;                // D 不在白名单内
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 32

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// D=16385（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorDMax_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 16385;            // D 不在白名单内
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// D 不在白名单内（D=66，TND 格式）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorDAlign_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 66;               // D 不在白名单内
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 264

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// phi 第 0 维不等于 N^2+N=20（phi dim0=10，错误）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorPhiRows_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 2560;           // 使用合法 D，确保 CheckXValid 通过
    int64_t phiRowsWrong = 10;  // 错误的 phi 行数（应为 20）
    int64_t phiCols = N * D;    // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRowsWrong, phiCols}, {phiRowsWrong, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi dim0 错误
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND},  // bias 使用正确的 phiRows=20
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// phi 第 1 维不等于 N*D（phi dim1=100，错误）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorPhiCols_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 2560;             // 使用合法 D，确保 CheckXValid 通过
    int64_t phiRows = N * N + N;  // 20
    int64_t phiColsWrong = 100;   // 错误的 phi 列数（应为 N*D=10240）

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiColsWrong}, {phiRows, phiColsWrong}}, ge::DT_FLOAT, ge::FORMAT_ND},  // phi dim1 错误
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// alpha shape 不为 [2]（alpha shape=[3]，错误）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorAlphaShape_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 2560;  // 使用合法 D，确保 CheckXValid/CheckPhiValid 通过
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},  // alpha shape=[3]（应为[2]）
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// bias shape 不为 [N^2+N]=20（bias shape=[10]，错误）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_bf16_errorBiasShape_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 2560;  // 使用合法 D，确保 CheckXValid/CheckPhiValid/CheckAlphaValid 通过
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240
    int64_t biasShapeWrong = 10;  // 错误的 bias shape（应为 20）

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{biasShapeWrong}, {biasShapeWrong}}, ge::DT_FLOAT, ge::FORMAT_ND},  // bias shape 错误
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== dtype 数据类型异常 ====================

// x dtype 不为 fp16 或 bf16（使用 float32）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_errorXDtype_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},  // x dtype=float32（错误）
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// phi dtype 不为 float32（使用 fp16）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_errorPhiDtype_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // phi dtype=fp16（错误）
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// alpha dtype 不为 float32（使用 fp16）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_errorAlphaDtype_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // alpha dtype=fp16（错误）
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// bias dtype 不为 float32（使用 fp16）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_errorBiasDtype_000000)
{
    int64_t T = 1;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // bias dtype=fp16（错误）
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== coresPerToken 策略边界用例 ====================
// 新策略：coresPerToken=2 仅在 dimLoop==1 且 totalLength < coreNum/2 时启用
// tilingKey 与 coresPerToken 无关，D<=2560 时固定为 MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE

// ==================== coreNum=48，coreNum/2=24（BSND 格式） ====================

// T=23 < 24：双核触发（D=2560，dimLoop=1）→ SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_corenum48_T23_D2560_dual_core)
{
    int64_t B = 1;
    int64_t S = 23;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;  // 192KB
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// T=24 >= 24：单核（T >= coreNum/2=24）→ SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_corenum48_T24_D2560_single_core)
{
    int64_t B = 1;
    int64_t S = 24;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// T=512 >> 24：单核，验证大 T 在 coreNum=48 下不触发双核 → SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_corenum48_T512_D2560_single_core)
{
    int64_t B = 128;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ==================== coreNum=40，coreNum/2=20（BSND 格式） ====================

// T=19 < 20：双核触发 → SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_corenum40_T19_D2560_dual_core)
{
    int64_t B = 1;
    int64_t S = 19;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// T=20 >= 20：单核（T >= coreNum/2=20）→ SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_corenum40_T20_D2560_single_core)
{
    int64_t B = 1;
    int64_t S = 20;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ==================== coreNum=40，coreNum/2=20（TND 格式） ====================

// T=19 < 20：双核触发 → SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_corenum40_T19_D2560_dual_core)
{
    int64_t T = 19;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// T=20 >= 20：单核（T >= coreNum/2=20）→ SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_corenum40_T20_D2560_single_core)
{
    int64_t T = 20;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// T=512：单核 → SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_TND_bf16_corenum40_T512_D2560_single_core)
{
    int64_t T = 512;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ==================== D 泛化与 norm_eps 校验用例 ====================

// BF16, T=B*S=32*4=128, N=4, D=6144（Pangu V3 典型值，泛化支持：dimLoop=3 → MULTI_TILE）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_6144_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 6144;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 24576

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, T=B*S=32*4=128, N=4, D=7168（Pangu V3 典型值，泛化支持：dimLoop=3 → MULTI_TILE）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_7168_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 7168;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 28672

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, T=128, N=4, D=16384（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_16384_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 16384;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 65536

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, T=128, N=4, D=16（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_16_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 16;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 64

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, T=128, N=4, D=2064（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_2064_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2064;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 8256

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, T=128, N=4, D=2576（D 白名单外，拒绝）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_128_4_2576_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2576;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10304

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, D=5120，norm_eps=-1（非法：必须大于 0）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_norm_eps_invalid)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(-1.0f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, D=5120，norm_eps=0（非法：必须大于 0）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_norm_eps_zero_invalid)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// BF16, D=5120，norm_eps=NaN（非法：必须为有限正数）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_norm_eps_nan_invalid)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(
                             std::numeric_limits<float>::quiet_NaN())},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== MULTI_TILE 双核边界用例 ====================

// BF16, D=5120, T=4（faker coreNum=64，T<32 → 双核/token，blockDim=8，MULTI 双核路径）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_dualcore_4_5120_1_4)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, D=5120, T=64（T>=32 → 单核/token 边界回归，blockDim=64）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_singlecore_64_5120_16_4)
{
    int64_t B = 16;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// ==================== Ascend950 用例 ====================

// 950 BF16, T=B*S=32*4=128, N=4, D=2560 → SINGLE_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_950_BSND_bf16_128_4_2560_32_4)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 950 FP16, T=19, N=4, D=2560（T < coreNum/2）→ SINGLE_TILE + 双核/token
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_950_TND_fp16_T19_D2560)
{
    int64_t T = 19;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 950 BF16, T=B*S=1*4=4, N=4, D=5120 → MULTI_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_950_BSND_bf16_4_4_5120_1_4)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 950 FP16, T=512, N=4, D=5120 → MULTI_TILE
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_950_TND_fp16_T512_D5120)
{
    int64_t T = 512;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{T, N, D}, {T, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 950 BF16, D=100（不支持，D 不在白名单内）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_950_BSND_bf16_4_4_100_1_4)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 100;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 400

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// ==================== phi 仅 post 形态（[N, N*D]，无 h_res）用例 ====================

// BF16, B*S=4, N=4, D=2560, phi=[4, 10240]（仅 post 形态）→ SINGLE_TILE + 双核
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_post_only_phiN_2560)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N;          // 仅 post 形态：phi 行数为 N
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, B*S=4, N=4, D=5120, phi=[4, 20480]（仅 post 形态，MULTI 路径）→ MULTI_TILE + 双核
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_post_only_phiN_5120)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N;          // 仅 post 形态：phi 行数为 N
    int64_t phiCols = N * D;      // 20480

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BF16, phi=[4, 10240] 但 bias=[20]（bias 行数须跟随 phi 行数）→ 拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_post_only_bias_mismatch)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N;          // 仅 post 形态：phi 行数为 N
    int64_t phiCols = N * D;      // 10240
    int64_t biasRows = N * N + N; // 20：与 phi 行数不匹配

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{biasRows}, {biasRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// 非 950 的 soc（如 Ascend310P）仍应被白名单拒绝（本算子仅支持 ascend950）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_soc310p_not_support)
{
    int64_t B = 32;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2560;
    int64_t phiRows = N * N + N;  // 20
    int64_t phiCols = N * D;      // 10240

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend310P");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

// D=2559（白名单下邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_2559_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2559;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=2561（白名单上邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_2561_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 2561;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=5119（白名单下邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_5119_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5119;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=5121（白名单上邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_5121_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5121;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=6143（白名单下邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_6143_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 6143;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=6145（白名单上邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_6145_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 6145;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=7167（白名单下邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_7167_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 7167;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=7169（白名单上邻）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_7169_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 7169;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// D=4096（白名单内部空档）→ D 白名单外拒绝
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_4096_1_4_dwhitelist)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 4096;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}


// Host 注入 C=1（无 sibling 可用，强制单核分组）：tiling 须成功且走 MULTI 路径（统计表 19~28 的 C=1 用例）
TEST_F(MhcPreSplitPostResTiling, mhc_pre_split_post_res_BSND_bf16_4_4_5120_1_4_core1)
{
    int64_t B = 1;
    int64_t S = 4;
    int64_t N = 4;
    int64_t D = 5120;
    int64_t phiRows = N * N + N;
    int64_t phiCols = N * D;

    optiling::MhcPreSplitPostResCompileInfo compileInfo = {};
    compileInfo.coreNum = 1;      // Host 注入 C=1：强制单核
    compileInfo.ubSize = 196608;  // 192KB
    gert::TilingContextPara tilingContextPara("MhcPreSplitPostRes",
        {
            {{{B, S, N, D}, {B, S, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiRows, phiCols}, {phiRows, phiCols}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
        },
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = optiling::MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE;
    std::string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
