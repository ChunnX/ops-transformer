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
 * \file test_mhc_sandwich_norm_post_preonly_tiling.cpp
 * \brief UT test cases for MhcSandwichNormPostPreonly tiling
 */

#include <gtest/gtest.h>
#include "./test_mhc_sandwich_norm_post_preonly_tiling.h"

using namespace std;

// ==================== 1. Functional: D-axis tiling ====================

// D=2560: dimLoop=1 (single tile), dimTile=2560, lastDimTile=0
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_tnd_D2560_dimLoop1)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalTokens, 4u);
    EXPECT_EQ(td->headCount, 4u);
    EXPECT_EQ(td->embedDim, 2560u);
    EXPECT_EQ(td->dimTile, 2560u);
    EXPECT_EQ(td->dimLoop, 1u);
    EXPECT_EQ(td->lastDimTile, 0u);
    EXPECT_EQ(td->coresPerToken, 2u);
    EXPECT_EQ(td->myHeadCount, 2u);
    EXPECT_EQ(td->useRmsMid, 1u);
    EXPECT_FLOAT_EQ(td->invD, 1.0f / 2560.0f);
    EXPECT_FLOAT_EQ(td->invK, 1.0f / (4.0f * 2560.0f));
}

// D=5120: dimLoop=2 (two tiles), dimTile=2560, lastDimTile=0
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_tnd_D5120_dimLoop2)
{
    auto para = BuildTilingContextTND(4, 4, 5120, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->embedDim, 5120u);
    EXPECT_EQ(td->dimTile, 2560u);
    EXPECT_EQ(td->dimLoop, 2u);
    EXPECT_EQ(td->lastDimTile, 0u);
    EXPECT_EQ(td->coresPerToken, 2u);
    EXPECT_EQ(td->myHeadCount, 2u);
    EXPECT_FLOAT_EQ(td->invD, 1.0f / 5120.0f);
    EXPECT_FLOAT_EQ(td->invK, 1.0f / (4.0f * 5120.0f));
}

// ==================== 2. Functional: BSND core mapping ====================

// B=2, S=3 -> totalTokens=6, non-divisible by numUnits
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_bsnd_B2S3_coreMapping)
{
    auto para = BuildTilingContextBSND(2, 3, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalTokens, 6u);
    uint32_t numUnits = td->usedCoreNum / td->coresPerToken;
    uint32_t covered = td->blockPivot * td->rowsPerCoreSp +
                       (numUnits - td->blockPivot) * td->rowsPerCore;
    EXPECT_EQ(covered, 6u);
}

// T=16, D=5120: max load, dual-core
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_T16_D5120_maxLoad)
{
    auto para = BuildTilingContextTND(16, 4, 5120, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalTokens, 16u);
    EXPECT_EQ(td->coresPerToken, 2u);
    EXPECT_EQ(td->usedCoreNum % td->coresPerToken, 0u);
    EXPECT_LE(td->usedCoreNum, 32u);
}

// ==================== 3. Functional: gamma_2 optional ====================

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_gamma2_absent)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, false);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->useRmsMid, 0u);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_gamma2_present)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->useRmsMid, 1u);
}

// ==================== 4. Functional: eps attr ====================

// Non-default eps: distinguishes correct read from fallback
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_custom_eps)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true, 1e-5f, 1e-7f);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_FLOAT_EQ(td->normEps, 1e-5f);
    EXPECT_FLOAT_EQ(td->hcEps, 1e-7f);
}

// Exact power-of-2 eps: 2^(-20) ≈ 9.5367e-7, 2^(-19) ≈ 1.9073e-6, 2^(-21) ≈ 4.7684e-7
// These are exactly representable in float — any bit error in attr passing will fail.
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_eps_pow2_nearest)
{
    const float eps20 = 1.0f / (1 << 20);  // 2^(-20) = 9.5367431640625e-7
    const float eps19 = 1.0f / (1 << 19);  // 2^(-19) = 1.9073486328125e-6
    const float eps21 = 1.0f / (1 << 21);  // 2^(-21) = 4.76837158203125e-7

    // 2^(-20) for norm_eps, 2^(-19) for hc_eps
    {
        auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true, eps20, eps19);
        TilingInfo info;
        auto td = RunTilingGetData(para, info);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->normEps, eps20);
        EXPECT_EQ(td->hcEps, eps19);
    }
    // 2^(-21) for both
    {
        auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true, eps21, eps21);
        TilingInfo info;
        auto td = RunTilingGetData(para, info);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->normEps, eps21);
        EXPECT_EQ(td->hcEps, eps21);
    }
}

// check return_h_in_f32 param
// returnHInF32 is stored as int8_t in the tiling buffer (TILING_DATA_FIELD_DEF does not support bool)
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_return_h_in_f32)
{
    {
        auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true, 1e-6f, 1e-6f, true);
        TilingInfo info;
        auto td = RunTilingGetData(para, info);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->returnHInF32, static_cast<int8_t>(1));
    }
    {
        auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true, 1e-6f, 1e-6f, false);
        TilingInfo info;
        auto td = RunTilingGetData(para, info);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->returnHInF32, static_cast<int8_t>(0));
    }
}

// Default eps
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_default_eps)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_FLOAT_EQ(td->normEps, 1e-6f);
    EXPECT_FLOAT_EQ(td->hcEps, 1e-6f);
    EXPECT_EQ(td->returnHInF32, static_cast<int8_t>(0));
}

// ==================== 5. Functional: workspace ====================

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_workspace_size)
{
    auto para = BuildTilingContextTND(1, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);

    ASSERT_FALSE(info.workspaceSizes.empty());
    int64_t ws = info.workspaceSizes[0];
    EXPECT_GE(ws, 64);

    // Workspace = sysWs (GetLibApiWorkSpaceSize) + dataWs + syncWs
    // sysWs is platform-dependent (~16MB), so check GE not EQ
    const uint32_t coreNumAiv = 64;
    size_t pairWsFloats = 16 + (size_t)4 * 5120;
    size_t singleWsFloats = (size_t)(4 + 1) * 5120;
    size_t wsPerUnit = std::max(pairWsFloats, singleWsFloats);
    size_t dataWs = coreNumAiv * wsPerUnit * sizeof(float);
    size_t syncWs = coreNumAiv * 8 * 1 * sizeof(int32_t);
    EXPECT_GE(ws, (int64_t)(dataWs + syncWs));
}

// ==================== 6. Functional: dtype / format coverage ====================

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_tnd_D2560_fp16)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_FLOAT16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalTokens, 4u);
    EXPECT_EQ(td->embedDim, 2560u);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_bsnd_B1_S4_D2560)
{
    auto para = BuildTilingContextBSND(1, 4, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalTokens, 4u);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_bsnd_B2_S2_D5120_fp16)
{
    auto para = BuildTilingContextBSND(2, 2, 4, 5120, ge::DT_FLOAT16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalTokens, 4u);
    EXPECT_EQ(td->embedDim, 5120u);
}

// ==================== 7. Functional: T-axis boundary ====================

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_T1_minimum)
{
    auto para = BuildTilingContextTND(1, 4, 2560, ge::DT_BF16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalTokens, 1u);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_T16_maximum)
{
    auto para = BuildTilingContextTND(16, 4, 2560, ge::DT_FLOAT16, true);
    TilingInfo info;
    auto td = RunTilingGetData(para, info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalTokens, 16u);
}

// ==================== 8. Abnormal inputs (expect GRAPH_FAILED) ====================

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_T0)
{
    auto para = BuildTilingContextTND(0, 4, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

// T is within [1, HOST_MAX_TOKENS=1048576]
TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_T1M)
{
    auto para = BuildTilingContextTND(1048577, 4, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_N3)
{
    auto para = BuildTilingContextTND(4, 3, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_N5)
{
    auto para = BuildTilingContextTND(4, 5, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_D1024)
{
    auto para = BuildTilingContextTND(4, 4, 1024, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_D4096)
{
    auto para = BuildTilingContextTND(4, 4, 4096, ge::DT_FLOAT16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_bsnd_B0)
{
    auto para = BuildTilingContextBSND(0, 4, 4, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_bsnd_S0)
{
    auto para = BuildTilingContextBSND(1, 0, 4, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_tiling_bsnd_BS_exceed_1M)
{
    auto para = BuildTilingContextBSND(1024, 1025, 4, 2560, ge::DT_BF16, true);
    RunTilingTest(para, true);
}

// ==================== 9. Abnormal: eps < 0 ====================

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_eps_negative_normEps)
{
    auto para = BuildTilingContextTND(4, 4, 2560, ge::DT_BF16, true, -1e-6f, 1e-6f);
    RunTilingTest(para, true);
}

TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_eps_negative_hcEps)
{
    auto para = BuildTilingContextBSND(1, 4, 4, 2560, ge::DT_BF16, true, 1e-6f, -1e-6f);
    RunTilingTest(para, true);
}

// ==================== 10. Abnormal: format consistency ====================

// h_out 2D [T,D] but residual 4D [B,S,N,D] — ndim mismatch (expect 3D residual)
TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_format_mixed_hout2d_res4d)
{
    auto para = BuildMixed_Hout2D_Res4D();
    RunTilingTest(para, true);
}

// h_out 3D [B,S,D] but residual 3D [T,N,D] — ndim mismatch (expect 4D residual)
TEST_F(MhcSandwichNormPostPreonlyTiling, abnormal_format_mixed_hout3d_res3d)
{
    auto para = BuildMixed_Hout3D_Res3D();
    RunTilingTest(para, true);
}

// ==================== 11. Functional: format equivalence ====================

// Same logical shape in TND and BSND should produce identical TilingData
TEST_F(MhcSandwichNormPostPreonlyTiling, tiling_format_equivalence)
{
    auto paraTND = BuildTilingContextTND(6, 4, 2560, ge::DT_BF16, true);
    TilingInfo infoTND;
    auto tdTND = RunTilingGetData(paraTND, infoTND);
    ASSERT_NE(tdTND, nullptr);

    auto paraBSND = BuildTilingContextBSND(2, 3, 4, 2560, ge::DT_BF16, true);
    TilingInfo infoBSND;
    auto tdBSND = RunTilingGetData(paraBSND, infoBSND);
    ASSERT_NE(tdBSND, nullptr);

    EXPECT_EQ(tdTND->totalTokens, tdBSND->totalTokens);
    EXPECT_EQ(tdTND->headCount, tdBSND->headCount);
    EXPECT_EQ(tdTND->embedDim, tdBSND->embedDim);
    EXPECT_EQ(tdTND->usedCoreNum, tdBSND->usedCoreNum);
    EXPECT_EQ(tdTND->rowsPerCore, tdBSND->rowsPerCore);
    EXPECT_EQ(tdTND->rowsPerCoreSp, tdBSND->rowsPerCoreSp);
    EXPECT_EQ(tdTND->blockPivot, tdBSND->blockPivot);
    EXPECT_EQ(tdTND->dimTile, tdBSND->dimTile);
    EXPECT_EQ(tdTND->dimLoop, tdBSND->dimLoop);
    EXPECT_EQ(tdTND->lastDimTile, tdBSND->lastDimTile);
    EXPECT_EQ(tdTND->phiRows, tdBSND->phiRows);
    EXPECT_EQ(tdTND->phiCols, tdBSND->phiCols);
    EXPECT_EQ(tdTND->normEps, tdBSND->normEps);
    EXPECT_EQ(tdTND->hcEps, tdBSND->hcEps);
    EXPECT_EQ(tdTND->invD, tdBSND->invD);
    EXPECT_EQ(tdTND->invK, tdBSND->invK);
    EXPECT_EQ(tdTND->useRmsMid, tdBSND->useRmsMid);
    EXPECT_EQ(tdTND->coresPerToken, tdBSND->coresPerToken);
    EXPECT_EQ(tdTND->myHeadCount, tdBSND->myHeadCount);
    EXPECT_EQ(tdTND->maxTokensPerPair, tdBSND->maxTokensPerPair);
    EXPECT_EQ(tdTND->returnHInF32, tdBSND->returnHInF32);
}
