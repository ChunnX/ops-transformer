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
#include "../../../../op_host/op_tiling/arch35/all_gather_hccl_utils.h"

// HCCL utils 符号对 UT executable 不可见，与 formulaic UT 相同通过 #include 编入本 TU。
#include "../../../../op_host/op_tiling/arch35/all_gather_hccl_utils.cpp"

namespace {

using namespace optiling;

CutResult MakeCutResult(uint64_t longTileLen)
{
    CutResult cut{};
    cut.longTileLen = longTileLen;
    cut.numLongTile = 1;
    cut.shortTileAtBack = true;
    return cut;
}

} // namespace

TEST(AllGatherHcclUtilsTest, CalcMaxTileMNoAdjustment)
{
    auto cut = MakeCutResult(128);
    EXPECT_EQ(CalcMaxTileMFromHcclLimit(cut, 8192, 2, 8, "ut"), HCCL_NO_ADJUSTMENT);
}

TEST(AllGatherHcclUtilsTest, CalcMaxTileMRequiresAdjustment)
{
    auto cut = MakeCutResult(4096);
    uint64_t maxTileM = CalcMaxTileMFromHcclLimit(cut, 8192, 2, 8, "ut");
    EXPECT_GT(maxTileM, HCCL_NO_ADJUSTMENT);
    EXPECT_NE(maxTileM, HCCL_UNSUPPORTED);
    EXPECT_EQ(maxTileM, 2048U);
}

TEST(AllGatherHcclUtilsTest, CalcMaxTileMUnsupportedWhenBelowAlign)
{
    auto cut = MakeCutResult(2048);
    EXPECT_EQ(CalcMaxTileMFromHcclLimit(cut, 65800, 2, 8, "ut"), HCCL_UNSUPPORTED);
}

TEST(AllGatherHcclUtilsTest, CalcMaxTileMZeroSingleRowSize)
{
    auto cut = MakeCutResult(128);
    EXPECT_EQ(CalcMaxTileMFromHcclLimit(cut, 0, 2, 8, "ut"), HCCL_UNSUPPORTED);
}

TEST(AllGatherHcclUtilsTest, SelectOptimalCandidateTileM)
{
    EXPECT_EQ(SelectOptimalCandidateTileM(5000), 2048U);
    EXPECT_EQ(SelectOptimalCandidateTileM(800), 512U);
    EXPECT_EQ(SelectOptimalCandidateTileM(300), 256U);
    EXPECT_EQ(SelectOptimalCandidateTileM(200), 0U);
}

TEST(AllGatherHcclUtilsTest, DetermineFinalTileMWithin63Tiles)
{
    EXPECT_EQ(DetermineFinalTileMWithLimit(1200, 256, 2048, 8192, 2, 8, "ut"), 256U);
}

TEST(AllGatherHcclUtilsTest, DetermineFinalTileMRecalcWhenTooManyTiles)
{
    // 候选列表从大到小取第一个 >= minTileM_align，故返回 2048 而非 512。
    EXPECT_EQ(DetermineFinalTileMWithLimit(16384, 256, 2048, 8192, 2, 8, "ut"), 2048U);
}

TEST(AllGatherHcclUtilsTest, DetermineFinalTileMUsesAlignedFallback)
{
    EXPECT_EQ(DetermineFinalTileMWithLimit(137088, 2048, 65536, 256, 2, 8, "ut"), 2304U);
}

TEST(AllGatherHcclUtilsTest, DetermineFinalTileMUnsupportedWhenExceedsMax)
{
    EXPECT_EQ(DetermineFinalTileMWithLimit(200000, 256, 256, 65535, 2, 8, "ut"), HCCL_UNSUPPORTED);
}

TEST(AllGatherHcclUtilsTest, ApplyTileSplitWithTail)
{
    auto cut = MakeCutResult(4096);
    ApplyTileSplit(cut, 1200, 256, 8192, 2, 8, "ut");
    EXPECT_EQ(cut.longTileLen, 256U);
    EXPECT_EQ(cut.numLongTile, 4U);
    EXPECT_EQ(cut.shortTileLen, 176U);
    EXPECT_EQ(cut.numShortTile, 1U);
}

TEST(AllGatherHcclUtilsTest, ApplyTileSplitPerfectAlign)
{
    auto cut = MakeCutResult(4096);
    ApplyTileSplit(cut, 512, 256, 8192, 2, 8, "ut");
    EXPECT_EQ(cut.shortTileLen, 0U);
    EXPECT_EQ(cut.numShortTile, 0U);
    EXPECT_EQ(cut.numLongTile, 2U);
}

TEST(AllGatherHcclUtilsTest, ApplyTileSplitSingleBlockWhenMLessThanTileM)
{
    auto cut = MakeCutResult(4096);
    ApplyTileSplit(cut, 100, 256, 8192, 2, 8, "ut");
    EXPECT_EQ(cut.longTileLen, 100U);
    EXPECT_EQ(cut.numLongTile, 1U);
    EXPECT_EQ(cut.shortTileLen, 0U);
}

TEST(AllGatherHcclUtilsTest, ApplyTileSplitExceedsLimitEarlyReturn)
{
    auto cut = MakeCutResult(4096);
    ApplyTileSplit(cut, 500, 512, 65535, 2, 8, "ut");
    EXPECT_EQ(cut.longTileLen, 4096U);
}

TEST(AllGatherHcclUtilsTest, AdjustCutResultForHCCLFullPath)
{
    auto cut = MakeCutResult(4096);
    AdjustCutResultForHCCL(cut, 5000, 8192, 2, 8, "ut");
    EXPECT_EQ(cut.longTileLen, 2048U);
    EXPECT_EQ(cut.shortTileLen, 904U);
    EXPECT_EQ(cut.numShortTile, 1U);
}

TEST(AllGatherHcclUtilsTest, AdjustCutResultForHCCLNoOp)
{
    auto cut = MakeCutResult(128);
    AdjustCutResultForHCCL(cut, 512, 8192, 2, 8, "ut");
    EXPECT_EQ(cut.longTileLen, 128U);
}

// mxfp4 走 UINT8 通信（每字节 2 个 fp4），tiling 侧 GetTilingResult 对 FP4 调用
// AdjustCutResultForHCCL(result, m, kValue/2, 1, rank, opName)。本用例验证：
// 传 (K/2, dtypeSize=1) 与字节量等价的 (K/4, dtypeSize=2) 产生完全相同的切分结果。
// 使用 K=32768 使两条路径均触发实际切分调整（非 NoOp），保证断言有意义。
TEST(AllGatherHcclUtilsTest, AdjustCutResultForHCCLMxfp4Equivalence)
{
    const uint64_t mValue = 5000U;
    const uint64_t kValue = 32768U;
    const uint64_t rankDim = 8U;

    // FP4 路径：每行字节量 = (K/2) * 1 = K/2
    auto cutFp4 = MakeCutResult(4096);
    AdjustCutResultForHCCL(cutFp4, mValue, kValue / 2, 1, rankDim, "ut_fp4");

    // 等价对照路径：每行字节量 = (K/4) * 2 = K/2（与 FP4 路径相同）
    auto cutRef = MakeCutResult(4096);
    AdjustCutResultForHCCL(cutRef, mValue, kValue / 4, 2, rankDim, "ut_ref");

    EXPECT_EQ(cutFp4.longTileLen, cutRef.longTileLen);
    EXPECT_EQ(cutFp4.numLongTile, cutRef.numLongTile);
    EXPECT_EQ(cutFp4.shortTileLen, cutRef.shortTileLen);
    EXPECT_EQ(cutFp4.numShortTile, cutRef.numShortTile);
    EXPECT_EQ(cutFp4.shortTileAtBack, cutRef.shortTileAtBack);
    EXPECT_NE(cutFp4.longTileLen, 4096U);
    EXPECT_GT(cutFp4.numLongTile, 0U);
}

TEST(AllGatherHcclUtilsTest, ApplyTileSplitMxfp4Equivalence)
{
    const uint64_t mValue = 1200U;
    const uint64_t kValue = 32768U;
    const uint64_t rankDim = 8U;
    const uint64_t tileM = 256U;

    auto cutFp4 = MakeCutResult(4096);
    ApplyTileSplit(cutFp4, mValue, tileM, kValue / 2, 1, rankDim, "ut_fp4");

    auto cutRef = MakeCutResult(4096);
    ApplyTileSplit(cutRef, mValue, tileM, kValue / 4, 2, rankDim, "ut_ref");

    EXPECT_EQ(cutFp4.longTileLen, cutRef.longTileLen);
    EXPECT_EQ(cutFp4.numLongTile, cutRef.numLongTile);
    EXPECT_EQ(cutFp4.shortTileLen, cutRef.shortTileLen);
    EXPECT_EQ(cutFp4.numShortTile, cutRef.numShortTile);
}
