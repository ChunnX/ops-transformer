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

#include <cstdint>
#include <initializer_list>

#include "../../../op_host/qkv_rms_norm_rope_cache_with_k_scale_contract.h"
#include "../../../op_host/qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

namespace {
using namespace optiling::qkv_rms_norm_rope_cache_with_k_scale;

TensorShapeInfo Shape(std::initializer_list<uint64_t> dims, bool present = true)
{
    TensorShapeInfo shape;
    shape.present = present;
    shape.rank = static_cast<uint32_t>(dims.size());
    auto it = dims.begin();
    if (it != dims.end()) {
        shape.dim0 = *it++;
    }
    if (it != dims.end()) {
        shape.dim1 = *it++;
    }
    if (it != dims.end()) {
        shape.dim2 = *it++;
    }
    if (it != dims.end()) {
        shape.dim3 = *it++;
    }
    return shape;
}

void RefreshShapes(ContractInput &input)
{
    const uint32_t htot = input.numQHeads + input.numKHeads + input.numVHeads;
    input.qkv.shape = Shape({input.totalTokens, htot, input.headDim});
    input.qGamma.shape = Shape({input.headDim});
    input.kGamma.shape = Shape({input.headDim});
    input.cosSin.shape = Shape({input.maxSeqLen, input.headDim});
    input.slotMapping.shape = Shape({input.totalTokens});
    input.kCache.shape = Shape({input.blockNum, input.numKHeads, input.blockSize, input.headDim});
    input.vCache.shape = Shape({input.blockNum, input.numVHeads, input.blockSize, input.headDim});
    input.kScaleCache.shape = Shape({input.blockNum, input.numKHeads, input.blockSize, 1});
    input.cuSeqLen.shape = Shape({static_cast<uint64_t>(input.batch) + 1});
    input.rotation.shape = Shape({input.headDim, input.headDim});
    input.vScale.shape = Shape({input.numVHeads});
}

ContractInput BuildInput(uint32_t totalTokens = 128, uint32_t numQHeads = 16, uint32_t numKHeads = 2,
    uint32_t numVHeads = 2, uint32_t headDim = 128)
{
    ContractInput input;
    input.totalTokens = totalTokens;
    input.batch = 1;
    input.numQHeads = numQHeads;
    input.numKHeads = numKHeads;
    input.numVHeads = numVHeads;
    input.headDim = headDim;
    input.maxSeqLen = 256;
    input.blockNum = 8;
    input.blockSize = 128;

    input.qkv.dtype = ContractDtype::kBf16;
    input.qGamma.dtype = ContractDtype::kFloat;
    input.kGamma.dtype = ContractDtype::kFloat;
    input.cosSin.dtype = ContractDtype::kFloat;
    input.slotMapping.dtype = ContractDtype::kInt32;
    input.kCache.dtype = ContractDtype::kFloat8E4M3fn;
    input.vCache.dtype = ContractDtype::kFloat8E4M3fn;
    input.kScaleCache.dtype = ContractDtype::kFloat;
    input.cuSeqLen.dtype = ContractDtype::kInt32;
    input.rotation.dtype = ContractDtype::kBf16;
    input.vScale.dtype = ContractDtype::kFloat;
    RefreshShapes(input);
    return input;
}

} // namespace

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TypicalHeadTableMatchesDesign)
{
    struct Case {
        uint32_t numQHeads;
        uint32_t numKHeads;
        uint32_t tokenTile;
        uint32_t rowTile;
        uint32_t rowTileAligned;
    };
    const Case cases[] = {
        {16, 2, 14, 252, 256},
        {16, 4, 12, 240, 240},
        {16, 8, 10, 240, 240},
        {32, 2, 6, 204, 208},
        {32, 4, 6, 216, 224},
        {32, 8, 6, 240, 240},
        {64, 2, 2, 132, 144},
        {64, 4, 2, 136, 144},
        {64, 8, 2, 144, 144},
    };

    for (const auto &item : cases) {
        ContractTiling tiling;
        auto input = BuildInput(128, item.numQHeads, item.numKHeads, item.numKHeads);
        EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
        EXPECT_EQ(tiling.tokenTile, item.tokenTile);
        EXPECT_EQ(tiling.tokenTilePerAiv, item.tokenTile / 2);
        EXPECT_EQ(tiling.rowTile, item.rowTile);
        EXPECT_EQ(tiling.rowTileAligned, item.rowTileAligned);
        EXPECT_EQ(tiling.qkHeadTile, item.numQHeads + item.numKHeads);
        EXPECT_EQ(tiling.vHeadTile, item.numKHeads);
        EXPECT_EQ(tiling.dimTile, 128U);
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, RejectsUnsupportedFastPath)
{
    ContractTiling tiling;
    auto input = BuildInput(128, 16, 2, 4);
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kUnsupportedNvNotEqualNk);

    input = BuildInput(128, 16, 2, 2, 64);
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kUnsupportedHeadDim);

    input = BuildInput(128, 0, 2, 2);
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidHeadNums);

    input = BuildInput(128, 128, 8, 8);
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kUnsupportedResource);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, RejectsMissingOptionalButRequiredInputs)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.rotation.shape.present = false;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kMissingRotation);

    input = BuildInput();
    input.vScale.shape.present = false;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kMissingVScale);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, RejectsInvalidShapeAndDtype)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.qkv.shape.dim1 += 1;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidQkvShape);

    input = BuildInput();
    input.kScaleCache.shape.dim3 = 2;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidScaleShape);

    input = BuildInput();
    input.qkv.dtype = ContractDtype::kFloat;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidDtype);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TailTileAndAivSlicesStayContiguous)
{
    ContractTiling tiling;
    auto input = BuildInput(13, 32, 4, 4);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.tokenTile, 6U);
    ASSERT_EQ(CalcTileNum(input.totalTokens, tiling.tokenTile), 3U);

    auto firstTile = MakeTileParam(0, input.totalTokens, tiling);
    EXPECT_EQ(firstTile.tokenOffset, 0U);
    EXPECT_EQ(firstTile.tokenSize, 6U);
    EXPECT_EQ(firstTile.rowSize, tiling.rowTile);

    auto tailTile = MakeTileParam(2, input.totalTokens, tiling);
    EXPECT_EQ(tailTile.tokenOffset, 12U);
    EXPECT_EQ(tailTile.tokenSize, 1U);
    EXPECT_EQ(tailTile.rowSize, tiling.qkHeadTile);

    auto aiv0 = MakeAivSlice(tailTile, tiling.qkHeadTile, 0);
    auto aiv1 = MakeAivSlice(tailTile, tiling.qkHeadTile, 1);
    EXPECT_EQ(aiv0.rowOffset, 0U);
    EXPECT_EQ(aiv0.rowSize, tiling.qkHeadTile);
    EXPECT_EQ(aiv1.rowOffset, tiling.qkHeadTile);
    EXPECT_EQ(aiv1.rowSize, 0U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TileRangeCoversWaveAndEmptyCoreCases)
{
    const auto wave0 = MakeTileRange(65, 0, 32);
    EXPECT_EQ(wave0.begin, 0U);
    EXPECT_EQ(wave0.end, 65U);
    EXPECT_EQ(wave0.stride, 32U);
    EXPECT_EQ(wave0.begin + wave0.stride, 32U);
    EXPECT_EQ(wave0.begin + 2 * wave0.stride, 64U);

    const auto wave31 = MakeTileRange(65, 31, 32);
    EXPECT_EQ(wave31.begin, 31U);
    EXPECT_EQ(wave31.end, 65U);
    EXPECT_EQ(wave31.stride, 32U);
    EXPECT_EQ(wave31.begin + wave31.stride, 63U);

    const auto empty = MakeTileRange(1, 1, 32);
    EXPECT_EQ(empty.begin, 0U);
    EXPECT_EQ(empty.end, 0U);
    EXPECT_EQ(empty.stride, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TilingDataDefaultsAreStable)
{
    optiling::QkvRmsNormRopeCacheWithKScaleTilingData tilingData;
    EXPECT_GT(sizeof(tilingData), 0U);
    EXPECT_EQ(sizeof(tilingData) % sizeof(uint64_t), 0U);
    EXPECT_EQ(tilingData.tokenTile, 0U);

    tilingData.totalTokens = 128;
    tilingData.numQHeads = 16;
    tilingData.tokenTile = 14;
    EXPECT_EQ(tilingData.totalTokens, 128U);
    EXPECT_EQ(tilingData.numQHeads, 16U);
    EXPECT_EQ(tilingData.tokenTile, 14U);
}
