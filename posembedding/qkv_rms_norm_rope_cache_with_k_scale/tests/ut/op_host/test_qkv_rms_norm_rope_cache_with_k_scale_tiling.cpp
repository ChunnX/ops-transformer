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

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "../../../op_host/qkv_rms_norm_rope_cache_with_k_scale_contract.h"
#include "../../../op_host/qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

namespace {
using namespace optiling::qkv_rms_norm_rope_cache_with_k_scale;

constexpr uint32_t kDoubleBufferNum = 2U;

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

TensorStrideInfo Stride(std::initializer_list<uint64_t> strides, bool present = true)
{
    TensorStrideInfo stride;
    stride.present = present;
    stride.rank = static_cast<uint32_t>(strides.size());
    auto it = strides.begin();
    if (it != strides.end()) {
        stride.stride0 = *it++;
    }
    if (it != strides.end()) {
        stride.stride1 = *it++;
    }
    if (it != strides.end()) {
        stride.stride2 = *it++;
    }
    if (it != strides.end()) {
        stride.stride3 = *it++;
    }
    return stride;
}

void RefreshShapes(ContractInput &input)
{
    const uint64_t htot = input.numQHeads + input.numKHeads + input.numVHeads;
    input.qkv.shape = Shape({input.totalTokens, htot, input.headDim});
    input.qGamma.shape = Shape({input.headDim});
    input.kGamma.shape = Shape({input.headDim});
    input.cosSin.shape = Shape({input.maxSeqLen, input.headDim});
    input.slotMapping.shape = Shape({input.totalTokens});
    input.kCache.shape = Shape({input.blockNum, input.numKHeads, input.blockSize, input.headDim});
    input.kCache.stride = Stride({input.numKHeads * input.blockSize * input.headDim, input.blockSize * input.headDim,
        input.headDim, 1});
    input.vCache.shape = Shape({input.blockNum, input.numVHeads, input.blockSize, input.headDim});
    input.vCache.stride = Stride({input.numVHeads * input.blockSize * input.headDim, input.blockSize * input.headDim,
        input.headDim, 1});
    input.kScaleCache.shape = Shape({input.blockNum, input.numKHeads, input.blockSize, 1});
    input.kScaleCache.stride = Stride({input.numKHeads * input.blockSize, input.blockSize, 1, 1});
    input.cuSeqLen.shape = Shape({static_cast<uint64_t>(input.batch) + 1});
    input.qRotation.shape = Shape({input.headDim, input.headDim});
    input.kRotation.shape = Shape({input.headDim, input.headDim});
    input.vScale.shape = Shape({input.numVHeads});
}

ContractInput BuildInput(uint64_t totalTokens = 128, uint64_t numQHeads = 16, uint64_t numKHeads = 2,
    uint64_t numVHeads = 2, uint64_t headDim = 128)
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
    input.aicNum = 32;

    input.qkv.dtype = ContractDtype::kBf16;
    input.qGamma.dtype = ContractDtype::kFloat;
    input.kGamma.dtype = ContractDtype::kFloat;
    input.cosSin.dtype = ContractDtype::kFloat;
    input.slotMapping.dtype = ContractDtype::kInt32;
    input.kCache.dtype = ContractDtype::kFloat8E4M3fn;
    input.vCache.dtype = ContractDtype::kFloat8E4M3fn;
    input.kScaleCache.dtype = ContractDtype::kFloat;
    input.cuSeqLen.dtype = ContractDtype::kInt32;
    input.qRotation.dtype = ContractDtype::kBf16;
    input.kRotation.dtype = ContractDtype::kBf16;
    input.vScale.dtype = ContractDtype::kFloat;
    RefreshShapes(input);
    return input;
}

struct CoreTokenRange {
    uint64_t begin = 0;
    uint64_t end = 0;
};

CoreTokenRange MakeCoreTokenRangeForTest(uint64_t totalTokens, uint64_t cubeIndex, const ContractTiling &tiling)
{
    CoreTokenRange range;
    if (tiling.coreGroupNum == 0 || tiling.coreTokenTile == 0 || cubeIndex >= tiling.coreGroupNum ||
        totalTokens == 0) {
        return range;
    }

    range.begin = cubeIndex * tiling.coreTokenTile;
    if (range.begin >= totalTokens) {
        range.begin = totalTokens;
        range.end = totalTokens;
        return range;
    }
    range.end = std::min(totalTokens, range.begin + tiling.coreTokenTile);
    return range;
}

uint32_t MakeOutputBufferIdForTest(uint64_t outBufferUseId)
{
    return static_cast<uint32_t>(outBufferUseId & (kDoubleBufferNum - 1U));
}

struct OutputUsePair {
    uint64_t preprocess = 0;
    uint64_t qk = 0;
};

class OutputUseTrackerForTest {
public:
    OutputUsePair NextTile()
    {
        OutputUsePair uses;
        uses.preprocess = nextUseId_++;
        uses.qk = nextUseId_++;
        return uses;
    }

private:
    uint64_t nextUseId_ = 0;
};

bool HasPreprocessWorkForTest(const AivSlice &slice)
{
    return slice.tokenSize != 0U && slice.rowSize != 0U;
}

TileParam MakeTileParamForTest(uint64_t tileId, uint64_t tokenOffset, uint64_t tokenSize, uint64_t totalTokens,
    const ContractTiling &tiling)
{
    TileParam tile;
    tile.tileId = tileId;
    tile.bufferId = tileId & 1U;
    tile.tokenOffset = tokenOffset;
    if (tile.tokenOffset >= totalTokens || tokenSize == 0) {
        return tile;
    }
    tile.tokenSize = std::min({tokenSize, tiling.tokenTile, totalTokens - tile.tokenOffset});
    tile.qkHeadOffset = 0;
    tile.qkHeadSize = tiling.qkHeadTile;
    tile.vHeadOffset = 0;
    tile.vHeadSize = tiling.vHeadTile;
    tile.rowOffset = 0;
    tile.rowSize = tile.tokenSize * tiling.qkHeadTile;
    return tile;
}

struct TileSyncCounts {
    uint64_t aivAReadySets = 0;
    uint64_t aicAReadyWaits = 0;
    uint64_t aicCReadySets = 0;
    uint64_t aivCReadyWaits = 0;
    uint64_t aivCConsumedSets = 0;
    uint64_t aicCConsumedWaits = 0;
    uint64_t coveredTokens = 0;
    uint64_t coveredRows = 0;
    uint64_t emptyAivSlices = 0;
};

TileSyncCounts CountFixedTileSyncProtocol(const TileParam &tile, uint64_t qkHeadTile)
{
    TileSyncCounts counts;
    if (tile.tokenSize == 0) {
        return counts;
    }

    counts.aicAReadyWaits = kAivPerAic;
    counts.aicCReadySets = kAivPerAic;
    counts.aicCConsumedWaits = kAivPerAic;
    for (uint64_t aivLocalId = 0; aivLocalId < kAivPerAic; ++aivLocalId) {
        const auto slice = MakeAivSlice(tile, qkHeadTile, aivLocalId);
        ++counts.aivAReadySets;
        ++counts.aivCReadyWaits;
        ++counts.aivCConsumedSets;
        counts.coveredTokens += slice.tokenSize;
        counts.coveredRows += slice.rowSize;
        if (slice.tokenSize == 0) {
            ++counts.emptyAivSlices;
        }
    }
    return counts;
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
        {16, 2, 6, 108, 112},
        {16, 4, 6, 120, 128},
        {16, 8, 4, 96, 96},
        {32, 2, 2, 68, 80},
        {32, 4, 2, 72, 80},
        {32, 8, 2, 80, 80},
    };

    for (const auto &item : cases) {
        ContractTiling tiling;
        auto input = BuildInput(128, item.numQHeads, item.numKHeads, item.numKHeads);
        EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
        EXPECT_EQ(tiling.coreTokenTile, 4U);
        EXPECT_EQ(tiling.coreGroupNum, 32U);
        EXPECT_EQ(tiling.tokenTile, item.tokenTile);
        EXPECT_EQ(tiling.tokenTilePerAiv, item.tokenTile / 2);
        EXPECT_EQ(tiling.rowTile, item.rowTile);
        EXPECT_EQ(tiling.rowTileAligned, item.rowTileAligned);
        EXPECT_EQ(tiling.qkHeadTile, item.numQHeads + item.numKHeads);
        EXPECT_EQ(tiling.vHeadTile, item.numKHeads);
        EXPECT_EQ(tiling.dimTile, 128U);
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, QkAlignedL0aRowsCanReduceTokenTile)
{
    ContractTiling tiling;
    auto input = BuildInput(128, 14, 4, 4);
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    EXPECT_EQ(tiling.tokenTile, 6U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 3U);
    EXPECT_EQ(tiling.rowTile, 108U);
    EXPECT_EQ(tiling.rowTileAligned, 112U);
    EXPECT_LE(AlignUp(tiling.tokenTile * input.numQHeads, 16) +
            AlignUp(tiling.tokenTile * input.numKHeads, 16),
        kL0aMaxRows);
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

    input = BuildInput(128, 64, 2, 2);
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    EXPECT_EQ(tiling.tokenTile, 1U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_EQ(tiling.rowTile, 66U);
    EXPECT_EQ(tiling.rowTileAligned, 80U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, AcceptsT512Q64K8V8Aic32)
{
    ContractTiling tiling;
    auto input = BuildInput(512, 64, 8, 8);
    input.aicNum = 32;
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    EXPECT_EQ(tiling.tokenTile, 1U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_EQ(tiling.rowTile, 72U);
    EXPECT_EQ(tiling.rowTileAligned, 80U);
    EXPECT_EQ(tiling.qkHeadTile, 72U);
    EXPECT_EQ(tiling.vHeadTile, 8U);
    EXPECT_EQ(tiling.coreTokenTile, 16U);
    EXPECT_EQ(tiling.coreGroupNum, 32U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, T1024Q16K2V2Aic32KeepsQCopyBlockLenBounded)
{
    ContractTiling tiling;
    auto input = BuildInput(1024, 16, 2, 2);
    input.aicNum = 32;
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    EXPECT_EQ(tiling.tokenTile, 6U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 3U);
    EXPECT_EQ(tiling.rowTile, 108U);
    EXPECT_EQ(tiling.rowTileAligned, 112U);
    EXPECT_EQ(tiling.coreTokenTile, 32U);
    EXPECT_EQ(tiling.coreGroupNum, 32U);

    const uint64_t qRowsPerAiv = tiling.tokenTilePerAiv * input.numQHeads;
    const uint64_t qCopyBlockLen = qRowsPerAiv * input.headDim;
    EXPECT_EQ(qRowsPerAiv, 48U);
    EXPECT_EQ(qCopyBlockLen, 6144U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, RejectsMissingOptionalButRequiredInputs)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.qRotation.shape.present = false;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kMissingQRotation);

    input = BuildInput();
    input.kRotation.shape.present = false;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kMissingKRotation);

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
    input.qRotation.shape.dim1 += 1;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidQRotationShape);

    input = BuildInput();
    input.kRotation.shape.dim0 += 1;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidKRotationShape);

    input = BuildInput();
    input.qkv.dtype = ContractDtype::kFloat;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidDtype);

    input = BuildInput();
    input.qRotation.dtype = ContractDtype::kFloat;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidDtype);

    input = BuildInput();
    input.kRotation.dtype = ContractDtype::kFloat;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidDtype);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, ContiguousCacheStridesAreReported)
{
    ContractTiling tiling;
    auto input = BuildInput();
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    EXPECT_EQ(tiling.kCacheStrideBlock, input.numKHeads * input.blockSize * input.headDim);
    EXPECT_EQ(tiling.kCacheStrideHead, input.blockSize * input.headDim);
    EXPECT_EQ(tiling.kCacheStrideToken, input.headDim);
    EXPECT_EQ(tiling.vCacheStrideBlock, input.numVHeads * input.blockSize * input.headDim);
    EXPECT_EQ(tiling.vCacheStrideHead, input.blockSize * input.headDim);
    EXPECT_EQ(tiling.vCacheStrideToken, input.headDim);
    EXPECT_EQ(tiling.kScaleCacheStrideBlock, input.numKHeads * input.blockSize);
    EXPECT_EQ(tiling.kScaleCacheStrideHead, input.blockSize);
    EXPECT_EQ(tiling.kScaleCacheStrideToken, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, AcceptsPaddedCacheStrides)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.kCache.stride = Stride({45000, 21000, 160, 1});
    input.vCache.stride = Stride({45000, 21000, 160, 1});
    input.kScaleCache.stride = Stride({1024, 256, 2, 1});
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    EXPECT_EQ(tiling.kCacheStrideBlock, 45000U);
    EXPECT_EQ(tiling.kCacheStrideHead, 21000U);
    EXPECT_EQ(tiling.kCacheStrideToken, 160U);
    EXPECT_EQ(tiling.vCacheStrideBlock, 45000U);
    EXPECT_EQ(tiling.vCacheStrideHead, 21000U);
    EXPECT_EQ(tiling.vCacheStrideToken, 160U);
    EXPECT_EQ(tiling.kScaleCacheStrideBlock, 1024U);
    EXPECT_EQ(tiling.kScaleCacheStrideHead, 256U);
    EXPECT_EQ(tiling.kScaleCacheStrideToken, 2U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, RejectsMismatchedKvCacheStrides)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.kCache.stride = Stride({45000, 21000, 160, 1});
    input.vCache.stride = Stride({45000, 21000, 192, 1});
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidCacheStride);

    input = BuildInput();
    input.kCache.stride = Stride({45000, 21000, 160, 1});
    input.vCache.stride = Stride({52000, 21000, 160, 1});
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidCacheStride);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, RejectsInvalidCacheStrides)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.kCache.stride.stride3 = 2;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidCacheStride);

    input = BuildInput();
    input.vCache.stride.stride1 = input.headDim - 1;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidCacheStride);

    input = BuildInput();
    input.kCache.stride.rank = 3;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidCacheStride);

    input = BuildInput();
    input.kScaleCache.stride.stride3 = 2;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidScaleStride);

    input = BuildInput();
    input.kScaleCache.stride.stride1 = 0;
    EXPECT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kInvalidScaleStride);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TailTileAndAivSlicesStayContiguous)
{
    ContractTiling tiling;
    auto input = BuildInput(13, 16, 2, 2);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.tokenTile, 6U);
    ASSERT_EQ(CalcTileNum(input.totalTokens, tiling.tokenTile), 3U);

    auto firstTile = MakeTileParamForTest(0, 0, tiling.tokenTile, input.totalTokens, tiling);
    EXPECT_EQ(firstTile.tokenOffset, 0U);
    EXPECT_EQ(firstTile.tokenSize, 6U);
    EXPECT_EQ(firstTile.rowSize, tiling.rowTile);

    auto tailTile = MakeTileParamForTest(0, 12, 1, input.totalTokens, tiling);
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

TEST(QkvRmsNormRopeCacheWithKScaleTiling, AivSlicesUseActualTailRows)
{
    ContractTiling tiling;
    auto input = BuildInput(11, 16, 2, 2);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.tokenTile, 6U);

    auto fullTile = MakeTileParamForTest(0, 0, tiling.tokenTile, input.totalTokens, tiling);
    auto fullAiv0 = MakeAivSlice(fullTile, tiling.qkHeadTile, 0);
    auto fullAiv1 = MakeAivSlice(fullTile, tiling.qkHeadTile, 1);
    OutputUseTrackerForTest outputUses;
    const auto fullUses = outputUses.NextTile();
    EXPECT_EQ(fullTile.bufferId, 0U);
    EXPECT_EQ(fullUses.preprocess, 0U);
    EXPECT_EQ(fullUses.qk, 1U);
    EXPECT_EQ(MakeOutputBufferIdForTest(fullUses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(fullUses.qk), 1U);
    EXPECT_EQ(fullAiv0.tokenOffset, 0U);
    EXPECT_EQ(fullAiv0.tokenSize, tiling.tokenTilePerAiv);
    EXPECT_EQ(fullAiv1.tokenOffset, tiling.tokenTilePerAiv);
    EXPECT_EQ(fullAiv1.tokenSize, tiling.tokenTilePerAiv);
    EXPECT_EQ(fullAiv1.rowOffset, fullAiv0.rowSize);
    EXPECT_EQ(fullAiv0.rowSize + fullAiv1.rowSize, fullTile.rowSize);

    auto tailTile = MakeTileParamForTest(1, 6, 5, input.totalTokens, tiling);
    auto tailAiv0 = MakeAivSlice(tailTile, tiling.qkHeadTile, 0);
    auto tailAiv1 = MakeAivSlice(tailTile, tiling.qkHeadTile, 1);
    const auto tailUses = outputUses.NextTile();
    ASSERT_EQ(tailTile.tokenOffset, 6U);
    ASSERT_EQ(tailTile.tokenSize, 5U);
    EXPECT_EQ(tailTile.bufferId, 1U);
    EXPECT_EQ(tailUses.preprocess, 2U);
    EXPECT_EQ(tailUses.qk, 3U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tailUses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tailUses.qk), 1U);
    EXPECT_EQ(tailTile.rowSize, 5U * tiling.qkHeadTile);
    EXPECT_EQ(tailAiv0.tokenOffset, 6U);
    EXPECT_EQ(tailAiv0.tokenSize, 3U);
    EXPECT_EQ(tailAiv0.rowOffset, 0U);
    EXPECT_EQ(tailAiv0.rowSize, 3U * tiling.qkHeadTile);
    EXPECT_EQ(tailAiv1.tokenOffset, tailTile.tokenOffset + CalcCubeHalfTokenSize(tailTile.tokenSize));
    EXPECT_EQ(tailAiv1.tokenSize, 2U);
    EXPECT_EQ(tailAiv1.rowOffset, tailAiv0.rowSize);
    EXPECT_EQ(tailAiv1.rowSize, 2U * tiling.qkHeadTile);
    EXPECT_EQ(tailAiv0.rowSize + tailAiv1.rowSize, tailTile.rowSize);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, AivSlicesUseDisjointGlobalTokenBases)
{
    ContractTiling tiling;
    auto input = BuildInput(17, 16, 2, 2);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);

    auto tile = MakeTileParamForTest(1, 6, 5, input.totalTokens, tiling);
    auto aiv0 = MakeAivSlice(tile, tiling.qkHeadTile, 0);
    auto aiv1 = MakeAivSlice(tile, tiling.qkHeadTile, 1);

    ASSERT_EQ(tile.tokenOffset, 6U);
    ASSERT_EQ(tile.tokenSize, 5U);
    EXPECT_EQ(CalcCubeHalfTokenSize(tile.tokenSize), 3U);
    EXPECT_EQ(aiv0.tokenOffset, tile.tokenOffset);
    EXPECT_EQ(aiv1.tokenOffset, tile.tokenOffset + CalcCubeHalfTokenSize(tile.tokenSize));
    EXPECT_EQ(aiv0.tokenOffset + aiv0.tokenSize, aiv1.tokenOffset);
    EXPECT_EQ(aiv0.tokenSize + aiv1.tokenSize, tile.tokenSize);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TailAndEmptySlicesKeepFixedCrossCoreProtocol)
{
    ContractTiling tiling;
    auto input = BuildInput(6, 16, 2, 2);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.tokenTile, 6U);

    for (uint64_t tokenSize = 1; tokenSize <= tiling.tokenTile; ++tokenSize) {
        const auto tile = MakeTileParamForTest(tokenSize - 1U, 0U, tokenSize, tokenSize, tiling);
        const auto aiv0 = MakeAivSlice(tile, tiling.qkHeadTile, 0);
        const auto aiv1 = MakeAivSlice(tile, tiling.qkHeadTile, 1);
        const auto counts = CountFixedTileSyncProtocol(tile, tiling.qkHeadTile);

        EXPECT_EQ(tile.tokenSize, tokenSize);
        EXPECT_EQ(aiv0.tokenOffset, 0U);
        EXPECT_EQ(aiv1.tokenOffset, aiv0.tokenSize);
        EXPECT_EQ(aiv0.rowOffset, 0U);
        EXPECT_EQ(aiv1.rowOffset, aiv0.rowSize);
        EXPECT_EQ(aiv0.tokenSize + aiv1.tokenSize, tile.tokenSize);
        EXPECT_EQ(aiv0.rowSize + aiv1.rowSize, tile.rowSize);

        EXPECT_EQ(counts.coveredTokens, tile.tokenSize);
        EXPECT_EQ(counts.coveredRows, tile.rowSize);
        EXPECT_EQ(counts.emptyAivSlices, tokenSize == 1U ? 1U : 0U);
        EXPECT_EQ(counts.aivAReadySets, kAivPerAic);
        EXPECT_EQ(counts.aicAReadyWaits, kAivPerAic);
        EXPECT_EQ(counts.aicCReadySets, kAivPerAic);
        EXPECT_EQ(counts.aivCReadyWaits, kAivPerAic);
        EXPECT_EQ(counts.aivCConsumedSets, kAivPerAic);
        EXPECT_EQ(counts.aicCConsumedWaits, kAivPerAic);
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, CoreTokenRangeUsesHostCeilTokenTile)
{
    EXPECT_EQ(CalcCoreTokenTile(65, 32), 3U);
    EXPECT_EQ(CalcCoreGroupNum(65, 32), 22U);
    EXPECT_EQ(CalcCoreTokenTile(13, 32), 1U);
    EXPECT_EQ(CalcCoreGroupNum(13, 32), 13U);
    EXPECT_EQ(CalcCoreTokenTile(0, 32), 0U);
    EXPECT_EQ(CalcCoreGroupNum(0, 32), 0U);
    EXPECT_EQ(CalcCoreTokenTile(128, 0), 0U);
    EXPECT_EQ(CalcCoreGroupNum(128, 0), 0U);

    ContractTiling tiling;
    auto input = BuildInput(65);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.coreTokenTile, 3U);
    ASSERT_EQ(tiling.coreGroupNum, 22U);

    uint64_t coveredEnd = 0;
    for (uint64_t cubeIndex = 0; cubeIndex < tiling.coreGroupNum; ++cubeIndex) {
        const auto range = MakeCoreTokenRangeForTest(input.totalTokens, cubeIndex, tiling);
        EXPECT_EQ(range.begin, coveredEnd);
        EXPECT_GE(range.end, range.begin);
        EXPECT_LE(range.end - range.begin, tiling.coreTokenTile);
        EXPECT_GT(range.end - range.begin, 0U);
        coveredEnd = range.end;
    }
    EXPECT_EQ(coveredEnd, 65U);

    input = BuildInput(13);
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    const auto first = MakeCoreTokenRangeForTest(input.totalTokens, 0, tiling);
    EXPECT_EQ(first.begin, 0U);
    EXPECT_EQ(first.end, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, CoreLocalTilesFollowContiguousTokenRange)
{
    ContractTiling tiling;
    auto input = BuildInput(40, 16, 2, 2);
    input.aicNum = 3;
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.tokenTile, 6U);
    ASSERT_EQ(tiling.coreTokenTile, 14U);
    ASSERT_EQ(tiling.coreGroupNum, 3U);

    const auto range = MakeCoreTokenRangeForTest(input.totalTokens, 0, tiling);
    ASSERT_EQ(range.begin, 0U);
    ASSERT_EQ(range.end, 14U);

    auto tile0 = MakeTileParamForTest(0, range.begin, tiling.tokenTile, input.totalTokens, tiling);
    auto tile1 = MakeTileParamForTest(
        1, tile0.tokenOffset + tile0.tokenSize, tiling.tokenTile, input.totalTokens, tiling);
    auto tile2 = MakeTileParamForTest(2, tile1.tokenOffset + tile1.tokenSize,
        range.end - tile1.tokenOffset - tile1.tokenSize, input.totalTokens, tiling);
    OutputUseTrackerForTest outputUses;
    const auto tile0Uses = outputUses.NextTile();
    const auto tile1Uses = outputUses.NextTile();
    const auto tile2Uses = outputUses.NextTile();

    EXPECT_EQ(tile0.tokenOffset, 0U);
    EXPECT_EQ(tile0.tokenSize, 6U);
    EXPECT_EQ(tile0.bufferId, 0U);
    EXPECT_EQ(tile0Uses.preprocess, 0U);
    EXPECT_EQ(tile0Uses.qk, 1U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile0Uses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile0Uses.qk), 1U);
    EXPECT_EQ(tile1.tokenOffset, 6U);
    EXPECT_EQ(tile1.tokenSize, 6U);
    EXPECT_EQ(tile1.bufferId, 1U);
    EXPECT_EQ(tile1Uses.preprocess, 2U);
    EXPECT_EQ(tile1Uses.qk, 3U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile1Uses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile1Uses.qk), 1U);
    EXPECT_EQ(tile2.tokenOffset, 12U);
    EXPECT_EQ(tile2.tokenSize, 2U);
    EXPECT_EQ(tile2.bufferId, 0U);
    EXPECT_EQ(tile2Uses.preprocess, 4U);
    EXPECT_EQ(tile2Uses.qk, 5U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile2Uses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile2Uses.qk), 1U);
    EXPECT_EQ(tile0.tokenSize + tile1.tokenSize + tile2.tokenSize, range.end - range.begin);
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, T512Aic40CoreTilesKeepSingleTokenTail)
{
    ContractTiling tiling;
    auto input = BuildInput(512, 16, 2, 2);
    input.aicNum = 40;
    ASSERT_EQ(ValidateAndBuildTiling(input, tiling), ContractStatus::kSuccess);
    ASSERT_EQ(tiling.tokenTile, 6U);
    ASSERT_EQ(tiling.tokenTilePerAiv, 3U);
    ASSERT_EQ(tiling.coreTokenTile, 13U);
    ASSERT_EQ(tiling.coreGroupNum, 40U);

    const auto group0 = MakeCoreTokenRangeForTest(input.totalTokens, 0, tiling);
    const auto group1 = MakeCoreTokenRangeForTest(input.totalTokens, 1, tiling);
    ASSERT_EQ(group0.begin, 0U);
    ASSERT_EQ(group0.end, 13U);
    ASSERT_EQ(group1.begin, 13U);
    ASSERT_EQ(group1.end, 26U);

    auto tile0 = MakeTileParamForTest(0, group0.begin, tiling.tokenTile, input.totalTokens, tiling);
    auto tile1 = MakeTileParamForTest(
        1, tile0.tokenOffset + tile0.tokenSize, tiling.tokenTile, input.totalTokens, tiling);
    auto tile2 = MakeTileParamForTest(2, tile1.tokenOffset + tile1.tokenSize,
        group0.end - tile1.tokenOffset - tile1.tokenSize, input.totalTokens, tiling);
    OutputUseTrackerForTest outputUses;
    const auto tile0Uses = outputUses.NextTile();
    const auto tile1Uses = outputUses.NextTile();
    const auto tile2Uses = outputUses.NextTile();

    EXPECT_EQ(tile0.tokenOffset, 0U);
    EXPECT_EQ(tile0.tokenSize, 6U);
    EXPECT_EQ(tile0.bufferId, 0U);
    EXPECT_EQ(tile0Uses.preprocess, 0U);
    EXPECT_EQ(tile0Uses.qk, 1U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile0Uses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile0Uses.qk), 1U);
    EXPECT_EQ(tile1.tokenOffset, 6U);
    EXPECT_EQ(tile1.tokenSize, 6U);
    EXPECT_EQ(tile1.bufferId, 1U);
    EXPECT_EQ(tile1Uses.preprocess, 2U);
    EXPECT_EQ(tile1Uses.qk, 3U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile1Uses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile1Uses.qk), 1U);
    EXPECT_EQ(tile2.tokenOffset, 12U);
    EXPECT_EQ(tile2.tokenSize, 1U);
    EXPECT_EQ(tile2.bufferId, 0U);
    EXPECT_EQ(tile2Uses.preprocess, 4U);
    EXPECT_EQ(tile2Uses.qk, 5U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile2Uses.preprocess), 0U);
    EXPECT_EQ(MakeOutputBufferIdForTest(tile2Uses.qk), 1U);

    const auto tile0Aiv0 = MakeAivSlice(tile0, tiling.qkHeadTile, 0);
    const auto tile1Aiv0 = MakeAivSlice(tile1, tiling.qkHeadTile, 0);
    const auto tile2Aiv0 = MakeAivSlice(tile2, tiling.qkHeadTile, 0);
    const auto tile0Aiv1 = MakeAivSlice(tile0, tiling.qkHeadTile, 1);
    const auto tile1Aiv1 = MakeAivSlice(tile1, tiling.qkHeadTile, 1);
    const auto tile2Aiv1 = MakeAivSlice(tile2, tiling.qkHeadTile, 1);
    EXPECT_EQ(tile0Aiv0.tokenSize, 3U);
    EXPECT_EQ(tile1Aiv0.tokenSize, 3U);
    EXPECT_EQ(tile2Aiv0.tokenSize, 1U);
    EXPECT_EQ(tile2Aiv0.tokenOffset, 12U);
    EXPECT_EQ(tile0Aiv1.tokenSize, 3U);
    EXPECT_EQ(tile1Aiv1.tokenSize, 3U);
    EXPECT_EQ(tile2Aiv1.tokenSize, 0U);
    EXPECT_TRUE(HasPreprocessWorkForTest(tile0Aiv1));
    EXPECT_TRUE(HasPreprocessWorkForTest(tile1Aiv1));
    EXPECT_FALSE(HasPreprocessWorkForTest(tile2Aiv1));
}

TEST(QkvRmsNormRopeCacheWithKScaleTiling, TilingDataDefaultsAreStable)
{
    optiling::QkvRmsNormRopeCacheWithKScaleTilingData tilingData;
    EXPECT_GT(sizeof(tilingData), 0U);
    EXPECT_EQ(sizeof(tilingData) % sizeof(uint64_t), 0U);
    EXPECT_EQ(tilingData.tokenTile, 0U);
    EXPECT_EQ(tilingData.coreTokenTile, 0U);
    EXPECT_EQ(tilingData.coreGroupNum, 0U);
    EXPECT_EQ(tilingData.kCacheStrideBlock, 0U);
    EXPECT_EQ(tilingData.vCacheStrideBlock, 0U);
    EXPECT_EQ(tilingData.kScaleCacheStrideBlock, 0U);

    tilingData.totalTokens = 128;
    tilingData.numQHeads = 16;
    tilingData.coreTokenTile = 4;
    tilingData.coreGroupNum = 32;
    tilingData.kCacheStrideToken = 128;
    tilingData.vCacheStrideToken = 128;
    tilingData.kScaleCacheStrideToken = 1;
    tilingData.tokenTile = 14;
    EXPECT_EQ(tilingData.totalTokens, 128U);
    EXPECT_EQ(tilingData.numQHeads, 16U);
    EXPECT_EQ(tilingData.coreTokenTile, 4U);
    EXPECT_EQ(tilingData.coreGroupNum, 32U);
    EXPECT_EQ(tilingData.kCacheStrideToken, 128U);
    EXPECT_EQ(tilingData.vCacheStrideToken, 128U);
    EXPECT_EQ(tilingData.kScaleCacheStrideToken, 1U);
    EXPECT_EQ(tilingData.tokenTile, 14U);
}
