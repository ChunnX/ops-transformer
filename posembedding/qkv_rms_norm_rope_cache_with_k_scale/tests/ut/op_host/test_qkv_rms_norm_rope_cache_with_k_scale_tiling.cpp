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
#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include "op_common/op_host/util/math_util.h"
#include "tiling_case_executor.h"

#include "../../../op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h"
#include "../../../op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

namespace {
using QkvTiling = optiling::QkvRmsNormRopeCacheWithKScale::QkvRmsNormRopeCacheWithKScaleBaseTiling;
using ContractInput = optiling::QkvRmsNormRopeCacheWithKScale::ContractInput;
using ContractTiling = optiling::QkvRmsNormRopeCacheWithKScale::ContractTiling;
using TensorContractInfo = optiling::QkvRmsNormRopeCacheWithKScale::TensorContractInfo;
using TileParam = optiling::QkvRmsNormRopeCacheWithKScale::TileParam;

constexpr uint32_t OUTPUT_PREPROCESS_BUFFER_ID = 0U;
constexpr uint32_t OUTPUT_QK_BUFFER_ID = 1U;
constexpr uint64_t TEST_BATCH = 1U;
constexpr uint64_t TEST_BLOCK_NUM = 8U;
constexpr uint64_t TEST_BLOCK_SIZE = 128U;
constexpr uint64_t TEST_MAX_SEQ_LEN = 256U;
constexpr uint64_t TEST_OP_WORKSPACE_SIZE = 4096U;
constexpr uint64_t RESERVED_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

struct AivSliceForTest {
    uint64_t tokenOffset = 0;
    uint64_t tokenSize = 0;
};

ge::graphStatus ValidateContractInput(const ContractInput &input)
{
    return QkvTiling::ValidateContractInput(input);
}

ge::graphStatus BuildContractTiling(const ContractInput &input, ContractTiling &tiling, uint64_t aicNum = 32)
{
    return QkvTiling::BuildContractTiling(input, aicNum, tiling);
}

ge::graphStatus ValidateThenBuildTiling(const ContractInput &input, ContractTiling &tiling, uint64_t aicNum = 32)
{
    const auto status = ValidateContractInput(input);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return BuildContractTiling(input, tiling, aicNum);
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return Ops::Base::CeilAlign(value, align);
}

AivSliceForTest MakeAivSlice(const TileParam &tile, uint64_t aivLocalId)
{
    AivSliceForTest slice;
    const uint64_t cubeHalfTokenSize = Ops::Base::CeilDiv(tile.tokenSize, QkvTiling::AIV_PER_AIC);
    const uint64_t tokenPrefix = aivLocalId == 0 ? 0 : cubeHalfTokenSize;
    const uint64_t tokenSize = aivLocalId == 0 ? cubeHalfTokenSize : tile.tokenSize - cubeHalfTokenSize;
    slice.tokenOffset = tile.tokenOffset + tokenPrefix;
    slice.tokenSize = tokenSize;
    return slice;
}

gert::Shape Shape(std::initializer_list<uint64_t> dims)
{
    gert::Shape shape;
    for (const auto dim : dims) {
        shape.AppendDim(static_cast<int64_t>(dim));
    }
    return shape;
}

void SetShape(TensorContractInfo &tensor, std::initializer_list<uint64_t> dims)
{
    tensor.shape = Shape(dims);
    tensor.shapePresent = true;
}

gert::Stride Stride(std::initializer_list<uint64_t> strides)
{
    gert::Stride stride;
    stride.SetDimNum(strides.size());
    int64_t index = 0;
    for (const auto dimStride : strides) {
        stride.SetStride(index, static_cast<int64_t>(dimStride));
        ++index;
    }
    return stride;
}

void RefreshShapes(ContractInput &input)
{
    const uint64_t qkvN = input.numQHeads + input.numKHeads + input.numVHeads;
    SetShape(input.qkv, {qkvN, input.totalTokens, input.headDim});
    SetShape(input.qGamma, {input.headDim});
    SetShape(input.kGamma, {input.headDim});
    SetShape(input.cosSin, {input.maxSeqLen, input.headDim});
    SetShape(input.slotMapping, {input.totalTokens});
    SetShape(input.kCache, {input.blockNum, input.numKHeads, input.blockSize, input.headDim});
    input.kCache.stride = Stride({input.numKHeads * input.blockSize * input.headDim, input.blockSize * input.headDim,
        input.headDim, 1});
    input.kCache.stridePresent = true;
    SetShape(input.vCache, {input.blockNum, input.numVHeads, input.blockSize, input.headDim});
    input.vCache.stride = Stride({input.numVHeads * input.blockSize * input.headDim, input.blockSize * input.headDim,
        input.headDim, 1});
    input.vCache.stridePresent = true;
    SetShape(input.kScaleCache, {input.blockNum, input.numKHeads, input.blockSize, 1});
    input.kScaleCache.stride = Stride({input.numKHeads * input.blockSize, input.blockSize, 1, 1});
    input.kScaleCache.stridePresent = true;
    SetShape(input.cuSeqLen, {static_cast<uint64_t>(input.batch) + 1});
    SetShape(input.actualSeqLens, {static_cast<uint64_t>(input.batch)});
    SetShape(input.qRotation, {input.headDim, input.headDim});
    SetShape(input.kRotation, {input.headDim, input.headDim});
    SetShape(input.vScale, {input.numVHeads});
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

    input.qkv.dtype = ge::DT_BF16;
    input.qGamma.dtype = ge::DT_FLOAT;
    input.kGamma.dtype = ge::DT_FLOAT;
    input.cosSin.dtype = ge::DT_FLOAT;
    input.slotMapping.dtype = ge::DT_INT32;
    input.kCache.dtype = ge::DT_FLOAT8_E4M3FN;
    input.vCache.dtype = ge::DT_FLOAT8_E4M3FN;
    input.kScaleCache.dtype = ge::DT_FLOAT;
    input.cuSeqLen.dtype = ge::DT_INT32;
    input.actualSeqLens.dtype = ge::DT_INT32;
    input.qRotation.dtype = ge::DT_BF16;
    input.kRotation.dtype = ge::DT_BF16;
    input.vScale.dtype = ge::DT_FLOAT;
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
    range.end = std::min(totalTokens, range.begin + tiling.coreTokenTile);
    return range;
}

bool HasPreprocessWorkForTest(const AivSliceForTest &slice)
{
    return slice.tokenSize != 0U;
}

uint64_t QkHeads(const ContractTiling &tiling)
{
    return tiling.numQHeads + tiling.numKHeads;
}

uint64_t QkvHeads(const ContractTiling &tiling)
{
    return QkHeads(tiling) + tiling.numVHeads;
}

TileParam MakeTileParamForTest(uint64_t tileId, uint64_t tokenOffset, uint64_t tokenSize, uint64_t totalTokens,
    const ContractTiling &tiling)
{
    TileParam tile;
    tile.tileId = tileId;
    tile.tokenOffset = tokenOffset;
    if (tile.tokenOffset >= totalTokens || tokenSize == 0) {
        return tile;
    }
    tile.tokenSize = std::min({tokenSize, tiling.tokenTile, totalTokens - tile.tokenOffset});
    tile.qkHeadOffset = 0;
    tile.qkHeadSize = QkHeads(tiling);
    tile.vHeadOffset = 0;
    tile.vHeadSize = tiling.numVHeads;
    tile.rowOffset = 0;
    tile.rowSize = tile.tokenSize * QkHeads(tiling);
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

    counts.aicAReadyWaits = QkvTiling::AIV_PER_AIC;
    counts.aicCReadySets = QkvTiling::AIV_PER_AIC;
    counts.aicCConsumedWaits = QkvTiling::AIV_PER_AIC;
    for (uint64_t aivLocalId = 0; aivLocalId < QkvTiling::AIV_PER_AIC; ++aivLocalId) {
        const auto slice = MakeAivSlice(tile, aivLocalId);
        ++counts.aivAReadySets;
        ++counts.aivCReadyWaits;
        ++counts.aivCConsumedSets;
        counts.coveredTokens += slice.tokenSize;
        counts.coveredRows += slice.tokenSize * qkHeadTile;
        if (slice.tokenSize == 0) {
            ++counts.emptyAivSlices;
        }
    }
    return counts;
}

struct ConcurrentTilingCase {
    uint64_t totalTokens;
    uint64_t numQHeads;
    uint64_t numKHeads;
    uint64_t numVHeads;
    uint64_t headDim;
    uint64_t aicNum;
    ge::graphStatus status;
    uint64_t tokenTile;
    uint64_t coreTokenTile;
    uint64_t coreGroupNum;
};

struct ConcurrentTilingResult {
    bool ok = true;
    uint64_t failedCase = 0;
    uint32_t failedIteration = 0;
    bool success = false;
    uint64_t tokenTile = 0;
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t blockNum = 0;
};

int64_t ToI64(uint64_t value)
{
    return static_cast<int64_t>(value);
}

gert::StorageShape MakeStorageShape(std::initializer_list<int64_t> dims)
{
    return gert::StorageShape(dims, dims);
}

gert::TilingContextPara::TensorDescription MakeTensorDesc(
    std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    return {MakeStorageShape(dims), dtype, ge::FORMAT_ND};
}

std::vector<gert::TilingContextPara::TensorDescription> BuildTilingInputs(const ConcurrentTilingCase &item)
{
    const uint64_t qkvN = item.numQHeads + item.numKHeads + item.numVHeads;
    return {
        MakeTensorDesc({ToI64(qkvN), ToI64(item.totalTokens), ToI64(item.headDim)}, ge::DT_BF16),
        MakeTensorDesc({ToI64(item.headDim)}, ge::DT_FLOAT),
        MakeTensorDesc({ToI64(item.headDim)}, ge::DT_FLOAT),
        MakeTensorDesc({ToI64(TEST_MAX_SEQ_LEN), ToI64(item.headDim)}, ge::DT_FLOAT),
        MakeTensorDesc({ToI64(item.totalTokens)}, ge::DT_INT32),
        MakeTensorDesc({ToI64(TEST_BLOCK_NUM), ToI64(item.numKHeads), ToI64(TEST_BLOCK_SIZE), ToI64(item.headDim)},
                       ge::DT_FLOAT8_E4M3FN),
        MakeTensorDesc({ToI64(TEST_BLOCK_NUM), ToI64(item.numVHeads), ToI64(TEST_BLOCK_SIZE), ToI64(item.headDim)},
                       ge::DT_FLOAT8_E4M3FN),
        MakeTensorDesc({ToI64(TEST_BLOCK_NUM), ToI64(item.numKHeads), ToI64(TEST_BLOCK_SIZE), 1}, ge::DT_FLOAT),
        MakeTensorDesc({ToI64(TEST_BATCH + 1)}, ge::DT_INT32),
        MakeTensorDesc({ToI64(TEST_BATCH)}, ge::DT_INT32),
        MakeTensorDesc({ToI64(item.headDim), ToI64(item.headDim)}, ge::DT_BF16),
        MakeTensorDesc({ToI64(item.headDim), ToI64(item.headDim)}, ge::DT_BF16),
        MakeTensorDesc({ToI64(item.numVHeads)}, ge::DT_FLOAT),
    };
}

std::vector<gert::TilingContextPara::TensorDescription> BuildTilingOutputs(const ConcurrentTilingCase &item)
{
    return {
        MakeTensorDesc({ToI64(item.numQHeads), ToI64(item.totalTokens), ToI64(item.headDim)}, ge::DT_FLOAT8_E4M3FN),
        MakeTensorDesc({ToI64(item.numQHeads), ToI64(item.totalTokens)}, ge::DT_FLOAT),
        MakeTensorDesc({ToI64(TEST_BLOCK_NUM), ToI64(item.numKHeads), ToI64(TEST_BLOCK_SIZE), ToI64(item.headDim)},
                       ge::DT_FLOAT8_E4M3FN),
        MakeTensorDesc({ToI64(TEST_BLOCK_NUM), ToI64(item.numVHeads), ToI64(TEST_BLOCK_SIZE), ToI64(item.headDim)},
                       ge::DT_FLOAT8_E4M3FN),
        MakeTensorDesc({ToI64(TEST_BLOCK_NUM), ToI64(item.numKHeads), ToI64(TEST_BLOCK_SIZE), 1}, ge::DT_FLOAT),
    };
}

std::vector<gert::TilingContextPara::OpAttr> BuildTilingAttrs(const ConcurrentTilingCase &item)
{
    const std::vector<int64_t> headNums = {
        ToI64(item.numQHeads),
        ToI64(item.numKHeads),
        ToI64(item.numVHeads),
    };
    return {
        {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
    };
}

std::vector<gert::TilingContextPara::OpAttr> BuildTilingAttrs(std::initializer_list<int64_t> headNums)
{
    return {
        {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>(headNums))},
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
    };
}

optiling::QkvRmsNormRopeCacheWithKScaleCompileInfo BuildCompileInfo(const ConcurrentTilingCase &item)
{
    optiling::QkvRmsNormRopeCacheWithKScaleCompileInfo compileInfo;
    compileInfo.aicNum = static_cast<uint32_t>(item.aicNum);
    compileInfo.aivNum = static_cast<uint32_t>(item.aicNum * QkvTiling::AIV_PER_AIC);
    compileInfo.ubSize = 262144U;
    compileInfo.l1Size = 524288U;
    compileInfo.l0cSize = 131072U;
    compileInfo.opWorkspaceSize = TEST_OP_WORKSPACE_SIZE;
    return compileInfo;
}

bool ExecuteConcurrentTilingCase(const ConcurrentTilingCase &item, TilingInfo &tilingInfo)
{
    auto compileInfo = BuildCompileInfo(item);
    const gert::TilingContextPara para(
        "QkvRmsNormRopeCacheWithKScale", BuildTilingInputs(item), BuildTilingOutputs(item), BuildTilingAttrs(item),
        &compileInfo, "Ascend950", 64, 262144, 4096);
    return ExecuteTiling(para, tilingInfo);
}

bool ExecuteTilingWithInputs(const ConcurrentTilingCase &item,
                             const std::vector<gert::TilingContextPara::TensorDescription> &inputs)
{
    auto compileInfo = BuildCompileInfo(item);
    TilingInfo tilingInfo;
    const gert::TilingContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildTilingOutputs(item),
        BuildTilingAttrs(item), &compileInfo, "Ascend950", 64, 262144, 4096);
    return ExecuteTiling(para, tilingInfo);
}

bool MatchesConcurrentTilingCase(const ConcurrentTilingCase &item, bool success, const TilingInfo &tilingInfo)
{
    if (success != (item.status == ge::GRAPH_SUCCESS)) {
        return false;
    }
    if (!success) {
        return true;
    }
    if (tilingInfo.tilingDataSize < sizeof(optiling::QkvRmsNormRopeCacheWithKScaleTilingData) ||
        tilingInfo.tilingData == nullptr || tilingInfo.workspaceSizes.empty()) {
        return false;
    }

    const auto *tilingData =
        reinterpret_cast<const optiling::QkvRmsNormRopeCacheWithKScaleTilingData *>(tilingInfo.tilingData.get());
    const int64_t expectedWorkspace = static_cast<int64_t>(RESERVED_WORKSPACE_SIZE + TEST_OP_WORKSPACE_SIZE);
    return tilingInfo.blockNum == item.coreGroupNum && tilingInfo.workspaceSizes[0] == expectedWorkspace &&
           tilingData->totalTokens == item.totalTokens && tilingData->qHeadNum == item.numQHeads &&
           tilingData->kvHeadNum == item.numKHeads &&
           tilingData->headDim == item.headDim && tilingData->blockSize == TEST_BLOCK_SIZE &&
           tilingData->tokenTile == item.tokenTile && tilingData->coreTokenTile == item.coreTokenTile &&
           tilingData->coreGroupNum == item.coreGroupNum;
}

} // namespace

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, BuildInputUsesNtdQkvShape)
{
    auto input = BuildInput(512, 64, 8, 8);
    EXPECT_EQ(input.qkv.shape.GetDimNum(), 3);
    EXPECT_EQ(static_cast<uint64_t>(input.qkv.shape.GetDim(0)), input.numQHeads + input.numKHeads + input.numVHeads);
    EXPECT_EQ(static_cast<uint64_t>(input.qkv.shape.GetDim(1)), input.totalTokens);
    EXPECT_EQ(static_cast<uint64_t>(input.qkv.shape.GetDim(2)), input.headDim);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, PreservesQ8K1V1HeadFieldsInRawTilingData)
{
    const ConcurrentTilingCase item = {128, 8, 1, 1, 128, 32, ge::GRAPH_SUCCESS, 16, 4, 32};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteConcurrentTilingCase(item, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::QkvRmsNormRopeCacheWithKScaleTilingData));
    ASSERT_NE(tilingInfo.tilingData, nullptr);

    const auto *tilingData =
        reinterpret_cast<const optiling::QkvRmsNormRopeCacheWithKScaleTilingData *>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->qHeadNum, 8U);
    EXPECT_EQ(tilingData->kvHeadNum, 1U);
    EXPECT_EQ(tilingData->headDim, 128U);
    EXPECT_EQ(tilingData->tokenTile, 16U);
    EXPECT_EQ(tilingData->coreTokenTile, 4U);
    EXPECT_EQ(tilingData->coreGroupNum, 32U);
    EXPECT_EQ(tilingInfo.blockNum, 32U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, TypicalHeadTableMatchesDesign)
{
    struct Case {
        uint32_t numQHeads;
        uint32_t numKHeads;
        uint32_t tokenTile;
        uint32_t rowTile;
        uint32_t rowTileAligned;
        uint32_t qkAlignedRows;
    };
    const Case cases[] = {
        {16, 2, 14, 252, 256, 256},
        {16, 4, 12, 240, 240, 240},
        {16, 8, 10, 240, 240, 240},
        {32, 2, 6, 204, 208, 208},
        {32, 4, 6, 216, 224, 224},
        {32, 8, 6, 240, 240, 240},
        {64, 2, 2, 132, 144, 144},
        {64, 4, 2, 136, 144, 144},
        {64, 8, 2, 144, 144, 144},
    };

    for (const auto &item : cases) {
        ContractTiling tiling;
        auto input = BuildInput(128, item.numQHeads, item.numKHeads, item.numKHeads);
        EXPECT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
        EXPECT_EQ(tiling.coreTokenTile, 4U);
        EXPECT_EQ(tiling.coreGroupNum, 32U);
        EXPECT_EQ(tiling.tokenTile, item.tokenTile);
        EXPECT_EQ(tiling.tokenTilePerAiv, item.tokenTile / 2);
        EXPECT_EQ(tiling.rowTile, item.rowTile);
        EXPECT_EQ(tiling.rowTileAligned, item.rowTileAligned);
        EXPECT_EQ(AlignUp(tiling.tokenTile * input.numQHeads, 16) +
                AlignUp(tiling.tokenTile * input.numKHeads, 16),
            item.qkAlignedRows);
        EXPECT_EQ(QkHeads(tiling), item.numQHeads + item.numKHeads);
        EXPECT_EQ(tiling.numVHeads, item.numKHeads);
        EXPECT_EQ(tiling.dimTile, 128U);
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, QkAlignedRowsOnlyLimitSideBySideL0c)
{
    ContractTiling tiling;
    auto input = BuildInput(128, 16, 2, 2);
    EXPECT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 14U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 7U);
    EXPECT_EQ(tiling.rowTile, 252U);
    EXPECT_EQ(tiling.rowTileAligned, 256U);
    const uint64_t qkAlignedRows = AlignUp(tiling.tokenTile * input.numQHeads, 16) +
        AlignUp(tiling.tokenTile * input.numKHeads, 16);
    EXPECT_GT(qkAlignedRows, QkvTiling::L0A_FULL_DIM_SEGMENT_ROWS);
    EXPECT_LE(qkAlignedRows, QkvTiling::L0C_MAX_ROWS);
    EXPECT_GT(AlignUp(tiling.tokenTile * input.numQHeads, 16), QkvTiling::L0A_FULL_DIM_SEGMENT_ROWS);
    EXPECT_LE(AlignUp(tiling.tokenTile * input.numKHeads, 16), QkvTiling::L0A_FULL_DIM_SEGMENT_ROWS);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, SelectsSingleTokenTileWhenOnlyMinimumFitsL0c)
{
    ContractTiling tiling;
    auto input = BuildInput(128, 121, 7, 7);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 1U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_EQ(tiling.rowTile, 128U);
    EXPECT_EQ(tiling.rowTileAligned, 128U);
    EXPECT_LE(AlignUp(input.numQHeads, 16) + AlignUp(input.numKHeads, 16), QkvTiling::L0C_MAX_ROWS);
    EXPECT_GT(AlignUp(2U * input.numQHeads, 16) + AlignUp(2U * input.numKHeads, 16),
        QkvTiling::L0C_MAX_ROWS);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsUnsupportedFastPath)
{
    ContractTiling tiling;
    auto input = BuildInput(128, 16, 2, 4);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput(128, 16, 2, 2, 64);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput(128, 0, 2, 2);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput(128, 128, 8, 8);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput(128, 64, 2, 2);
    EXPECT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 2U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_EQ(tiling.rowTile, 132U);
    EXPECT_EQ(tiling.rowTileAligned, 144U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsMinimumTokenTileResourceBoundary)
{
    auto input = BuildInput(128, 121, 8, 8);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput(128, 80, 41, 41);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput(128, 1, 81, 81);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsNonPositiveHeadNumsAttr)
{
    const ConcurrentTilingCase item = {128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32};
    auto compileInfo = BuildCompileInfo(item);
    TilingInfo tilingInfo;
    const gert::TilingContextPara zeroHeadPara(
        "QkvRmsNormRopeCacheWithKScale", BuildTilingInputs(item), BuildTilingOutputs(item),
        BuildTilingAttrs({0, 2, 2}), &compileInfo, "Ascend950", 64, 262144, 4096);
    EXPECT_FALSE(ExecuteTiling(zeroHeadPara, tilingInfo));

    const gert::TilingContextPara negativeHeadPara(
        "QkvRmsNormRopeCacheWithKScale", BuildTilingInputs(item), BuildTilingOutputs(item),
        BuildTilingAttrs({16, -1, 2}), &compileInfo, "Ascend950", 64, 262144, 4096);
    EXPECT_FALSE(ExecuteTiling(negativeHeadPara, tilingInfo));
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsNonPositiveDimsBeforeShapeDerivedFields)
{
    const ConcurrentTilingCase item = {128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32};
    const uint64_t qkvN = item.numQHeads + item.numKHeads + item.numVHeads;

    auto inputs = BuildTilingInputs(item);
    inputs[0].shape_ = MakeStorageShape({ToI64(qkvN), -1, ToI64(item.headDim)});
    EXPECT_FALSE(ExecuteTilingWithInputs(item, inputs));

    inputs = BuildTilingInputs(item);
    inputs[8].shape_ = MakeStorageShape({0});
    EXPECT_FALSE(ExecuteTilingWithInputs(item, inputs));

    inputs = BuildTilingInputs(item);
    inputs[8].shape_ = MakeStorageShape({1});
    EXPECT_FALSE(ExecuteTilingWithInputs(item, inputs));

    inputs = BuildTilingInputs(item);
    inputs[5].shape_ = MakeStorageShape({ToI64(TEST_BLOCK_NUM), ToI64(item.numKHeads), -1, ToI64(item.headDim)});
    EXPECT_FALSE(ExecuteTilingWithInputs(item, inputs));
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsViewCacheWithoutStride)
{
    const ConcurrentTilingCase item = {128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32};
    auto inputs = BuildTilingInputs(item);
    inputs[5].isView_ = true;
    EXPECT_FALSE(ExecuteTilingWithInputs(item, inputs));
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsNonTwoToOneAivAicRatioInPlatformInfo)
{
    const ConcurrentTilingCase item = {128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32};
    auto compileInfo = BuildCompileInfo(item);
    compileInfo.aivNum = compileInfo.aicNum * QkvTiling::AIV_PER_AIC - 1;
    TilingInfo tilingInfo;
    const gert::TilingContextPara fewerAivPara(
        "QkvRmsNormRopeCacheWithKScale", BuildTilingInputs(item), BuildTilingOutputs(item), BuildTilingAttrs(item),
        &compileInfo, "Ascend950", 64, 262144, 4096);
    EXPECT_FALSE(ExecuteTiling(fewerAivPara, tilingInfo));

    compileInfo = BuildCompileInfo(item);
    compileInfo.aivNum = compileInfo.aicNum * QkvTiling::AIV_PER_AIC + 1;
    const gert::TilingContextPara moreAivPara(
        "QkvRmsNormRopeCacheWithKScale", BuildTilingInputs(item), BuildTilingOutputs(item), BuildTilingAttrs(item),
        &compileInfo, "Ascend950", 64, 262144, 4096);
    EXPECT_FALSE(ExecuteTiling(moreAivPara, tilingInfo));
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsNonAscend950TilingDispatch)
{
    const ConcurrentTilingCase item = {128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32};
    auto compileInfo = BuildCompileInfo(item);
    TilingInfo tilingInfo;
    const gert::TilingContextPara para(
        "QkvRmsNormRopeCacheWithKScale", BuildTilingInputs(item), BuildTilingOutputs(item), BuildTilingAttrs(item),
        &compileInfo, "Ascend910B", 64, 262144, 4096);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, AcceptsT512Q64K8V8Aic32)
{
    ContractTiling tiling;
    auto input = BuildInput(512, 64, 8, 8);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 2U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_EQ(tiling.rowTile, 144U);
    EXPECT_EQ(tiling.rowTileAligned, 144U);
    EXPECT_EQ(QkHeads(tiling), 72U);
    EXPECT_EQ(tiling.numVHeads, 8U);
    EXPECT_EQ(tiling.coreTokenTile, 16U);
    EXPECT_EQ(tiling.coreGroupNum, 32U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, SeparateResourceCapacitiesAreEnforced)
{
    ContractTiling tiling;
    auto input = BuildInput(128, 120, 8, 8);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 2U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_EQ(tiling.tokenTilePerAiv * QkHeads(tiling), QkvTiling::QK_OUTPUT_ROWS_PER_AIV);

    input = BuildInput(128, 1, 79, 79);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 2U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 1U);
    EXPECT_LE(tiling.tokenTilePerAiv * tiling.numVHeads, QkvTiling::V_OUTPUT_ROWS_PER_AIV);
    EXPECT_LE(tiling.tokenTilePerAiv * QkvHeads(tiling), QkvTiling::QKV_INPUT_ROWS_PER_AIV);

    input = BuildInput(128, 121, 8, 8);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, T1024Q16K2V2Aic32FitsSeparateResourceCapacities)
{
    ContractTiling tiling;
    auto input = BuildInput(1024, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.tokenTile, 14U);
    EXPECT_EQ(tiling.tokenTilePerAiv, 7U);
    EXPECT_EQ(tiling.rowTile, 252U);
    EXPECT_EQ(tiling.rowTileAligned, 256U);
    EXPECT_EQ(tiling.coreTokenTile, 32U);
    EXPECT_EQ(tiling.coreGroupNum, 32U);

    const uint64_t qRowsPerAiv = tiling.tokenTilePerAiv * tiling.numQHeads;
    EXPECT_EQ(qRowsPerAiv, 112U);
    EXPECT_LE(tiling.tokenTilePerAiv * QkHeads(tiling), QkvTiling::QK_OUTPUT_ROWS_PER_AIV);
    EXPECT_LE(tiling.tokenTilePerAiv * QkvHeads(tiling), QkvTiling::QKV_INPUT_ROWS_PER_AIV);
    EXPECT_LE(tiling.tokenTilePerAiv * tiling.numVHeads, QkvTiling::V_OUTPUT_ROWS_PER_AIV);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RealTilingIsStableAcrossThreads)
{
    constexpr uint32_t THREAD_COUNT = 8U;
    constexpr uint32_t ITERATIONS = 64U;
    const std::array<ConcurrentTilingCase, 5U> cases = {{
        {128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32},
        {512, 64, 8, 8, 128, 32, ge::GRAPH_SUCCESS, 2, 16, 32},
        {1024, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 32, 32},
        {128, 128, 8, 8, 128, 32, ge::GRAPH_FAILED, 0, 0, 0},
        {128, 16, 2, 2, 64, 32, ge::GRAPH_FAILED, 0, 0, 0},
    }};
    std::atomic<bool> start{false};
    std::array<ConcurrentTilingResult, THREAD_COUNT> results;
    std::array<std::thread, THREAD_COUNT> threads;

    for (uint32_t threadIdx = 0U; threadIdx < THREAD_COUNT; ++threadIdx) {
        threads[threadIdx] = std::thread([threadIdx, &cases, &results, &start]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t iter = 0U; iter < ITERATIONS; ++iter) {
                const uint64_t caseIdx = (static_cast<uint64_t>(threadIdx) + iter) % cases.size();
                const auto &item = cases[caseIdx];
                TilingInfo tilingInfo;
                const bool success = ExecuteConcurrentTilingCase(item, tilingInfo);
                if (!MatchesConcurrentTilingCase(item, success, tilingInfo)) {
                    results[threadIdx].ok = false;
                    results[threadIdx].failedCase = caseIdx;
                    results[threadIdx].failedIteration = iter;
                    results[threadIdx].success = success;
                    results[threadIdx].blockNum = tilingInfo.blockNum;
                    if (success && tilingInfo.tilingDataSize >= sizeof(optiling::QkvRmsNormRopeCacheWithKScaleTilingData) &&
                        tilingInfo.tilingData != nullptr) {
                        const auto *tilingData = reinterpret_cast<const optiling::QkvRmsNormRopeCacheWithKScaleTilingData *>(
                            tilingInfo.tilingData.get());
                        results[threadIdx].tokenTile = tilingData->tokenTile;
                        results[threadIdx].coreTokenTile = tilingData->coreTokenTile;
                        results[threadIdx].coreGroupNum = tilingData->coreGroupNum;
                    }
                    return;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &thread : threads) {
        thread.join();
    }

    for (uint32_t threadIdx = 0U; threadIdx < THREAD_COUNT; ++threadIdx) {
        EXPECT_TRUE(results[threadIdx].ok)
            << "threadIdx=" << threadIdx << " failedCase=" << results[threadIdx].failedCase
            << " failedIteration=" << results[threadIdx].failedIteration
            << " success=" << results[threadIdx].success
            << " blockNum=" << results[threadIdx].blockNum
            << " tokenTile=" << results[threadIdx].tokenTile
            << " coreTokenTile=" << results[threadIdx].coreTokenTile
            << " coreGroupNum=" << results[threadIdx].coreGroupNum;
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsMissingOptionalButRequiredInputs)
{
    auto input = BuildInput();
    input.qRotation.shapePresent = false;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kRotation.shapePresent = false;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.vScale.shapePresent = false;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsInvalidShapeAndDtype)
{
    auto input = BuildInput();
    const uint64_t invalidQkvN = input.numQHeads + input.numKHeads + input.numVHeads + 1;
    SetShape(input.qkv, {invalidQkvN, input.totalTokens, input.headDim});
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kCache.shape[1] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.vCache.shape[1] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kCache.shape[2] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.vCache.shape[0] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kScaleCache.shape[2] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kScaleCache.shape[3] = 2;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.qkv.shape[1] = -1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.cuSeqLen.shape[0] = 0;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.cuSeqLen.shape[0] = 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.actualSeqLens.shape[0] = 0;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.actualSeqLens.shape[0] = input.batch + 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kCache.shape[2] = -1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kScaleCache.shape[0] = -1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.qRotation.shape[1] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kRotation.shape[0] += 1;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.qkv.dtype = ge::DT_FLOAT;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.qRotation.dtype = ge::DT_FLOAT;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kRotation.dtype = ge::DT_FLOAT;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.actualSeqLens.dtype = ge::DT_FLOAT;
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, ContiguousCacheStridesAreReported)
{
    ContractTiling tiling;
    auto input = BuildInput();
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.kvCacheStrideBlock, input.numKHeads * input.blockSize * input.headDim);
    EXPECT_EQ(tiling.kvCacheStrideHead, input.blockSize * input.headDim);
    EXPECT_EQ(tiling.kvCacheStrideToken, input.headDim);
    EXPECT_EQ(tiling.kScaleCacheStrideBlock, input.numKHeads * input.blockSize);
    EXPECT_EQ(tiling.kScaleCacheStrideHead, input.blockSize);
    EXPECT_EQ(tiling.kScaleCacheStrideToken, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, AcceptsPaddedCacheStrides)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.kCache.stride = Stride({45000, 21000, 160, 1});
    input.vCache.stride = Stride({45000, 21000, 160, 1});
    input.kScaleCache.stride = Stride({1024, 256, 2, 1});
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.kvCacheStrideBlock, 45000U);
    EXPECT_EQ(tiling.kvCacheStrideHead, 21000U);
    EXPECT_EQ(tiling.kvCacheStrideToken, 160U);
    EXPECT_EQ(tiling.kScaleCacheStrideBlock, 1024U);
    EXPECT_EQ(tiling.kScaleCacheStrideHead, 256U);
    EXPECT_EQ(tiling.kScaleCacheStrideToken, 2U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, AcceptsMatchedCompactViewStrides)
{
    ContractTiling tiling;
    auto input = BuildInput();
    input.kCache.stride = Stride({1, 1, 1, 1});
    input.vCache.stride = Stride({1, 1, 1, 1});
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.kvCacheStrideBlock, 1U);
    EXPECT_EQ(tiling.kvCacheStrideHead, 1U);
    EXPECT_EQ(tiling.kvCacheStrideToken, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsMismatchedKvCacheStrides)
{
    auto input = BuildInput();
    input.kCache.stride = Stride({45000, 21000, 160, 1});
    input.vCache.stride = Stride({45000, 21000, 192, 1});
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kCache.stride = Stride({45000, 21000, 160, 1});
    input.vCache.stride = Stride({52000, 21000, 160, 1});
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RejectsInvalidCacheStrides)
{
    auto input = BuildInput();
    input.kCache.stride.SetStride(3, 2);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.vCache.stride.SetStride(2, 0);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kCache.stride.SetDimNum(3);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kScaleCache.stride.SetStride(3, 2);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);

    input = BuildInput();
    input.kScaleCache.stride.SetStride(1, 0);
    EXPECT_EQ(ValidateContractInput(input), ge::GRAPH_FAILED);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, TailTileAndAivSlicesStayContiguous)
{
    ContractTiling tiling;
    auto input = BuildInput(21, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling.tokenTile, 14U);
    ASSERT_EQ(Ops::Base::CeilDiv(input.totalTokens, tiling.tokenTile), 2U);

    auto firstTile = MakeTileParamForTest(0, 0, tiling.tokenTile, input.totalTokens, tiling);
    EXPECT_EQ(firstTile.tokenOffset, 0U);
    EXPECT_EQ(firstTile.tokenSize, 14U);
    EXPECT_EQ(firstTile.rowSize, tiling.rowTile);

    auto tailTile = MakeTileParamForTest(0, 20, 1, input.totalTokens, tiling);
    EXPECT_EQ(tailTile.tokenOffset, 20U);
    EXPECT_EQ(tailTile.tokenSize, 1U);
    EXPECT_EQ(tailTile.rowSize, QkHeads(tiling));

    auto aiv0 = MakeAivSlice(tailTile, 0);
    auto aiv1 = MakeAivSlice(tailTile, 1);
    EXPECT_EQ(aiv0.tokenOffset, 20U);
    EXPECT_EQ(aiv0.tokenSize, 1U);
    EXPECT_EQ(aiv1.tokenOffset, 21U);
    EXPECT_EQ(aiv1.tokenSize, 0U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, AivSlicesUseActualTailTokens)
{
    ContractTiling tiling;
    auto input = BuildInput(19, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling.tokenTile, 14U);

    auto fullTile = MakeTileParamForTest(0, 0, tiling.tokenTile, input.totalTokens, tiling);
    auto fullAiv0 = MakeAivSlice(fullTile, 0);
    auto fullAiv1 = MakeAivSlice(fullTile, 1);
    EXPECT_EQ(OUTPUT_PREPROCESS_BUFFER_ID, 0U);
    EXPECT_EQ(OUTPUT_QK_BUFFER_ID, 1U);
    EXPECT_EQ(fullAiv0.tokenOffset, 0U);
    EXPECT_EQ(fullAiv0.tokenSize, tiling.tokenTilePerAiv);
    EXPECT_EQ(fullAiv1.tokenOffset, tiling.tokenTilePerAiv);
    EXPECT_EQ(fullAiv1.tokenSize, tiling.tokenTilePerAiv);
    EXPECT_EQ((fullAiv0.tokenSize + fullAiv1.tokenSize) * QkHeads(tiling), fullTile.rowSize);

    auto tailTile = MakeTileParamForTest(1, 14, 5, input.totalTokens, tiling);
    auto tailAiv0 = MakeAivSlice(tailTile, 0);
    auto tailAiv1 = MakeAivSlice(tailTile, 1);
    ASSERT_EQ(tailTile.tokenOffset, 14U);
    ASSERT_EQ(tailTile.tokenSize, 5U);
    EXPECT_EQ(OUTPUT_PREPROCESS_BUFFER_ID, 0U);
    EXPECT_EQ(OUTPUT_QK_BUFFER_ID, 1U);
    EXPECT_EQ(tailTile.rowSize, 5U * QkHeads(tiling));
    EXPECT_EQ(tailAiv0.tokenOffset, 14U);
    EXPECT_EQ(tailAiv0.tokenSize, 3U);
    EXPECT_EQ(tailAiv1.tokenOffset, tailTile.tokenOffset + tailAiv0.tokenSize);
    EXPECT_EQ(tailAiv1.tokenSize, 2U);
    EXPECT_EQ((tailAiv0.tokenSize + tailAiv1.tokenSize) * QkHeads(tiling), tailTile.rowSize);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, AivSlicesUseDisjointGlobalTokenBases)
{
    ContractTiling tiling;
    auto input = BuildInput(29, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);

    auto tile = MakeTileParamForTest(1, 14, 9, input.totalTokens, tiling);
    auto aiv0 = MakeAivSlice(tile, 0);
    auto aiv1 = MakeAivSlice(tile, 1);

    ASSERT_EQ(tile.tokenOffset, 14U);
    ASSERT_EQ(tile.tokenSize, 9U);
    EXPECT_EQ(aiv0.tokenSize, 5U);
    EXPECT_EQ(aiv0.tokenOffset, tile.tokenOffset);
    EXPECT_EQ(aiv1.tokenOffset, tile.tokenOffset + aiv0.tokenSize);
    EXPECT_EQ(aiv0.tokenOffset + aiv0.tokenSize, aiv1.tokenOffset);
    EXPECT_EQ(aiv0.tokenSize + aiv1.tokenSize, tile.tokenSize);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, TailAndEmptySlicesKeepFixedCrossCoreProtocol)
{
    ContractTiling tiling;
    auto input = BuildInput(10, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling.tokenTile, 14U);

    for (uint64_t tokenSize = 1; tokenSize <= tiling.tokenTile; ++tokenSize) {
        const auto tile = MakeTileParamForTest(tokenSize - 1U, 0U, tokenSize, tokenSize, tiling);
        const auto aiv0 = MakeAivSlice(tile, 0);
        const auto aiv1 = MakeAivSlice(tile, 1);
        const auto counts = CountFixedTileSyncProtocol(tile, QkHeads(tiling));

        EXPECT_EQ(tile.tokenSize, tokenSize);
        EXPECT_EQ(aiv0.tokenOffset, 0U);
        EXPECT_EQ(aiv1.tokenOffset, aiv0.tokenSize);
        EXPECT_EQ(aiv0.tokenSize + aiv1.tokenSize, tile.tokenSize);
        EXPECT_EQ((aiv0.tokenSize + aiv1.tokenSize) * QkHeads(tiling), tile.rowSize);

        EXPECT_EQ(counts.coveredTokens, tile.tokenSize);
        EXPECT_EQ(counts.coveredRows, tile.rowSize);
        EXPECT_EQ(counts.emptyAivSlices, tokenSize == 1U ? 1U : 0U);
        EXPECT_EQ(counts.aivAReadySets, QkvTiling::AIV_PER_AIC);
        EXPECT_EQ(counts.aicAReadyWaits, QkvTiling::AIV_PER_AIC);
        EXPECT_EQ(counts.aicCReadySets, QkvTiling::AIV_PER_AIC);
        EXPECT_EQ(counts.aivCReadyWaits, QkvTiling::AIV_PER_AIC);
        EXPECT_EQ(counts.aivCConsumedSets, QkvTiling::AIV_PER_AIC);
        EXPECT_EQ(counts.aicCConsumedWaits, QkvTiling::AIV_PER_AIC);
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, CoreTokenRangeUsesHostCeilTokenTile)
{
    ContractTiling tiling;
    auto input = BuildInput(65);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
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
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling), ge::GRAPH_SUCCESS);
    const auto first = MakeCoreTokenRangeForTest(input.totalTokens, 0, tiling);
    EXPECT_EQ(first.begin, 0U);
    EXPECT_EQ(first.end, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, CoreLocalTilesFollowContiguousTokenRange)
{
    ContractTiling tiling;
    auto input = BuildInput(68, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling, 3), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling.tokenTile, 14U);
    ASSERT_EQ(tiling.coreTokenTile, 23U);
    ASSERT_EQ(tiling.coreGroupNum, 3U);

    const auto range = MakeCoreTokenRangeForTest(input.totalTokens, 0, tiling);
    ASSERT_EQ(range.begin, 0U);
    ASSERT_EQ(range.end, 23U);

    auto tile0 = MakeTileParamForTest(0, range.begin, tiling.tokenTile, input.totalTokens, tiling);
    auto tile1 = MakeTileParamForTest(
        1, tile0.tokenOffset + tile0.tokenSize, range.end - tile0.tokenOffset - tile0.tokenSize, input.totalTokens,
        tiling);

    EXPECT_EQ(tile0.tokenOffset, 0U);
    EXPECT_EQ(tile0.tokenSize, 14U);
    EXPECT_EQ(OUTPUT_PREPROCESS_BUFFER_ID, 0U);
    EXPECT_EQ(OUTPUT_QK_BUFFER_ID, 1U);
    EXPECT_EQ(tile1.tokenOffset, 14U);
    EXPECT_EQ(tile1.tokenSize, 9U);
    EXPECT_EQ(OUTPUT_PREPROCESS_BUFFER_ID, 0U);
    EXPECT_EQ(OUTPUT_QK_BUFFER_ID, 1U);
    EXPECT_EQ(tile0.tokenSize + tile1.tokenSize, range.end - range.begin);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, T512Aic25CoreTilesFollowFourteenTokenTile)
{
    ContractTiling tiling;
    auto input = BuildInput(512, 16, 2, 2);
    ASSERT_EQ(ValidateThenBuildTiling(input, tiling, 25), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling.tokenTile, 14U);
    ASSERT_EQ(tiling.tokenTilePerAiv, 7U);
    ASSERT_EQ(tiling.coreTokenTile, 21U);
    ASSERT_EQ(tiling.coreGroupNum, 25U);

    const auto group0 = MakeCoreTokenRangeForTest(input.totalTokens, 0, tiling);
    const auto group1 = MakeCoreTokenRangeForTest(input.totalTokens, 1, tiling);
    ASSERT_EQ(group0.begin, 0U);
    ASSERT_EQ(group0.end, 21U);
    ASSERT_EQ(group1.begin, 21U);
    ASSERT_EQ(group1.end, 42U);

    auto tile0 = MakeTileParamForTest(0, group0.begin, tiling.tokenTile, input.totalTokens, tiling);
    auto tile1 = MakeTileParamForTest(
        1, tile0.tokenOffset + tile0.tokenSize, group0.end - tile0.tokenOffset - tile0.tokenSize, input.totalTokens,
        tiling);

    EXPECT_EQ(tile0.tokenOffset, 0U);
    EXPECT_EQ(tile0.tokenSize, 14U);
    EXPECT_EQ(OUTPUT_PREPROCESS_BUFFER_ID, 0U);
    EXPECT_EQ(OUTPUT_QK_BUFFER_ID, 1U);
    EXPECT_EQ(tile1.tokenOffset, 14U);
    EXPECT_EQ(tile1.tokenSize, 7U);
    EXPECT_EQ(OUTPUT_PREPROCESS_BUFFER_ID, 0U);
    EXPECT_EQ(OUTPUT_QK_BUFFER_ID, 1U);
    const auto tile0Aiv0 = MakeAivSlice(tile0, 0);
    const auto tile1Aiv0 = MakeAivSlice(tile1, 0);
    const auto tile0Aiv1 = MakeAivSlice(tile0, 1);
    const auto tile1Aiv1 = MakeAivSlice(tile1, 1);
    EXPECT_EQ(tile0Aiv0.tokenSize, 7U);
    EXPECT_EQ(tile1Aiv0.tokenSize, 4U);
    EXPECT_EQ(tile1Aiv0.tokenOffset, 14U);
    EXPECT_EQ(tile0Aiv1.tokenSize, 7U);
    EXPECT_EQ(tile1Aiv1.tokenSize, 3U);
    EXPECT_TRUE(HasPreprocessWorkForTest(tile0Aiv1));
    EXPECT_TRUE(HasPreprocessWorkForTest(tile1Aiv1));
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, TilingDataDefaultsAreStable)
{
    optiling::QkvRmsNormRopeCacheWithKScaleTilingData tilingData;
    EXPECT_GT(sizeof(tilingData), 0U);
    EXPECT_EQ(sizeof(tilingData) % sizeof(uint64_t), 0U);
    EXPECT_EQ(tilingData.tokenTile, 0U);
    EXPECT_EQ(tilingData.coreTokenTile, 0U);
    EXPECT_EQ(tilingData.coreGroupNum, 0U);
    EXPECT_EQ(tilingData.kvCacheStrideBlock, 0U);
    EXPECT_EQ(tilingData.kScaleCacheStrideBlock, 0U);

    tilingData.totalTokens = 128;
    tilingData.qHeadNum = 16;
    tilingData.coreTokenTile = 4;
    tilingData.coreGroupNum = 32;
    tilingData.kvCacheStrideToken = 128;
    tilingData.kScaleCacheStrideToken = 1;
    tilingData.tokenTile = 14;
    EXPECT_EQ(tilingData.totalTokens, 128U);
    EXPECT_EQ(tilingData.qHeadNum, 16U);
    EXPECT_EQ(tilingData.coreTokenTile, 4U);
    EXPECT_EQ(tilingData.coreGroupNum, 32U);
    EXPECT_EQ(tilingData.kvCacheStrideToken, 128U);
    EXPECT_EQ(tilingData.kScaleCacheStrideToken, 1U);
    EXPECT_EQ(tilingData.tokenTile, 14U);
}
