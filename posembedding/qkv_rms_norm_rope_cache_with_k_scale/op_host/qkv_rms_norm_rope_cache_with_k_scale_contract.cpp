/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale_contract.h"

#include <algorithm>

namespace optiling {
namespace qkv_rms_norm_rope_cache_with_k_scale {
namespace {

bool IsShape1D(const TensorShapeInfo &shape, uint64_t d0)
{
    return shape.present && shape.rank == 1 && shape.dim0 == d0;
}

bool IsShape2D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1)
{
    return shape.present && shape.rank == 2 && shape.dim0 == d0 && shape.dim1 == d1;
}

bool IsShape3D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1, uint64_t d2)
{
    return shape.present && shape.rank == 3 && shape.dim0 == d0 && shape.dim1 == d1 && shape.dim2 == d2;
}

bool IsShape4D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    return shape.present && shape.rank == 4 && shape.dim0 == d0 && shape.dim1 == d1 && shape.dim2 == d2 &&
           shape.dim3 == d3;
}

bool HasDtype(const TensorContractInfo &tensor, ContractDtype expected)
{
    return tensor.dtype == expected;
}

} // namespace

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

ContractStatus ValidateAndBuildTiling(const ContractInput &input, ContractTiling &tiling)
{
    tiling = {};
    if (input.totalTokens == 0) {
        return ContractStatus::kInvalidTotalTokens;
    }
    if (input.batch == 0) {
        return ContractStatus::kInvalidBatch;
    }
    if (input.numQHeads == 0 || input.numKHeads == 0 || input.numVHeads == 0) {
        return ContractStatus::kInvalidHeadNums;
    }
    if (input.numVHeads != input.numKHeads) {
        return ContractStatus::kUnsupportedNvNotEqualNk;
    }
    if (input.headDim != kHeadDimFast) {
        return ContractStatus::kUnsupportedHeadDim;
    }
    if (input.maxSeqLen == 0) {
        return ContractStatus::kInvalidMaxSeqLen;
    }
    if (input.blockNum == 0 || input.blockSize == 0) {
        return ContractStatus::kInvalidBlockShape;
    }
    if (!input.rotation.shape.present) {
        return ContractStatus::kMissingRotation;
    }
    if (!input.vScale.shape.present) {
        return ContractStatus::kMissingVScale;
    }

    const uint32_t hqk = input.numQHeads + input.numKHeads;
    const uint32_t htot = input.numQHeads + input.numKHeads + input.numVHeads;
    if (hqk == 0 || htot == 0) {
        return ContractStatus::kInvalidHeadNums;
    }

    if (!HasDtype(input.qkv, ContractDtype::kBf16) || !HasDtype(input.qGamma, ContractDtype::kFloat) ||
        !HasDtype(input.kGamma, ContractDtype::kFloat) || !HasDtype(input.cosSin, ContractDtype::kFloat) ||
        !HasDtype(input.slotMapping, ContractDtype::kInt32) ||
        !HasDtype(input.kCache, ContractDtype::kFloat8E4M3fn) ||
        !HasDtype(input.vCache, ContractDtype::kFloat8E4M3fn) ||
        !HasDtype(input.kScaleCache, ContractDtype::kFloat) || !HasDtype(input.cuSeqLen, ContractDtype::kInt32) ||
        !HasDtype(input.rotation, ContractDtype::kBf16) || !HasDtype(input.vScale, ContractDtype::kFloat)) {
        return ContractStatus::kInvalidDtype;
    }

    if (!IsShape3D(input.qkv.shape, input.totalTokens, htot, input.headDim)) {
        return ContractStatus::kInvalidQkvShape;
    }
    if (!IsShape1D(input.qGamma.shape, input.headDim) || !IsShape1D(input.kGamma.shape, input.headDim)) {
        return ContractStatus::kInvalidGammaShape;
    }
    if (!IsShape2D(input.cosSin.shape, input.maxSeqLen, input.headDim)) {
        return ContractStatus::kInvalidCosSinShape;
    }
    if (!IsShape1D(input.slotMapping.shape, input.totalTokens)) {
        return ContractStatus::kInvalidSlotMappingShape;
    }
    if (!IsShape1D(input.cuSeqLen.shape, static_cast<uint64_t>(input.batch) + 1)) {
        return ContractStatus::kInvalidCuSeqLenShape;
    }
    if (!IsShape2D(input.rotation.shape, input.headDim, input.headDim)) {
        return ContractStatus::kInvalidRotationShape;
    }
    if (!IsShape1D(input.vScale.shape, input.numVHeads)) {
        return ContractStatus::kInvalidVScaleShape;
    }
    if (!IsShape4D(input.kCache.shape, input.blockNum, input.numKHeads, input.blockSize, input.headDim) ||
        !IsShape4D(input.vCache.shape, input.blockNum, input.numVHeads, input.blockSize, input.headDim)) {
        return ContractStatus::kInvalidCacheShape;
    }
    if (!IsShape4D(input.kScaleCache.shape, input.blockNum, input.numKHeads, input.blockSize, 1)) {
        return ContractStatus::kInvalidScaleShape;
    }

    const uint32_t taMax = std::min({kMaxTokenTile / 2, kUbOutputRowsPerAiv / hqk, kUbInputRowsPerAiv / htot});
    if (taMax == 0) {
        return ContractStatus::kUnsupportedResource;
    }

    tiling.tokenTilePerAiv = taMax;
    tiling.tokenTile = 2 * taMax;
    tiling.qkHeadTile = hqk;
    tiling.vHeadTile = input.numVHeads;
    tiling.dimTile = kDimTileFast;
    tiling.rowTile = tiling.tokenTile * hqk;
    tiling.rowTileAligned = AlignUp(tiling.rowTile, 16);
    if (tiling.rowTile > kL0cMaxRows || tiling.rowTileAligned > kL0cMaxRows) {
        return ContractStatus::kUnsupportedResource;
    }

    return ContractStatus::kSuccess;
}

uint32_t CalcTileNum(uint32_t totalTokens, uint32_t tokenTile)
{
    return CeilDiv(totalTokens, tokenTile);
}

TileRange MakeTileRange(uint32_t totalTiles, uint32_t cubeIndex, uint32_t coreGroupNum)
{
    TileRange range;
    if (coreGroupNum == 0 || cubeIndex >= coreGroupNum || cubeIndex >= totalTiles) {
        return range;
    }
    range.begin = cubeIndex;
    range.end = totalTiles;
    range.stride = coreGroupNum;
    return range;
}

TileParam MakeTileParam(uint32_t tileId, uint32_t totalTokens, const ContractTiling &tiling)
{
    TileParam tile;
    tile.tileId = tileId;
    tile.bufferId = tileId & 1U;
    tile.tokenOffset = tileId * tiling.tokenTile;
    if (tile.tokenOffset >= totalTokens) {
        return tile;
    }
    tile.tokenSize = std::min(tiling.tokenTile, totalTokens - tile.tokenOffset);
    tile.qkHeadOffset = 0;
    tile.qkHeadSize = tiling.qkHeadTile;
    tile.vHeadOffset = 0;
    tile.vHeadSize = tiling.vHeadTile;
    tile.rowOffset = 0;
    tile.rowSize = tile.tokenSize * tiling.qkHeadTile;
    return tile;
}

AivSlice MakeAivSlice(const TileParam &tile, uint32_t qkHeadTile, uint32_t aivLocalId)
{
    AivSlice slice;
    const uint32_t aiv0TokenSize = CeilDiv(tile.tokenSize, 2);
    const uint32_t tokenPrefix = aivLocalId == 0 ? 0 : aiv0TokenSize;
    const uint32_t tokenSize = aivLocalId == 0 ? aiv0TokenSize : tile.tokenSize - aiv0TokenSize;
    slice.tokenOffset = tile.tokenOffset + tokenPrefix;
    slice.tokenSize = tokenSize;
    slice.rowOffset = tokenPrefix * qkHeadTile;
    slice.rowSize = tokenSize * qkHeadTile;
    return slice;
}

const char *ContractStatusName(ContractStatus status)
{
    switch (status) {
        case ContractStatus::kSuccess:
            return "success";
        case ContractStatus::kInvalidTotalTokens:
            return "invalid_total_tokens";
        case ContractStatus::kInvalidBatch:
            return "invalid_batch";
        case ContractStatus::kInvalidHeadNums:
            return "invalid_head_nums";
        case ContractStatus::kInvalidDtype:
            return "invalid_dtype";
        case ContractStatus::kUnsupportedNvNotEqualNk:
            return "unsupported_nv_not_equal_nk";
        case ContractStatus::kUnsupportedHeadDim:
            return "unsupported_head_dim";
        case ContractStatus::kInvalidMaxSeqLen:
            return "invalid_max_seq_len";
        case ContractStatus::kInvalidBlockShape:
            return "invalid_block_shape";
        case ContractStatus::kMissingRotation:
            return "missing_rotation";
        case ContractStatus::kMissingVScale:
            return "missing_v_scale";
        case ContractStatus::kInvalidQkvShape:
            return "invalid_qkv_shape";
        case ContractStatus::kInvalidGammaShape:
            return "invalid_gamma_shape";
        case ContractStatus::kInvalidCosSinShape:
            return "invalid_cos_sin_shape";
        case ContractStatus::kInvalidSlotMappingShape:
            return "invalid_slot_mapping_shape";
        case ContractStatus::kInvalidCuSeqLenShape:
            return "invalid_cu_seq_len_shape";
        case ContractStatus::kInvalidRotationShape:
            return "invalid_rotation_shape";
        case ContractStatus::kInvalidVScaleShape:
            return "invalid_v_scale_shape";
        case ContractStatus::kInvalidCacheShape:
            return "invalid_cache_shape";
        case ContractStatus::kInvalidScaleShape:
            return "invalid_scale_shape";
        case ContractStatus::kUnsupportedResource:
            return "unsupported_resource";
        default:
            return "unknown";
    }
}

} // namespace qkv_rms_norm_rope_cache_with_k_scale
} // namespace optiling
