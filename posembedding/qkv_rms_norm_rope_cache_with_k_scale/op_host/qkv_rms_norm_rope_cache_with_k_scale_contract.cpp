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
#include <limits>

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

bool FitsUint32(uint64_t value)
{
    return value <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
}

bool FitsByteStride(uint64_t value, uint32_t elementBytes)
{
    if (elementBytes == 0) {
        return false;
    }
    return value <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / elementBytes;
}

bool CacheMaxOffsetFitsUint32(
    const TensorStrideInfo &stride, uint32_t blockNum, uint32_t heads, uint32_t blockSize, uint32_t headDim)
{
    const uint64_t maxOffset = (static_cast<uint64_t>(blockNum) - 1U) * stride.stride0 +
        (static_cast<uint64_t>(heads) - 1U) * stride.stride1 +
        (static_cast<uint64_t>(blockSize) - 1U) * stride.stride2 + (static_cast<uint64_t>(headDim) - 1U);
    return FitsUint32(maxOffset);
}

bool ScaleMaxOffsetFitsUint32(
    const TensorStrideInfo &stride, uint32_t blockNum, uint32_t heads, uint32_t blockSize)
{
    const uint64_t maxOffset = (static_cast<uint64_t>(blockNum) - 1U) * stride.stride0 +
        (static_cast<uint64_t>(heads) - 1U) * stride.stride1 +
        (static_cast<uint64_t>(blockSize) - 1U) * stride.stride2;
    return FitsUint32(maxOffset);
}

bool IsCacheStrideValid(
    const TensorStrideInfo &stride, uint32_t blockNum, uint32_t heads, uint32_t blockSize, uint32_t headDim,
    uint32_t elementBytes)
{
    return stride.present && stride.rank == 4 && stride.stride0 > 0 && stride.stride1 >= headDim &&
           stride.stride2 >= headDim && stride.stride3 == 1 && FitsUint32(stride.stride0) &&
           FitsUint32(stride.stride1) && FitsUint32(stride.stride2) && FitsByteStride(stride.stride1 - headDim,
           elementBytes) && CacheMaxOffsetFitsUint32(stride, blockNum, heads, blockSize, headDim);
}

bool IsScaleStrideValid(const TensorStrideInfo &stride, uint32_t blockNum, uint32_t heads, uint32_t blockSize)
{
    return stride.present && stride.rank == 4 && stride.stride0 > 0 && stride.stride1 >= 1 && stride.stride2 >= 1 &&
           stride.stride3 == 1 && FitsUint32(stride.stride0) && FitsUint32(stride.stride1) &&
           FitsUint32(stride.stride2) && FitsByteStride(stride.stride1 - 1, sizeof(float)) &&
           ScaleMaxOffsetFitsUint32(stride, blockNum, heads, blockSize);
}

uint32_t CalcQkL0cRows(uint32_t tokenTile, uint32_t numQHeads, uint32_t numKHeads)
{
    return AlignUp(tokenTile * numQHeads, 16) + AlignUp(tokenTile * numKHeads, 16);
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
    if (input.aicNum == 0) {
        return ContractStatus::kUnsupportedResource;
    }
    if (!input.qRotation.shape.present) {
        return ContractStatus::kMissingQRotation;
    }
    if (!input.kRotation.shape.present) {
        return ContractStatus::kMissingKRotation;
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
        !HasDtype(input.qRotation, ContractDtype::kBf16) || !HasDtype(input.kRotation, ContractDtype::kBf16) ||
        !HasDtype(input.vScale, ContractDtype::kFloat)) {
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
    if (!IsShape2D(input.qRotation.shape, input.headDim, input.headDim)) {
        return ContractStatus::kInvalidQRotationShape;
    }
    if (!IsShape2D(input.kRotation.shape, input.headDim, input.headDim)) {
        return ContractStatus::kInvalidKRotationShape;
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
    if (!IsCacheStrideValid(input.kCache.stride, input.blockNum, input.numKHeads, input.blockSize, input.headDim, 1U) ||
        !IsCacheStrideValid(input.vCache.stride, input.blockNum, input.numVHeads, input.blockSize, input.headDim, 1U)) {
        return ContractStatus::kInvalidCacheStride;
    }
    if (!IsScaleStrideValid(input.kScaleCache.stride, input.blockNum, input.numKHeads, input.blockSize)) {
        return ContractStatus::kInvalidScaleStride;
    }

    uint32_t tokenTilePerAiv = std::min({kMaxTokenTile / 2, kUbOutputRowsPerAiv / hqk, kUbInputRowsPerAiv / htot});
    while (tokenTilePerAiv > 0) {
        const uint32_t tokenTile = 2 * tokenTilePerAiv;
        const uint32_t rowTile = tokenTile * hqk;
        const uint32_t rowTileAligned = AlignUp(rowTile, 16);
        if (CalcQkL0cRows(tokenTile, input.numQHeads, input.numKHeads) <= kL0aMaxRows &&
            rowTile <= kL0cMaxRows && rowTileAligned <= kL0cMaxRows &&
            CalcQkL0cRows(tokenTile, input.numQHeads, input.numKHeads) <= kL0cMaxRows) {
            break;
        }
        --tokenTilePerAiv;
    }
    if (tokenTilePerAiv == 0) {
        return ContractStatus::kUnsupportedResource;
    }

    tiling.coreTokenTile = CeilDiv(input.totalTokens, input.aicNum);
    tiling.coreGroupNum = CeilDiv(input.totalTokens, tiling.coreTokenTile);
    tiling.tokenTilePerAiv = tokenTilePerAiv;
    tiling.tokenTile = 2 * tokenTilePerAiv;
    tiling.qkHeadTile = hqk;
    tiling.vHeadTile = input.numVHeads;
    tiling.dimTile = kDimTileFast;
    tiling.rowTile = tiling.tokenTile * hqk;
    tiling.rowTileAligned = AlignUp(tiling.rowTile, 16);
    tiling.kCacheStrideBlock = static_cast<uint32_t>(input.kCache.stride.stride0);
    tiling.kCacheStrideHead = static_cast<uint32_t>(input.kCache.stride.stride1);
    tiling.kCacheStrideToken = static_cast<uint32_t>(input.kCache.stride.stride2);
    tiling.vCacheStrideBlock = static_cast<uint32_t>(input.vCache.stride.stride0);
    tiling.vCacheStrideHead = static_cast<uint32_t>(input.vCache.stride.stride1);
    tiling.vCacheStrideToken = static_cast<uint32_t>(input.vCache.stride.stride2);
    tiling.kScaleCacheStrideBlock = static_cast<uint32_t>(input.kScaleCache.stride.stride0);
    tiling.kScaleCacheStrideHead = static_cast<uint32_t>(input.kScaleCache.stride.stride1);
    tiling.kScaleCacheStrideToken = static_cast<uint32_t>(input.kScaleCache.stride.stride2);

    return ContractStatus::kSuccess;
}

uint32_t CalcTileNum(uint32_t totalTokens, uint32_t tokenTile)
{
    return CeilDiv(totalTokens, tokenTile);
}

uint32_t CalcCoreTokenTile(uint32_t totalTokens, uint32_t aicNum)
{
    return CeilDiv(totalTokens, aicNum);
}

uint32_t CalcCoreGroupNum(uint32_t totalTokens, uint32_t aicNum)
{
    const uint32_t coreTokenTile = CalcCoreTokenTile(totalTokens, aicNum);
    return CeilDiv(totalTokens, coreTokenTile);
}

AivSlice MakeAivSlice(const TileParam &tile, uint32_t qkHeadTile, uint32_t aivLocalId)
{
    AivSlice slice;
    const uint32_t aiv0TokenSize = CeilDiv(tile.tokenSize, kAivPerAic);
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
        case ContractStatus::kMissingQRotation:
            return "missing_q_rotation";
        case ContractStatus::kMissingKRotation:
            return "missing_k_rotation";
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
        case ContractStatus::kInvalidQRotationShape:
            return "invalid_q_rotation_shape";
        case ContractStatus::kInvalidKRotationShape:
            return "invalid_k_rotation_shape";
        case ContractStatus::kInvalidVScaleShape:
            return "invalid_v_scale_shape";
        case ContractStatus::kInvalidCacheShape:
            return "invalid_cache_shape";
        case ContractStatus::kInvalidScaleShape:
            return "invalid_scale_shape";
        case ContractStatus::kInvalidCacheStride:
            return "invalid_cache_stride";
        case ContractStatus::kInvalidScaleStride:
            return "invalid_scale_stride";
        case ContractStatus::kUnsupportedResource:
            return "unsupported_resource";
        default:
            return "unknown";
    }
}

} // namespace qkv_rms_norm_rope_cache_with_k_scale
} // namespace optiling
