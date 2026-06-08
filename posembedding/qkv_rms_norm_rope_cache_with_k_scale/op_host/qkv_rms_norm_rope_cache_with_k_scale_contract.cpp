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

#include "log/log.h"

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

bool IsCacheStrideValid(const TensorStrideInfo &stride, uint64_t headDim)
{
    return stride.present && stride.rank == 4 && stride.stride0 > 0 && stride.stride1 >= headDim &&
           stride.stride2 >= headDim && stride.stride3 == 1;
}

bool IsScaleStrideValid(const TensorStrideInfo &stride)
{
    return stride.present && stride.rank == 4 && stride.stride0 > 0 && stride.stride1 >= 1 && stride.stride2 >= 1 &&
           stride.stride3 == 1;
}

bool CacheNonInnermostStridesMatch(const TensorStrideInfo &lhs, const TensorStrideInfo &rhs)
{
    return lhs.stride0 == rhs.stride0 && lhs.stride1 == rhs.stride1 && lhs.stride2 == rhs.stride2;
}

uint64_t CalcQkAlignedRows(uint64_t tokenTile, uint64_t numQHeads, uint64_t numKHeads)
{
    return AlignUp(tokenTile * numQHeads, 16) + AlignUp(tokenTile * numKHeads, 16);
}

bool ShouldLog(const char *opName)
{
    return opName != nullptr && opName[0] != '\0';
}

void LogContractInputSummary(const char *opName, const ContractInput &input, uint64_t hqk, uint64_t htot)
{
    if (!ShouldLog(opName)) {
        return;
    }
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale contract input: T=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu hqk=%lu htot=%lu "
            "headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.totalTokens, input.batch, input.numQHeads, input.numKHeads, input.numVHeads, hqk, htot,
            input.headDim, input.maxSeqLen, input.blockNum, input.blockSize, input.aicNum);
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale resource constants: maxTokenTile=%lu ubInputRowsPerAiv=%lu "
            "ubOutputRowsPerAiv=%lu l0aMaxRows=%lu l0cMaxRows=%lu aivPerAic=%lu.",
            kMaxTokenTile, kUbInputRowsPerAiv, kUbOutputRowsPerAiv, kL0aMaxRows, kL0cMaxRows, kAivPerAic);
}

void LogTokenTileCandidate(const char *opName, uint64_t tokenTilePerAiv, uint64_t tokenTile, uint64_t rowTile,
                           uint64_t rowTileAligned, uint64_t qkAlignedRows, bool fitsUb, bool fitsL0a, bool fitsL0c,
                           bool accepted)
{
    if (!ShouldLog(opName)) {
        return;
    }
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale token tile candidate: tokenTilePerAiv=%lu tokenTile=%lu rowTile=%lu "
            "rowTileAligned=%lu qkAlignedRows=%lu fitsUb=%u fitsL0a=%u fitsL0c=%u accepted=%u.",
            tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb ? 1U : 0U,
            fitsL0a ? 1U : 0U, fitsL0c ? 1U : 0U, accepted ? 1U : 0U);
}

} // namespace

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

uint64_t CalcCubeTokenSize(uint64_t tokenSize)
{
    return AlignUp(tokenSize, kAivPerAic);
}

uint64_t CalcCubeHalfTokenSize(uint64_t tokenSize)
{
    return CalcCubeTokenSize(tokenSize) / kAivPerAic;
}

ContractStatus ValidateAndBuildTiling(const ContractInput &input, ContractTiling &tiling, const char *opName)
{
    tiling = {};
    if (ShouldLog(opName)) {
        OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract validation start.");
    }
    if (input.totalTokens == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: totalTokens is 0.");
        }
        return ContractStatus::kInvalidTotalTokens;
    }
    if (input.batch == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: batch is 0.");
        }
        return ContractStatus::kInvalidBatch;
    }
    if (input.numQHeads == 0 || input.numKHeads == 0 || input.numVHeads == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: invalid head nums Nq=%lu Nk=%lu Nv=%lu.",
                    input.numQHeads, input.numKHeads, input.numVHeads);
        }
        return ContractStatus::kInvalidHeadNums;
    }
    if (input.numVHeads != input.numKHeads) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: numVHeads[%lu] != numKHeads[%lu].",
                    input.numVHeads, input.numKHeads);
        }
        return ContractStatus::kUnsupportedNvNotEqualNk;
    }
    if (input.headDim != kHeadDimFast) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: unsupported headDim=%lu.",
                    input.headDim);
        }
        return ContractStatus::kUnsupportedHeadDim;
    }
    if (input.maxSeqLen == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: maxSeqLen is 0.");
        }
        return ContractStatus::kInvalidMaxSeqLen;
    }
    if (input.blockNum == 0 || input.blockSize == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: blockNum=%lu blockSize=%lu.",
                    input.blockNum, input.blockSize);
        }
        return ContractStatus::kInvalidBlockShape;
    }
    if (input.aicNum == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: aicNum is 0.");
        }
        return ContractStatus::kUnsupportedResource;
    }
    if (!input.qRotation.shape.present) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: qRotation is missing.");
        }
        return ContractStatus::kMissingQRotation;
    }
    if (!input.kRotation.shape.present) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: kRotation is missing.");
        }
        return ContractStatus::kMissingKRotation;
    }
    if (!input.vScale.shape.present) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: vScale is missing.");
        }
        return ContractStatus::kMissingVScale;
    }

    const uint64_t hqk = input.numQHeads + input.numKHeads;
    const uint64_t htot = input.numQHeads + input.numKHeads + input.numVHeads;
    LogContractInputSummary(opName, input, hqk, htot);
    if (hqk == 0 || htot == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: derived hqk=%lu htot=%lu.", hqk,
                    htot);
        }
        return ContractStatus::kInvalidHeadNums;
    }

    if (!HasDtype(input.qkv, ContractDtype::kBf16) || !HasDtype(input.qGamma, ContractDtype::kFloat) ||
        !HasDtype(input.kGamma, ContractDtype::kFloat) || !HasDtype(input.cosSin, ContractDtype::kFloat) ||
        !HasDtype(input.slotMapping, ContractDtype::kInt32) || !HasDtype(input.kCache, ContractDtype::kFloat8E4M3fn) ||
        !HasDtype(input.vCache, ContractDtype::kFloat8E4M3fn) || !HasDtype(input.kScaleCache, ContractDtype::kFloat) ||
        !HasDtype(input.cuSeqLen, ContractDtype::kInt32) || !HasDtype(input.qRotation, ContractDtype::kBf16) ||
        !HasDtype(input.kRotation, ContractDtype::kBf16) || !HasDtype(input.vScale, ContractDtype::kFloat)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: dtype check failed qkv=%u qGamma=%u kGamma=%u "
                    "cosSin=%u slotMapping=%u kCache=%u vCache=%u kScaleCache=%u cuSeqLen=%u qRotation=%u "
                    "kRotation=%u vScale=%u.",
                    static_cast<uint32_t>(input.qkv.dtype), static_cast<uint32_t>(input.qGamma.dtype),
                    static_cast<uint32_t>(input.kGamma.dtype), static_cast<uint32_t>(input.cosSin.dtype),
                    static_cast<uint32_t>(input.slotMapping.dtype), static_cast<uint32_t>(input.kCache.dtype),
                    static_cast<uint32_t>(input.vCache.dtype), static_cast<uint32_t>(input.kScaleCache.dtype),
                    static_cast<uint32_t>(input.cuSeqLen.dtype), static_cast<uint32_t>(input.qRotation.dtype),
                    static_cast<uint32_t>(input.kRotation.dtype), static_cast<uint32_t>(input.vScale.dtype));
        }
        return ContractStatus::kInvalidDtype;
    }

    if (!IsShape3D(input.qkv.shape, input.totalTokens, htot, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: qkv shape rank=%u dims={%lu,%lu,%lu,%lu}.",
                    input.qkv.shape.rank, input.qkv.shape.dim0, input.qkv.shape.dim1, input.qkv.shape.dim2,
                    input.qkv.shape.dim3);
        }
        return ContractStatus::kInvalidQkvShape;
    }
    if (!IsShape1D(input.qGamma.shape, input.headDim) || !IsShape1D(input.kGamma.shape, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: gamma shape qGamma rank=%u dim0=%lu, "
                    "kGamma rank=%u dim0=%lu.",
                    input.qGamma.shape.rank, input.qGamma.shape.dim0, input.kGamma.shape.rank,
                    input.kGamma.shape.dim0);
        }
        return ContractStatus::kInvalidGammaShape;
    }
    if (!IsShape2D(input.cosSin.shape, input.maxSeqLen, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: cosSin shape rank=%u dims={%lu,%lu}.",
                    input.cosSin.shape.rank, input.cosSin.shape.dim0, input.cosSin.shape.dim1);
        }
        return ContractStatus::kInvalidCosSinShape;
    }
    if (!IsShape1D(input.slotMapping.shape, input.totalTokens)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: slotMapping shape rank=%u dim0=%lu.",
                    input.slotMapping.shape.rank, input.slotMapping.shape.dim0);
        }
        return ContractStatus::kInvalidSlotMappingShape;
    }
    if (!IsShape1D(input.cuSeqLen.shape, static_cast<uint64_t>(input.batch) + 1)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: cuSeqLen shape rank=%u dim0=%lu.",
                    input.cuSeqLen.shape.rank, input.cuSeqLen.shape.dim0);
        }
        return ContractStatus::kInvalidCuSeqLenShape;
    }
    if (!IsShape2D(input.qRotation.shape, input.headDim, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: qRotation shape rank=%u dims={%lu,%lu}.",
                    input.qRotation.shape.rank, input.qRotation.shape.dim0, input.qRotation.shape.dim1);
        }
        return ContractStatus::kInvalidQRotationShape;
    }
    if (!IsShape2D(input.kRotation.shape, input.headDim, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: kRotation shape rank=%u dims={%lu,%lu}.",
                    input.kRotation.shape.rank, input.kRotation.shape.dim0, input.kRotation.shape.dim1);
        }
        return ContractStatus::kInvalidKRotationShape;
    }
    if (!IsShape1D(input.vScale.shape, input.numVHeads)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: vScale shape rank=%u dim0=%lu.",
                    input.vScale.shape.rank, input.vScale.shape.dim0);
        }
        return ContractStatus::kInvalidVScaleShape;
    }
    if (!IsShape4D(input.kCache.shape, input.blockNum, input.numKHeads, input.blockSize, input.headDim) ||
        !IsShape4D(input.vCache.shape, input.blockNum, input.numVHeads, input.blockSize, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: cache shape kCache rank=%u dims={%lu,%lu,%lu,%lu}, "
                    "vCache rank=%u dims={%lu,%lu,%lu,%lu}.",
                    input.kCache.shape.rank, input.kCache.shape.dim0, input.kCache.shape.dim1,
                    input.kCache.shape.dim2, input.kCache.shape.dim3, input.vCache.shape.rank,
                    input.vCache.shape.dim0, input.vCache.shape.dim1, input.vCache.shape.dim2,
                    input.vCache.shape.dim3);
        }
        return ContractStatus::kInvalidCacheShape;
    }
    if (!IsShape4D(input.kScaleCache.shape, input.blockNum, input.numKHeads, input.blockSize, 1)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache shape rank=%u dims={%lu,%lu,%lu,%lu}.",
                    input.kScaleCache.shape.rank, input.kScaleCache.shape.dim0, input.kScaleCache.shape.dim1,
                    input.kScaleCache.shape.dim2, input.kScaleCache.shape.dim3);
        }
        return ContractStatus::kInvalidScaleShape;
    }
    if (!IsCacheStrideValid(input.kCache.stride, input.headDim) ||
        !IsCacheStrideValid(input.vCache.stride, input.headDim)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: cache stride kCache present=%u rank=%u "
                    "strides={%lu,%lu,%lu,%lu}, vCache present=%u rank=%u strides={%lu,%lu,%lu,%lu}.",
                    input.kCache.stride.present ? 1U : 0U, input.kCache.stride.rank,
                    input.kCache.stride.stride0, input.kCache.stride.stride1, input.kCache.stride.stride2,
                    input.kCache.stride.stride3, input.vCache.stride.present ? 1U : 0U, input.vCache.stride.rank,
                    input.vCache.stride.stride0, input.vCache.stride.stride1, input.vCache.stride.stride2,
                    input.vCache.stride.stride3);
        }
        return ContractStatus::kInvalidCacheStride;
    }
    if (!CacheNonInnermostStridesMatch(input.kCache.stride, input.vCache.stride)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: k/v cache non-innermost strides mismatch "
                    "k={%lu,%lu,%lu} v={%lu,%lu,%lu}.",
                    input.kCache.stride.stride0, input.kCache.stride.stride1, input.kCache.stride.stride2,
                    input.vCache.stride.stride0, input.vCache.stride.stride1, input.vCache.stride.stride2);
        }
        return ContractStatus::kInvalidCacheStride;
    }
    if (!IsScaleStrideValid(input.kScaleCache.stride)) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName,
                    "QkvRmsNormRopeCacheWithKScale contract reject: kScaleCache stride present=%u rank=%u "
                    "strides={%lu,%lu,%lu,%lu}.",
                    input.kScaleCache.stride.present ? 1U : 0U, input.kScaleCache.stride.rank,
                    input.kScaleCache.stride.stride0, input.kScaleCache.stride.stride1,
                    input.kScaleCache.stride.stride2, input.kScaleCache.stride.stride3);
        }
        return ContractStatus::kInvalidScaleStride;
    }

    const uint64_t initialTokenTilePerAiv =
        std::min({kMaxTokenTile / 2, kUbOutputRowsPerAiv / hqk, kUbInputRowsPerAiv / htot});
    if (ShouldLog(opName)) {
        OP_LOGD(opName,
                "QkvRmsNormRopeCacheWithKScale token tile search start: initialTokenTilePerAiv=%lu "
                "ubOutputLimit=%lu ubInputLimit=%lu.",
                initialTokenTilePerAiv, kUbOutputRowsPerAiv / hqk, kUbInputRowsPerAiv / htot);
    }
    uint64_t selectedTokenTile = 0;
    uint64_t selectedTokenTilePerAiv = 0;
    const auto trySelectTokenTile = [&](uint64_t tokenTile) {
        const uint64_t tokenTilePerAiv = CalcCubeHalfTokenSize(tokenTile);
        const uint64_t rowTile = tokenTile * hqk;
        const uint64_t rowTileAligned = AlignUp(rowTile, 16);
        const uint64_t qkAlignedRows = CalcQkAlignedRows(tokenTile, input.numQHeads, input.numKHeads);
        const bool fitsUb = tokenTilePerAiv * hqk <= kUbOutputRowsPerAiv &&
                            tokenTilePerAiv * htot <= kUbInputRowsPerAiv;
        const bool fitsL0a = qkAlignedRows <= kL0aMaxRows;
        const bool fitsL0c = rowTile <= kL0cMaxRows && rowTileAligned <= kL0cMaxRows &&
                             qkAlignedRows <= kL0cMaxRows;
        const bool accepted = fitsUb && fitsL0a && fitsL0c;
        LogTokenTileCandidate(opName, tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, fitsUb,
                              fitsL0a, fitsL0c, accepted);
        if (accepted) {
            selectedTokenTile = tokenTile;
            selectedTokenTilePerAiv = tokenTilePerAiv;
            return true;
        }
        return false;
    };
    uint64_t searchTokenTilePerAiv = initialTokenTilePerAiv;
    while (searchTokenTilePerAiv > 0) {
        if (trySelectTokenTile(2 * searchTokenTilePerAiv)) {
            break;
        }
        --searchTokenTilePerAiv;
    }
    if (selectedTokenTile == 0) {
        (void)trySelectTokenTile(1);
    }
    if (selectedTokenTile == 0) {
        if (ShouldLog(opName)) {
            OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale contract reject: no token tile candidate fits.");
        }
        return ContractStatus::kUnsupportedResource;
    }

    tiling.coreTokenTile = CeilDiv(input.totalTokens, input.aicNum);
    tiling.coreGroupNum = CeilDiv(input.totalTokens, tiling.coreTokenTile);
    tiling.tokenTilePerAiv = selectedTokenTilePerAiv;
    tiling.tokenTile = selectedTokenTile;
    tiling.qkHeadTile = hqk;
    tiling.vHeadTile = input.numVHeads;
    tiling.dimTile = kDimTileFast;
    tiling.rowTile = tiling.tokenTile * hqk;
    tiling.rowTileAligned = AlignUp(tiling.rowTile, 16);
    tiling.kCacheStrideBlock = input.kCache.stride.stride0;
    tiling.kCacheStrideHead = input.kCache.stride.stride1;
    tiling.kCacheStrideToken = input.kCache.stride.stride2;
    tiling.vCacheStrideBlock = input.vCache.stride.stride0;
    tiling.vCacheStrideHead = input.vCache.stride.stride1;
    tiling.vCacheStrideToken = input.vCache.stride.stride2;
    tiling.kScaleCacheStrideBlock = input.kScaleCache.stride.stride0;
    tiling.kScaleCacheStrideHead = input.kScaleCache.stride.stride1;
    tiling.kScaleCacheStrideToken = input.kScaleCache.stride.stride2;

    if (ShouldLog(opName)) {
        OP_LOGD(opName,
                "QkvRmsNormRopeCacheWithKScale contract tiling result: coreTokenTile=%lu coreGroupNum=%lu "
                "tokenTile=%lu tokenTilePerAiv=%lu qkHeadTile=%lu vHeadTile=%lu rowTile=%lu rowTileAligned=%lu "
                "dimTile=%lu.",
                tiling.coreTokenTile, tiling.coreGroupNum, tiling.tokenTile, tiling.tokenTilePerAiv,
                tiling.qkHeadTile, tiling.vHeadTile, tiling.rowTile, tiling.rowTileAligned, tiling.dimTile);
        OP_LOGD(opName,
                "QkvRmsNormRopeCacheWithKScale contract stride result: kCache={%lu,%lu,%lu} "
                "vCache={%lu,%lu,%lu} kScaleCache={%lu,%lu,%lu}.",
                tiling.kCacheStrideBlock, tiling.kCacheStrideHead, tiling.kCacheStrideToken,
                tiling.vCacheStrideBlock, tiling.vCacheStrideHead, tiling.vCacheStrideToken,
                tiling.kScaleCacheStrideBlock, tiling.kScaleCacheStrideHead, tiling.kScaleCacheStrideToken);
    }
    return ContractStatus::kSuccess;
}

uint64_t CalcTileNum(uint64_t totalTokens, uint64_t tokenTile)
{
    return CeilDiv(totalTokens, tokenTile);
}

uint64_t CalcCoreTokenTile(uint64_t totalTokens, uint64_t aicNum)
{
    return CeilDiv(totalTokens, aicNum);
}

uint64_t CalcCoreGroupNum(uint64_t totalTokens, uint64_t aicNum)
{
    const uint64_t coreTokenTile = CalcCoreTokenTile(totalTokens, aicNum);
    return CeilDiv(totalTokens, coreTokenTile);
}

AivSlice MakeAivSlice(const TileParam &tile, uint64_t qkHeadTile, uint64_t aivLocalId)
{
    AivSlice slice;
    const uint64_t cubeHalfTokenSize = CalcCubeHalfTokenSize(tile.tokenSize);
    const uint64_t tokenPrefix = aivLocalId == 0 ? 0 : cubeHalfTokenSize;
    const uint64_t tokenSize = aivLocalId == 0 ? cubeHalfTokenSize : tile.tokenSize - cubeHalfTokenSize;
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
