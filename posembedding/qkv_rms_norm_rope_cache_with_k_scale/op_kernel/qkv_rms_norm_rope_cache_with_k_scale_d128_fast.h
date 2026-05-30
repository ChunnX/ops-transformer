/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_

#include "kernel_operator.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h"

#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST 350128

namespace QkvRmsNormRopeCacheWithKScale {
using namespace AscendC;
using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

constexpr uint32_t QKV_RRC_MIX_AIV_PER_AIC = 2U;

struct TokenRange {
    uint32_t begin;
    uint32_t end;
};

struct TileParam {
    uint32_t tileId;
    uint32_t bufferId;
    uint32_t tokenOffset;
    uint32_t tokenSize;
    uint32_t qkHeadOffset;
    uint32_t qkHeadSize;
    uint32_t vHeadOffset;
    uint32_t vHeadSize;
    uint32_t rowOffset;
    uint32_t rowSize;
};

struct AivSlice {
    uint32_t tokenOffset;
    uint32_t tokenSize;
    uint32_t rowOffset;
    uint32_t rowSize;
};

struct GlobalTensors {
    GM_ADDR qkv;
    GM_ADDR qGamma;
    GM_ADDR kGamma;
    GM_ADDR cosSin;
    GM_ADDR slotMapping;
    GM_ADDR kCache;
    GM_ADDR vCache;
    GM_ADDR kScaleCache;
    GM_ADDR cuSeqLen;
    GM_ADDR rotation;
    GM_ADDR vScale;
    GM_ADDR qOut;
    GM_ADDR qScale;
    GM_ADDR kCacheOut;
    GM_ADDR vCacheOut;
    GM_ADDR kScaleCacheOut;
    GM_ADDR workspace;
};

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return 0U;
    }
    return (value + factor - 1U) / factor;
}

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t GetCubeIndex()
{
    if (g_coreType == AIC) {
        return GetBlockIdx();
    }
    return GetBlockIdx() / QKV_RRC_MIX_AIV_PER_AIC;
}

__aicore__ inline uint32_t GetCubeGroupNum()
{
    if (g_coreType == AIC) {
        return GetBlockNum();
    }
    return CeilDiv(GetBlockNum(), QKV_RRC_MIX_AIV_PER_AIC);
}

__aicore__ inline TokenRange MakeCoreTokenRange(uint32_t totalTokens, uint32_t cubeIndex, uint32_t cubeGroupNum)
{
    TokenRange range = {0U, 0U};
    if (cubeGroupNum == 0U || cubeIndex >= cubeGroupNum || totalTokens == 0U) {
        return range;
    }

    const uint32_t baseTokenSize = totalTokens / cubeGroupNum;
    const uint32_t extraTokenNum = totalTokens - baseTokenSize * cubeGroupNum;
    range.begin = cubeIndex * baseTokenSize + MinU32(cubeIndex, extraTokenNum);
    range.end = range.begin + baseTokenSize + (cubeIndex < extraTokenNum ? 1U : 0U);
    return range;
}

__aicore__ inline TileParam MakeTileParam(uint32_t tileId, uint32_t tokenOffset, uint32_t tokenSize,
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
{
    TileParam tile = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
    tile.tileId = tileId;
    tile.bufferId = tileId & 1U;
    tile.tokenOffset = tokenOffset;
    if (tile.tokenOffset >= tilingData->totalTokens || tokenSize == 0U) {
        return tile;
    }

    tile.tokenSize = MinU32(MinU32(tokenSize, tilingData->tokenTile), tilingData->totalTokens - tile.tokenOffset);
    tile.qkHeadOffset = 0U;
    tile.qkHeadSize = tilingData->qkHeadTile;
    tile.vHeadOffset = 0U;
    tile.vHeadSize = tilingData->vHeadTile;
    tile.rowOffset = 0U;
    tile.rowSize = tile.tokenSize * tilingData->qkHeadTile;
    return tile;
}

__aicore__ inline AivSlice MakeAivSlice(const TileParam &tile, uint32_t qkHeadTile,
    uint32_t aivLocalId)
{
    AivSlice slice = {0U, 0U, 0U, 0U};
    const uint32_t aiv0TokenSize = CeilDiv(tile.tokenSize, QKV_RRC_MIX_AIV_PER_AIC);
    const uint32_t tokenPrefix = aivLocalId == 0U ? 0U : aiv0TokenSize;
    const uint32_t tokenSize = aivLocalId == 0U ? aiv0TokenSize : tile.tokenSize - aiv0TokenSize;

    slice.tokenOffset = tile.tokenOffset + tokenPrefix;
    slice.tokenSize = tokenSize;
    slice.rowOffset = tokenPrefix * qkHeadTile;
    slice.rowSize = tokenSize * qkHeadTile;
    return slice;
}

class D128FastBasicBlock {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, GM_ADDR workspace)
    {
        tensors_ = tensors;
        tilingData_ = tilingData;
        workspace_ = workspace;
        cubeIndex_ = GetCubeIndex();
        aivLocalId_ = g_coreType == AIC ? 0U : GetSubBlockIdx();
        InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        if (g_coreType == AIC) {
            ComputeAicTile(tile);
        } else {
            ComputeAivTile(tile);
        }
    }

    __aicore__ inline void End()
    {
        EndIntraCoreEvents();
    }

private:
    __aicore__ inline void InitIntraCoreEvents()
    {
        // T05 will add the real event-id initialization. T01 keeps lifecycle hooks in place.
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        // T05 will add the matching event-id waits.
    }

    __aicore__ inline void ComputeAivTile(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        (void)slice;
    }

    __aicore__ inline void ComputeAicTile(const TileParam &tile)
    {
        (void)tile;
        (void)cubeIndex_;
    }

    GlobalTensors tensors_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
    GM_ADDR workspace_;
    uint32_t cubeIndex_;
    uint32_t aivLocalId_;
};

class D128FastController {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        tensors_ = tensors;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Process()
    {
        const TokenRange tokenRange = MakeCoreTokenRange(tilingData_->totalTokens, GetCubeIndex(), GetCubeGroupNum());

        D128FastBasicBlock basicBlock;
        basicBlock.Init(tensors_, tilingData_, tensors_.workspace);
        uint32_t localTileId = 0U;
        for (uint32_t tokenOffset = tokenRange.begin; tokenOffset < tokenRange.end;) {
            const uint32_t tokenSize = MinU32(tilingData_->tokenTile, tokenRange.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }
            const TileParam tile = MakeTileParam(localTileId, tokenOffset, tokenSize, tilingData_);
            basicBlock.ComputeTile(tile);
            tokenOffset += tokenSize;
            ++localTileId;
        }
        basicBlock.End();
    }

private:
    GlobalTensors tensors_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_
