/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_basic_block.h"

namespace QkvRmsNormRopeCacheWithKScale {

class D128FastController {
private:
    struct TokenRange {
        uint32_t begin;
        uint32_t end;
    };

    __aicore__ inline TokenRange MakeCoreTokenRange(
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t cubeIndex) const
    {
        TokenRange range = {0U, 0U};
        if (tilingData->coreGroupNum == 0U || tilingData->coreTokenTile == 0U ||
            cubeIndex >= tilingData->coreGroupNum || tilingData->totalTokens == 0U) {
            return range;
        }

        range.begin = cubeIndex * tilingData->coreTokenTile;
        if (range.begin >= tilingData->totalTokens) {
            range.begin = tilingData->totalTokens;
            range.end = tilingData->totalTokens;
            return range;
        }
        range.end = MinU32(tilingData->totalTokens, range.begin + tilingData->coreTokenTile);
        return range;
    }

    __aicore__ inline TileParam MakeTileParam(uint32_t tileId, uint32_t tokenOffset, uint32_t tokenSize,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData) const
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

public:
    __aicore__ inline void Process(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        const TokenRange tokenRange = MakeCoreTokenRange(tilingData, GetCubeIndex());

        D128FastBasicBlock basicBlock;
        basicBlock.Init(tensors, tilingData);
        uint32_t localTileId = 0U;
        for (uint32_t tokenOffset = tokenRange.begin; tokenOffset < tokenRange.end;) {
            const uint32_t tokenSize = MinU32(tilingData->tokenTile, tokenRange.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }
            const TileParam tile = MakeTileParam(localTileId, tokenOffset, tokenSize, tilingData);
            basicBlock.ComputeTile(tile);
            tokenOffset += tokenSize;
            ++localTileId;
        }
        basicBlock.End();
    }
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
