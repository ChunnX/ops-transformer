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

class QkvRmsNormRopeCacheWithKScaleController {
public:
    __aicore__ inline void Process(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
#if defined(__CCE_AICORE__)
        if ASCEND_IS_AIC {
            ProcessAic(tensors, tilingData);
        } else {
            ProcessAiv(tensors, tilingData);
        }
#endif
    }

private:
    __aicore__ inline TokenRange MakeCoreTokenRange(const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                                    uint32_t cubeIndex) const
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
        range.end = MinU64(tilingData->totalTokens, range.begin + tilingData->coreTokenTile);
        return range;
    }

    __aicore__ inline void ResetTileParam(TileParam &tile) const
    {
        tile.tileId = 0U;
        tile.tokenOffset = 0U;
        tile.tokenSize = 0U;
        tile.qkHeadOffset = 0U;
        tile.qkHeadSize = 0U;
        tile.vHeadOffset = 0U;
        tile.vHeadSize = 0U;
        tile.rowOffset = 0U;
        tile.rowSize = 0U;
    }

    __aicore__ inline void FillTileParam(TileParam &tile, const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                         uint64_t tileId, uint64_t tokenOffset, uint64_t tokenSize) const
    {
        ResetTileParam(tile);
        tile.tileId = tileId;
        tile.tokenOffset = tokenOffset;
        if (tile.tokenOffset >= tilingData->totalTokens || tokenSize == 0U) {
            return;
        }

        tile.tokenSize = MinU64(MinU64(tokenSize, tilingData->tokenTile), tilingData->totalTokens - tile.tokenOffset);
        tile.qkHeadOffset = 0U;
        const uint64_t qkHeadTile = tilingData->qHeadNum + tilingData->kvHeadNum;
        tile.qkHeadSize = qkHeadTile;
        tile.vHeadOffset = 0U;
        tile.vHeadSize = tilingData->kvHeadNum;
        tile.rowOffset = 0U;
        tile.rowSize = tile.tokenSize * qkHeadTile;
    }

    __aicore__ inline void ForEachTileAic(QkvRmsNormRopeCacheWithKScaleBasicBlock &basicBlock,
                                          const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                          const TokenRange &range) const
    {
        uint64_t localTileId = 0U;
        TileParam tile;
        ResetTileParam(tile);
        for (uint64_t tokenOffset = range.begin; tokenOffset < range.end;) {
            const uint64_t tokenSize = MinU64(tilingData->tokenTile, range.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }

            FillTileParam(tile, tilingData, localTileId, tokenOffset, tokenSize);
            basicBlock.ComputeTileAic(tile);
            tokenOffset += tokenSize;
            ++localTileId;
        }
    }

    __aicore__ inline uint64_t ForEachTileAiv(QkvRmsNormRopeCacheWithKScaleBasicBlock &basicBlock,
                                              const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                              const TokenRange &range,
                                              TileParam tileParam[QKV_RRC_DOUBLE_BUFFER_NUM]) const
    {
        uint64_t localTileId = 0U;
        for (uint64_t tokenOffset = range.begin; tokenOffset < range.end;) {
            const uint64_t tokenSize = MinU64(tilingData->tokenTile, range.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }

            const uint32_t processId = GetProcessId(localTileId);
            const uint32_t lastProcessId = GetLastProcessId(localTileId);
            FillTileParam(tileParam[processId], tilingData, localTileId, tokenOffset, tokenSize);
            basicBlock.ComputeTileAiv(tileParam[processId], tileParam[lastProcessId]);
            tokenOffset += tokenSize;
            ++localTileId;
        }
        return localTileId;
    }

    __aicore__ inline void ProcessAic(const GlobalTensors &tensors,
                                      const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        const uint32_t cubeIndex = GetAicCubeIndex();
        const TokenRange tokenRange = MakeCoreTokenRange(tilingData, cubeIndex);
        QkvRmsNormRopeCacheWithKScaleBasicBlock basicBlock;
        basicBlock.InitAic(tensors, tilingData);
        basicBlock.PrepareBeforeLoopAic();
        ForEachTileAic(basicBlock, tilingData, tokenRange);
        basicBlock.EndAic();
    }

    __aicore__ inline void ProcessAiv(const GlobalTensors &tensors,
                                      const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        const uint32_t cubeIndex = GetAivCubeIndex();
        const TokenRange tokenRange = MakeCoreTokenRange(tilingData, cubeIndex);
        QkvRmsNormRopeCacheWithKScaleBasicBlock basicBlock;
        TileParam tileParam[QKV_RRC_DOUBLE_BUFFER_NUM];
        InitTileParamBuffer(tileParam);
        basicBlock.InitAiv(tensors, tilingData, GetSubBlockIdx());
        const uint64_t localTileId = ForEachTileAiv(basicBlock, tilingData, tokenRange, tileParam);
        basicBlock.EndAiv(tileParam[GetLastProcessId(localTileId)]);
    }

    __aicore__ inline void InitTileParamBuffer(TileParam tileParam[QKV_RRC_DOUBLE_BUFFER_NUM]) const
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            ResetTileParam(tileParam[bufferId]);
        }
    }

    __aicore__ inline uint32_t GetProcessId(uint64_t tileId) const
    {
        return static_cast<uint32_t>(tileId & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
    }

    __aicore__ inline uint32_t GetLastProcessId(uint64_t tileId) const
    {
        return static_cast<uint32_t>((tileId + QKV_RRC_DOUBLE_BUFFER_NUM - 1U) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
    }
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
