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

struct TileRange {
    uint32_t begin;
    uint32_t end;
    uint32_t stride;
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

__aicore__ inline TileRange MakeTileRange(uint32_t totalTileNum, uint32_t cubeIndex, uint32_t cubeGroupNum)
{
    TileRange range = {0U, 0U, 1U};
    if (cubeGroupNum == 0U || cubeIndex >= cubeGroupNum || cubeIndex >= totalTileNum) {
        return range;
    }

    range.begin = cubeIndex;
    range.end = totalTileNum;
    range.stride = cubeGroupNum;
    return range;
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

    __aicore__ inline void ComputeRange(const TileRange &range)
    {
        if (g_coreType == AIC) {
            ComputeAicRolling(range);
        } else {
            ComputeAivRolling(range);
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

    __aicore__ inline void ComputeAivRolling(const TileRange &range)
    {
        for (uint32_t tileId = range.begin; tileId < range.end; tileId += range.stride) {
            (void)tileId;
            (void)aivLocalId_;
            (void)tilingData_;
        }
    }

    __aicore__ inline void ComputeAicRolling(const TileRange &range)
    {
        for (uint32_t tileId = range.begin; tileId < range.end; tileId += range.stride) {
            (void)tileId;
            (void)cubeIndex_;
            (void)tilingData_;
        }
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
        const uint32_t totalTileNum = CeilDiv(tilingData_->totalTokens, tilingData_->tokenTile);
        const TileRange range = MakeTileRange(totalTileNum, GetCubeIndex(), GetCubeGroupNum());

        D128FastBasicBlock basicBlock;
        basicBlock.Init(tensors_, tilingData_, tensors_.workspace);
        basicBlock.ComputeRange(range);
        basicBlock.End();
    }

private:
    GlobalTensors tensors_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_
