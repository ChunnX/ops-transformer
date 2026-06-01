/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_cube.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_vec.h"

namespace QkvRmsNormRopeCacheWithKScale {

class D128FastBasicBlock {
public:
    __aicore__ inline void InitAic(const GlobalTensors &tensors, uint32_t cubeIndex)
    {
        pendingCConsumedCount_ = 0U;
        aivTileCount_ = 0U;
        cube_.Init(cubeIndex, tensors);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId)
    {
        pendingCConsumedCount_ = 0U;
        aivTileCount_ = 0U;
        vec_.Init(tilingData, aivLocalId, tensors);
        vec_.InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile)
    {
        WaitReusableOutputBufferAic();
        WaitAReadyFromAiv();
        cube_.ComputeTile(tile);
        NotifyCReadyToAiv();
        ++pendingCConsumedCount_;
    }

    __aicore__ inline void PrepareBeforeLoopAic()
    {
        cube_.PrepareRotationBeforeLoop();
    }

    __aicore__ inline void ComputeTileAiv(const TileParam &tile, const TileParam &lastTile)
    {
        PreprocessTileAiv(tile);
        if (aivTileCount_ > 0U) {
            PostprocessTileAiv(lastTile);
        }
        ++aivTileCount_;
    }

    __aicore__ inline void EndAic()
    {
        DrainAic();
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        DrainAiv(lastTile);
        vec_.EndIntraCoreEvents();
    }

private:
    __aicore__ inline void WaitReusableOutputBufferAic()
    {
        if (pendingCConsumedCount_ < QKV_RRC_DOUBLE_BUFFER_NUM) {
            return;
        }
        WaitCConsumedFromAiv();
        --pendingCConsumedCount_;
    }

    __aicore__ inline void DrainAic()
    {
        while (pendingCConsumedCount_ > 0U) {
            WaitCConsumedFromAiv();
            --pendingCConsumedCount_;
        }
    }

    __aicore__ inline void DrainAiv(const TileParam &lastTile)
    {
        if (aivTileCount_ == 0U) {
            return;
        }
        PostprocessTileAiv(lastTile);
        aivTileCount_ = 0U;
    }

    __aicore__ inline void PreprocessTileAiv(const TileParam &tile)
    {
        vec_.ComputeTile(tile);
        NotifyAReadyToAic();
        vec_.PostprocessV(tile);
    }

    __aicore__ inline void PostprocessTileAiv(const TileParam &tile)
    {
        WaitCReadyFromAic();
        vec_.PostprocessQk(tile);
        NotifyCConsumedToAic();
    }

    __aicore__ inline void NotifyAReadyToAic()
    {
        SetAivToAic<PIPE_MTE3>(SYNC_A_READY);
    }

    __aicore__ inline void WaitAReadyFromAiv()
    {
        WaitAivToAic<PIPE_MTE1>(SYNC_A_READY);
    }

    __aicore__ inline void NotifyCReadyToAiv()
    {
        SetAicToAiv<PIPE_FIX>(SYNC_C_READY);
    }

    __aicore__ inline void WaitCReadyFromAic()
    {
        WaitAicToAiv<PIPE_V>(SYNC_C_READY);
    }

    __aicore__ inline void NotifyCConsumedToAic()
    {
        SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED);
    }

    __aicore__ inline void WaitCConsumedFromAiv()
    {
        WaitAivToAic<PIPE_FIX>(SYNC_C_CONSUMED);
    }

    uint32_t pendingCConsumedCount_;
    uint32_t aivTileCount_;
    D128FastCube cube_;
    D128FastVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
