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
        aivTileCount_ = 0U;
        cube_.Init(cubeIndex, tensors);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId)
    {
        aivTileCount_ = 0U;
        vec_.Init(tilingData, aivLocalId, tensors);
        vec_.InitIntraCoreEvents();
        InitReusableOutputBufferCreditsAiv();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile)
    {
        WaitInputAReadyAic();
        WaitReusableOutputBufferAic();
        cube_.ComputeTile(tile);
        NotifyAConsumedToAiv();
        NotifyCReadyToAiv();
    }

    __aicore__ inline void PrepareBeforeLoopAic()
    {
        cube_.PrepareRotationBeforeLoop();
        InitReusableInputBufferCreditsAic();
    }

    __aicore__ inline void ComputeTileAiv(const TileParam &tile, const TileParam &lastTile)
    {
        WaitReusableInputBufferAiv();
        vec_.ComputeTile(tile);
        NotifyAReadyToAic();
        vec_.PostprocessV(tile);
        if (aivTileCount_ > 0U) {
            WaitCReadyFromAic();
            vec_.PostprocessQk(lastTile);
            NotifyCConsumedToAic();
        }
        ++aivTileCount_;
    }

    __aicore__ inline void EndAic()
    {
        DrainReusableOutputBuffersAic();
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        if (aivTileCount_ > 0U) {
            WaitCReadyFromAic();
            vec_.PostprocessQk(lastTile);
            NotifyCConsumedToAic();
            aivTileCount_ = 0U;
        }
        DrainReusableInputBuffersAiv();
        vec_.EndIntraCoreEvents();
    }

private:
    __aicore__ inline void InitReusableInputBufferCreditsAic()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            NotifyAConsumedToAiv();
        }
    }

    __aicore__ inline void InitReusableOutputBufferCreditsAiv()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            NotifyCConsumedToAic();
        }
    }

    __aicore__ inline void WaitInputAReadyAic()
    {
        WaitAReadyFromAiv();
    }

    __aicore__ inline void WaitReusableInputBufferAiv()
    {
        WaitAicToAiv<PIPE_MTE3>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void WaitReusableOutputBufferAic()
    {
        WaitCConsumedFromAiv();
    }

    __aicore__ inline void DrainReusableInputBuffersAiv()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            WaitReusableInputBufferAiv();
        }
    }

    __aicore__ inline void DrainReusableOutputBuffersAic()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            WaitReusableOutputBufferAic();
        }
    }

    __aicore__ inline void NotifyAReadyToAic()
    {
        SetAivToAic<PIPE_MTE3>(SYNC_A_READY);
    }

    __aicore__ inline void WaitAReadyFromAiv()
    {
        WaitAivToAic<PIPE_MTE1>(SYNC_A_READY);
    }

    __aicore__ inline void NotifyAConsumedToAiv()
    {
        SetAicToAiv<PIPE_FIX>(SYNC_A_CONSUMED);
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

    uint32_t aivTileCount_;
    D128FastCube cube_;
    D128FastVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
