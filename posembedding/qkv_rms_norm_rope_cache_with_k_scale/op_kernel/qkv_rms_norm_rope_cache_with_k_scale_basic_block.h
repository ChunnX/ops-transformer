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
        isAiv_ = false;
        pendingCConsumedCount_ = 0U;
        hasPendingQkPostprocess_ = false;
        pendingQkTile_ = EmptyTileParam();
        cube_.Init(cubeIndex, tensors);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId)
    {
        isAiv_ = true;
        pendingCConsumedCount_ = 0U;
        hasPendingQkPostprocess_ = false;
        pendingQkTile_ = EmptyTileParam();
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

    __aicore__ inline void ComputeTileAiv(const TileParam &tile)
    {
        PreprocessTileAiv(tile);
        if (hasPendingQkPostprocess_) {
            PostprocessTileAiv(pendingQkTile_);
        }
        pendingQkTile_ = tile;
        hasPendingQkPostprocess_ = true;
    }

    __aicore__ inline void End()
    {
        if (isAiv_) {
            DrainAiv();
            vec_.EndIntraCoreEvents();
        } else {
            DrainAic();
            cube_.EndIntraCoreEvents();
        }
    }

private:
    __aicore__ inline TileParam EmptyTileParam() const
    {
        TileParam tile = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
        return tile;
    }

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

    __aicore__ inline void DrainAiv()
    {
        if (!hasPendingQkPostprocess_) {
            return;
        }
        PostprocessTileAiv(pendingQkTile_);
        pendingQkTile_ = EmptyTileParam();
        hasPendingQkPostprocess_ = false;
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

    bool isAiv_;
    uint32_t pendingCConsumedCount_;
    bool hasPendingQkPostprocess_;
    TileParam pendingQkTile_;
    D128FastCube cube_;
    D128FastVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
