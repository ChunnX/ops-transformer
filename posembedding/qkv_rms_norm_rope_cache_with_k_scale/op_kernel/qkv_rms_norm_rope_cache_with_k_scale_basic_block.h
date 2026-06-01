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
        SetInitialAivMte3ToAicFixCConsumedCredits();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile)
    {
        WaitAivMte3ToAicMte1AReady();
        WaitAivMte3ToAicFixCConsumed();
        cube_.ComputeTile(tile);
        SetAicFixToAivMte3AConsumed();
        SetAicFixToAivVCReady();
    }

    __aicore__ inline void PrepareBeforeLoopAic()
    {
        cube_.PrepareRotationBeforeLoop();
        SetInitialAicFixToAivMte3AConsumedCredits();
    }

    __aicore__ inline void ComputeTileAiv(const TileParam &tile, const TileParam &lastTile)
    {
        WaitAicFixToAivMte3AConsumed();
        vec_.ComputeTile(tile);
        SetAivMte3ToAicMte1AReady();
        vec_.PostprocessV(tile);
        if (aivTileCount_ > 0U) {
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile);
            SetAivMte3ToAicFixCConsumed();
        }
        ++aivTileCount_;
    }

    __aicore__ inline void EndAic()
    {
        DrainAivMte3ToAicFixCConsumedCredits();
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        if (aivTileCount_ > 0U) {
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile);
            SetAivMte3ToAicFixCConsumed();
            aivTileCount_ = 0U;
        }
        DrainAicFixToAivMte3AConsumedCredits();
        vec_.EndIntraCoreEvents();
    }

private:
    __aicore__ inline void SetInitialAicFixToAivMte3AConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            SetAicFixToAivMte3AConsumed();
        }
    }

    __aicore__ inline void SetInitialAivMte3ToAicFixCConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            SetAivMte3ToAicFixCConsumed();
        }
    }

    __aicore__ inline void DrainAicFixToAivMte3AConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            WaitAicFixToAivMte3AConsumed();
        }
    }

    __aicore__ inline void DrainAivMte3ToAicFixCConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            WaitAivMte3ToAicFixCConsumed();
        }
    }

    __aicore__ inline void SetAivMte3ToAicMte1AReady()
    {
        SetAivToAic<PIPE_MTE3>(SYNC_A_READY);
    }

    __aicore__ inline void WaitAivMte3ToAicMte1AReady()
    {
        WaitAivToAic<PIPE_MTE1>(SYNC_A_READY);
    }

    __aicore__ inline void SetAicFixToAivMte3AConsumed()
    {
        SetAicToAiv<PIPE_FIX>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void WaitAicFixToAivMte3AConsumed()
    {
        WaitAicToAiv<PIPE_MTE3>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void SetAicFixToAivVCReady()
    {
        SetAicToAiv<PIPE_FIX>(SYNC_C_READY);
    }

    __aicore__ inline void WaitAicFixToAivVCReady()
    {
        WaitAicToAiv<PIPE_V>(SYNC_C_READY);
    }

    __aicore__ inline void SetAivMte3ToAicFixCConsumed()
    {
        SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED);
    }

    __aicore__ inline void WaitAivMte3ToAicFixCConsumed()
    {
        WaitAivToAic<PIPE_FIX>(SYNC_C_CONSUMED);
    }

    uint32_t aivTileCount_;
    D128FastCube cube_;
    D128FastVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
