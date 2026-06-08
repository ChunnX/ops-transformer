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

class QkvRmsNormRopeCacheWithKScaleBasicBlock {
public:
    __aicore__ inline void InitAic(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t cubeIndex)
    {
        aivTileCount_ = 0U;
        outBufferUseId_ = 0U;
        pendingQkOutHead_ = 0U;
        pendingQkOutCount_ = 0U;
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            pendingQkOutBufferIds_[i] = 0U;
            aivQkOutUseIds_[i] = 0U;
        }
        cube_.Init(cubeIndex, tilingData, tensors);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId)
    {
        aivTileCount_ = 0U;
        outBufferUseId_ = 0U;
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            aivQkOutUseIds_[i] = 0U;
        }
        vec_.Init(tilingData, aivLocalId, tensors);
        vec_.InitIntraCoreEvents();
        SetInitialAivMte3ToAicFixCConsumedCredits();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile)
    {
        ReservePreprocessOutUseId();
        const uint64_t qkOutUseId = ReserveQkOutUseId();
        WaitUntilQkOutBufferReusable(qkOutUseId);
        WaitAivMte3ToAicMte1AReady();
        cube_.ComputeTile(tile, qkOutUseId);
        SetAicMte1ToAivMte3AConsumed();
        SetAicFixToAivVCReady();
        PushPendingQkOutBuffer(qkOutUseId);
    }

    __aicore__ inline void PrepareBeforeLoopAic()
    {
        cube_.PrepareRotationBeforeLoop();
        SetInitialAicMte1ToAivMte3AConsumedCredits();
    }

    __aicore__ inline void ComputeTileAiv(TileParam &tile, const TileParam &lastTile)
    {
        const uint64_t preprocessOutUseId = ReservePreprocessOutUseId();
        aivQkOutUseIds_[MakeDataBufferId(tile.bufferId)] = ReserveQkOutUseId();
        WaitAicMte1ToAivMte3AConsumed();
        vec_.WaitVToMte2InputBufferReusable(tile);
        // Empty tail slices still reserve this credit so dual-dst FIXPIPE sees one address sequence.
        vec_.WaitMte3ToVQkOutBufferReusable(preprocessOutUseId);
        vec_.ComputeTile(tile, preprocessOutUseId);
        vec_.SetMte3ToVQkOutBufferReusable(preprocessOutUseId);
        SetAivMte3ToAicMte1AReady();
        vec_.PostprocessV(tile);
        vec_.SetVToMte2InputBufferReusable(tile);
        if (aivTileCount_ > 0U) {
            const uint64_t lastQkOutUseId = aivQkOutUseIds_[MakeDataBufferId(lastTile.bufferId)];
            vec_.WaitMte3ToVQkOutBufferReusable(lastQkOutUseId);
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile, lastQkOutUseId);
            vec_.SetMte3ToVQkOutBufferReusable(lastQkOutUseId);
            SetAivMte3ToAicFixCConsumed();
        }
        ++aivTileCount_;
    }

    __aicore__ inline void EndAic()
    {
        DrainPendingCConsumed();
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        if (aivTileCount_ > 0U) {
            const uint64_t lastQkOutUseId = aivQkOutUseIds_[MakeDataBufferId(lastTile.bufferId)];
            vec_.WaitMte3ToVQkOutBufferReusable(lastQkOutUseId);
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile, lastQkOutUseId);
            vec_.SetMte3ToVQkOutBufferReusable(lastQkOutUseId);
            SetAivMte3ToAicFixCConsumed();
            aivTileCount_ = 0U;
        }
        DrainAicMte1ToAivMte3AConsumedCredits();
        vec_.EndIntraCoreEvents();
    }

private:
    __aicore__ inline uint64_t ReservePreprocessOutUseId()
    {
        return outBufferUseId_++;
    }

    __aicore__ inline uint64_t ReserveQkOutUseId()
    {
        return outBufferUseId_++;
    }

    __aicore__ inline void SetInitialAicMte1ToAivMte3AConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            SetAicMte1ToAivMte3AConsumed();
        }
    }

    __aicore__ inline void SetInitialAivMte3ToAicFixCConsumedCredits()
    {
    }

    __aicore__ inline void DrainAicMte1ToAivMte3AConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            WaitAicMte1ToAivMte3AConsumed();
        }
    }

    __aicore__ inline bool IsQkOutBufferPending(uint32_t outBufferId) const
    {
        for (uint32_t i = 0U; i < pendingQkOutCount_; ++i) {
            const uint32_t index = (pendingQkOutHead_ + i) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U);
            if (pendingQkOutBufferIds_[index] == outBufferId) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void WaitAndPopOldestPendingQkOutBuffer()
    {
        if (pendingQkOutCount_ == 0U) {
            return;
        }
        WaitAivMte3ToAicFixCConsumed();
        pendingQkOutHead_ = (pendingQkOutHead_ + 1U) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U);
        --pendingQkOutCount_;
    }

    __aicore__ inline void WaitUntilQkOutBufferReusable(uint64_t outBufferUseId)
    {
        const uint32_t outBufferId = MakeOutputBufferId(outBufferUseId);
        while (IsQkOutBufferPending(outBufferId)) {
            WaitAndPopOldestPendingQkOutBuffer();
        }
    }

    __aicore__ inline void PushPendingQkOutBuffer(uint64_t outBufferUseId)
    {
        const uint32_t outBufferId = MakeOutputBufferId(outBufferUseId);
        const uint32_t index = (pendingQkOutHead_ + pendingQkOutCount_) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U);
        pendingQkOutBufferIds_[index] = outBufferId;
        ++pendingQkOutCount_;
    }

    __aicore__ inline void DrainPendingCConsumed()
    {
        while (pendingQkOutCount_ > 0U) {
            WaitAndPopOldestPendingQkOutBuffer();
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

    __aicore__ inline void SetAicMte1ToAivMte3AConsumed()
    {
        SetAicToAiv<PIPE_MTE1>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void WaitAicMte1ToAivMte3AConsumed()
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

    uint64_t aivTileCount_;
    uint64_t outBufferUseId_;
    uint32_t pendingQkOutHead_;
    uint32_t pendingQkOutCount_;
    uint32_t pendingQkOutBufferIds_[QKV_RRC_DOUBLE_BUFFER_NUM];
    uint64_t aivQkOutUseIds_[QKV_RRC_DOUBLE_BUFFER_NUM];
    QkvRmsNormRopeCacheWithKScaleCube cube_;
    QkvRmsNormRopeCacheWithKScaleVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
