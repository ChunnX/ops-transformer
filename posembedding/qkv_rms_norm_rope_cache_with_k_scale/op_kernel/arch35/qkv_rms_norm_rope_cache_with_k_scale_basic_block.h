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
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        tileCount_ = 0U;
        aRotL1BufferUseId_ = 0U;
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        cube_.Init(tilingData, tensors);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId)
    {
        tileCount_ = 0U;
        aRotL1BufferUseId_ = 0U;
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        vec_.Init(tilingData, aivLocalId, tensors);
        vec_.InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile)
    {
        const uint64_t aRotL1BufferUseId = ReserveARotL1BufferUseId();
        const uint32_t aRotL1BufferId =
            static_cast<uint32_t>(aRotL1BufferUseId & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        if (tileCount_ > 0U) {
            WaitAivMte3ToAicFixCConsumed();
        }
        WaitAivMte3ToAicMte1AReady();
        cube_.ComputeTile(tile, aRotL1Nz, QKV_RRC_OUTPUT_QK_BUFFER_ID);
        SetAicMte1ToAivMte3AConsumed();
        SetAicFixToAivVCReady();
        ++tileCount_;
    }

    __aicore__ inline void PrepareBeforeLoopAic()
    {
        cube_.PrepareRotationBeforeLoop();
        SetInitialAicMte1ToAivMte3AConsumedCredits();
    }

    __aicore__ inline void ComputeTileAiv(TileParam &tile, const TileParam &lastTile)
    {
        const uint64_t aRotL1BufferUseId = ReserveARotL1BufferUseId();
        const uint32_t aRotL1BufferId =
            static_cast<uint32_t>(aRotL1BufferUseId & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        WaitAicMte1ToAivMte3AConsumed();
        vec_.WaitVToMte2InputBufferReusable();
        vec_.WaitMte3ToVQkOutBufferReusable(QKV_RRC_OUTPUT_PREPROCESS_BUFFER_ID);
        vec_.ComputeTile(tile, aRotL1Nz, QKV_RRC_OUTPUT_PREPROCESS_BUFFER_ID);
        vec_.SetMte3ToVQkOutBufferReusable(QKV_RRC_OUTPUT_PREPROCESS_BUFFER_ID);
        SetAivMte3ToAicMte1AReady();
        vec_.PostprocessV(tile);
        vec_.SetVToMte2InputBufferReusable();
        if (tileCount_ > 0U) {
            vec_.WaitMte3ToVQkOutBufferReusable(QKV_RRC_OUTPUT_QK_BUFFER_ID);
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile, QKV_RRC_OUTPUT_QK_BUFFER_ID);
            vec_.SetMte3ToVQkOutBufferReusable(QKV_RRC_OUTPUT_QK_BUFFER_ID);
            SetAivMte3ToAicFixCConsumed();
        }
        ++tileCount_;
    }

    __aicore__ inline void EndAic()
    {
        if (tileCount_ > 0U) {
            WaitAivMte3ToAicFixCConsumed();
            tileCount_ = 0U;
        }
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        if (tileCount_ > 0U) {
            vec_.WaitMte3ToVQkOutBufferReusable(QKV_RRC_OUTPUT_QK_BUFFER_ID);
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile, QKV_RRC_OUTPUT_QK_BUFFER_ID);
            vec_.SetMte3ToVQkOutBufferReusable(QKV_RRC_OUTPUT_QK_BUFFER_ID);
            SetAivMte3ToAicFixCConsumed();
            tileCount_ = 0U;
        }
        DrainAicMte1ToAivMte3AConsumedCredits();
        vec_.EndIntraCoreEvents();
    }

private:
    __aicore__ inline uint64_t ReserveARotL1BufferUseId()
    {
        return aRotL1BufferUseId_++;
    }

    __aicore__ inline void SetInitialAicMte1ToAivMte3AConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            SetAicMte1ToAivMte3AConsumed();
        }
    }

    __aicore__ inline void DrainAicMte1ToAivMte3AConsumedCredits()
    {
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            WaitAicMte1ToAivMte3AConsumed();
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

    uint64_t tileCount_;
    uint64_t aRotL1BufferUseId_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    QkvRmsNormRopeCacheWithKScaleCube cube_;
    QkvRmsNormRopeCacheWithKScaleVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
