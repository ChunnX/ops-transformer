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

constexpr uint8_t QKV_K_SCALE_CROSS_CORE_SYNC_MODE = 4U;
constexpr uint64_t QKV_K_SCALE_AIV1_FLAG_OFFSET = 16U;
constexpr uint64_t SYNC_A_READY = 0U;
constexpr uint64_t SYNC_C_READY = 1U;
constexpr uint64_t SYNC_C_CONSUMED = 2U;
constexpr uint64_t SYNC_A_CONSUMED = 3U;

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT>
class QkvRmsNormRopeCacheWithKScaleBasicBlock {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch, uint64_t qHeadNum,
                                uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize, uint64_t tokenTile,
                                uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead, uint64_t kvCacheStrideToken,
                                uint64_t kScaleCacheStrideBlock, uint64_t kScaleCacheStrideHead,
                                uint64_t kScaleCacheStrideToken, float epsilon)
    {
        if ASCEND_IS_AIC {
            InitAic(tensors, qHeadNum, kvHeadNum);
        } else {
            InitAiv(tensors, totalTokens, batch, qHeadNum, kvHeadNum, headDim, blockSize, tokenTile, kvCacheStrideBlock,
                    kvCacheStrideHead, kvCacheStrideToken, kScaleCacheStrideBlock, kScaleCacheStrideHead,
                    kScaleCacheStrideToken, epsilon);
        }
    }

    __aicore__ inline void PrepareBeforeLoop()
    {
        if ASCEND_IS_AIC {
            cube_.PrepareRotationBeforeLoop();
            for (uint32_t i = 0U; i < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++i) {
                SetAicMte1ToAivMte3AConsumed();
            }
        } else {
            vec_.PrepareBeforeLoop();
            for (uint32_t i = 0U; i < QKV_K_SCALE_C_OUTPUT_BUFFER_CREDITS; ++i) {
                SetAivMte3ToAicFixCConsumed();
            }
        }
    }

    __aicore__ inline void ComputeTile(TileParam &tile, const TileParam &lastTile, bool isLastTile)
    {
        if ASCEND_IS_AIC {
            ComputeTileAic(tile, isLastTile);
        } else {
            ComputeTileAiv(tile, lastTile, isLastTile);
        }
    }

    __aicore__ inline void End(const TileParam &lastTile)
    {
        if ASCEND_IS_AIC {
            EndAic();
        } else {
            EndAiv(lastTile);
        }
    }

private:
    __aicore__ inline void InitAic(const GlobalTensors &tensors, uint64_t qHeadNum, uint64_t kvHeadNum)
    {
        tileCount_ = 0U;
        outputBufferUseId_ = 0U;
        aRotL1Pool_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_A_ROT_L1_POOL_OFFSET,
                                              QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS);
        outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET,
                                             QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        cube_.Init(tensors, qHeadNum, kvHeadNum);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch,
                                   uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize,
                                   uint64_t tokenTile, uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead,
                                   uint64_t kvCacheStrideToken, uint64_t kScaleCacheStrideBlock,
                                   uint64_t kScaleCacheStrideHead, uint64_t kScaleCacheStrideToken, float epsilon)
    {
        tileCount_ = 0U;
        outputBufferUseId_ = 0U;
        aRotL1Pool_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_A_ROT_L1_POOL_OFFSET,
                                              QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS);
        outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET,
                                             QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        vec_.Init(tensors, totalTokens, batch, qHeadNum, kvHeadNum, headDim, blockSize, tokenTile, kvCacheStrideBlock,
                  kvCacheStrideHead, kvCacheStrideToken, kScaleCacheStrideBlock, kScaleCacheStrideHead,
                  kScaleCacheStrideToken, epsilon);
        vec_.InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile, bool isLastTile)
    {
        const uint32_t aRotL1BufferId = static_cast<uint32_t>(tileCount_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        outputBufferUseId_ += isLastTile ? 1U : QKV_K_SCALE_AIV_OUTPUT_BUFFER_USES_PER_TILE;
        const uint32_t outputBufferId =
            static_cast<uint32_t>(outputBufferUseId_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        WaitAivMte3ToAicFixCConsumed();
        WaitAivMte3ToAicMte1AReady();
        cube_.ComputeTile(tile, aRotL1Nz,
                          outputDbPoolUb_[outputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS]);
        SetAicMte1ToAivMte3AConsumed();
        SetAicFixToAivVCReady();
        ++tileCount_;
    }

    __aicore__ inline void ComputeTileAiv(TileParam &tile, const TileParam &lastTile, bool isLastTile)
    {
        const uint32_t aRotL1BufferId = static_cast<uint32_t>(tileCount_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        const uint32_t outputBufferId = static_cast<uint32_t>(outputBufferUseId_ &
                                                              (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        WaitAicMte1ToAivMte3AConsumed();
        vec_.ComputeTile(tile, aRotL1Nz, outputDbPoolUb_[outputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                         outputBufferId);
        ++outputBufferUseId_;
        if (isLastTile && tileCount_ > 0U) {
            SetAivMte3ToAicFixCConsumed();
        }
        SetAivMte3ToAicMte1AReady();
        if (tileCount_ > 0U) {
            const uint32_t lastOutputBufferId = static_cast<uint32_t>(outputBufferUseId_ &
                                                                      (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile,
                               outputDbPoolUb_[lastOutputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                               lastOutputBufferId);
            if (!isLastTile) {
                SetAivMte3ToAicFixCConsumed();
            }
            ++outputBufferUseId_;
        }
        ++tileCount_;
    }

    __aicore__ inline void EndAic()
    {
        for (uint32_t i = 0U; i < QKV_K_SCALE_C_OUTPUT_BUFFER_CREDITS; ++i) {
            WaitAivMte3ToAicFixCConsumed();
        }
        tileCount_ = 0U;
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        if (tileCount_ > 0U) {
            const uint32_t lastOutputBufferId = static_cast<uint32_t>(outputBufferUseId_ &
                                                                      (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
            WaitAicFixToAivVCReady();
            vec_.PostprocessQk(lastTile,
                               outputDbPoolUb_[lastOutputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                               lastOutputBufferId);
            SetAivMte3ToAicFixCConsumed();
            ++outputBufferUseId_;
            tileCount_ = 0U;
        }
        for (uint32_t i = 0U; i < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++i) {
            WaitAicMte1ToAivMte3AConsumed();
        }
        vec_.EndIntraCoreEvents();
    }

    __aicore__ inline void SetAivMte3ToAicMte1AReady()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_A_READY);
    }

    __aicore__ inline void WaitAivMte3ToAicMte1AReady()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_READY +
                                                                                QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_READY);
    }

    __aicore__ inline void SetAicMte1ToAivMte3AConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_CONSUMED +
                                                                               QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void WaitAicMte1ToAivMte3AConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void SetAicFixToAivVCReady()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_C_READY +
                                                                              QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_C_READY);
    }

    __aicore__ inline void WaitAicFixToAivVCReady()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_C_READY);
    }

    __aicore__ inline void SetAivMte3ToAicFixCConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_C_CONSUMED);
    }

    __aicore__ inline void WaitAivMte3ToAicFixCConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_C_CONSUMED +
                                                                               QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_C_CONSUMED);
    }

    static constexpr uint32_t QKV_K_SCALE_AIV_OUTPUT_BUFFER_USES_PER_TILE = 2U;
    static constexpr uint32_t QKV_K_SCALE_C_OUTPUT_BUFFER_CREDITS = 1U;

    uint64_t tileCount_;
    uint64_t outputBufferUseId_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<float> outputDbPoolUb_;
    QkvRmsNormRopeCacheWithKScaleCube cube_;
    QkvRmsNormRopeCacheWithKScaleVec<QKV_LAYOUT, Q_OUT_LAYOUT> vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
