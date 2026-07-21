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

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE, uint32_t K_QUANT_MODE,
          uint32_t Q_QUANT_MODE, bool MROPE_COMPACT_PIPELINE>
class QkvRmsNormRopeCacheWithKScaleBasicBlock {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch, uint64_t qHeadNum,
                                uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize, uint64_t tokenTile,
                                uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead, uint64_t kvCacheStrideToken,
                                uint64_t kScaleCacheStrideBlock, uint64_t kScaleCacheStrideHead,
                                uint64_t kScaleCacheStrideToken, float epsilon, uint64_t mropeSectionH,
                                uint64_t mropeSectionW)
    {
        if ASCEND_IS_AIC {
            InitAic(tensors, qHeadNum, kvHeadNum);
        } else {
            InitAiv(tensors, totalTokens, batch, qHeadNum, kvHeadNum, headDim, blockSize, tokenTile, kvCacheStrideBlock,
                    kvCacheStrideHead, kvCacheStrideToken, kScaleCacheStrideBlock, kScaleCacheStrideHead,
                    kScaleCacheStrideToken, epsilon, mropeSectionH, mropeSectionW);
        }
    }

    __aicore__ inline void PrepareBeforeLoop(uint64_t coreTokenBegin, uint64_t coreTokenEnd)
    {
        if ASCEND_IS_AIC {
            cube_.PrepareRotationBeforeLoop();
            for (uint32_t i = 0U; i < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++i) {
                SetAicMte1ToAivMte3AConsumed();
            }
        } else {
            vec_.PrepareBeforeLoop(coreTokenBegin, coreTokenEnd);
            for (uint32_t i = 0U; i < MROPE_K_CONSUMED_CREDITS; ++i) {
                if constexpr (DIRECT_MROPE_Q_OUT) {
                    if constexpr (MROPE_COMPACT_PIPELINE) {
                        SetAivVToAicFixMropeKConsumed();
                    } else {
                        SetAivMte3ToAicFixMropeKConsumed();
                    }
                } else {
                    SetAivMte3ToAicFixCConsumed();
                }
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
        hasPendingKTile_ = false;
        pendingKQuantBufferId_ = 0U;
        aRotL1Pool_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_A_ROT_L1_POOL_OFFSET,
                                              QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS);
        if constexpr (MROPE_COMPACT_PIPELINE) {
            kRotationDbPoolUb_ = LocalTensor<float>(
                TPosition::LCM, QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_OFFSET,
                QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_FLOAT_ELEMENTS);
        } else {
            outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        }
        cube_.Init(tensors, qHeadNum, kvHeadNum);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch,
                                   uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize,
                                   uint64_t tokenTile, uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead,
                                   uint64_t kvCacheStrideToken, uint64_t kScaleCacheStrideBlock,
                                   uint64_t kScaleCacheStrideHead, uint64_t kScaleCacheStrideToken, float epsilon,
                                   uint64_t mropeSectionH, uint64_t mropeSectionW)
    {
        tileCount_ = 0U;
        outputBufferUseId_ = 0U;
        hasPendingKTile_ = false;
        pendingKQuantBufferId_ = 0U;
        aRotL1Pool_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_A_ROT_L1_POOL_OFFSET,
                                              QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS);
        if constexpr (MROPE_COMPACT_PIPELINE) {
            qkPreprocessDbPoolUb_ = LocalTensor<float>(
                TPosition::LCM, QKV_K_SCALE_MROPE_COMPACT_QK_PREPROCESS_DB_POOL_OFFSET,
                QKV_K_SCALE_MROPE_COMPACT_QK_PREPROCESS_DB_POOL_FLOAT_ELEMENTS);
            kRotationDbPoolUb_ = LocalTensor<float>(
                TPosition::LCM, QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_OFFSET,
                QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_FLOAT_ELEMENTS);
        } else {
            outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        }
        vec_.Init(tensors, totalTokens, batch, qHeadNum, kvHeadNum, headDim, blockSize, tokenTile, kvCacheStrideBlock,
                  kvCacheStrideHead, kvCacheStrideToken, kScaleCacheStrideBlock, kScaleCacheStrideHead,
                  kScaleCacheStrideToken, epsilon, mropeSectionH, mropeSectionW);
        vec_.InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile, bool isLastTile)
    {
        const uint32_t aRotL1BufferId = static_cast<uint32_t>(tileCount_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        uint32_t outputBufferId = 0U;
        if constexpr (MROPE_COMPACT_PIPELINE) {
            outputBufferId = static_cast<uint32_t>(tileCount_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
            WaitAivVToAicFixMropeKConsumed();
        } else {
            outputBufferUseId_ += isLastTile ? 1U : QKV_K_SCALE_AIV_OUTPUT_BUFFER_USES_PER_TILE;
            outputBufferId = static_cast<uint32_t>(outputBufferUseId_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
            if constexpr (DIRECT_MROPE_Q_OUT) {
                WaitAivMte3ToAicFixMropeKConsumed();
            } else {
                WaitAivMte3ToAicFixCConsumed();
            }
        }
        WaitAivMte3ToAicMte1AReady();
        if constexpr (MROPE_COMPACT_PIPELINE) {
            cube_.ComputeTile(tile, aRotL1Nz,
                              kRotationDbPoolUb_[outputBufferId *
                                                  QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS]);
        } else {
            cube_.ComputeTile(tile, aRotL1Nz,
                              outputDbPoolUb_[outputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS]);
        }
        SetAicMte1ToAivMte3AConsumed();
        if constexpr (DIRECT_MROPE_Q_OUT) {
            // K is already in the shared UB. Let VEC consume it before the
            // independent Q L0C->GM Fixpipe is submitted.
            SetAicFixToAivMropeKReady();
            cube_.FixpipeQToGm(tile);
        } else {
            SetAicFixToAivVCReady();
        }
        ++tileCount_;
    }

    __aicore__ inline void ComputeTileAiv(TileParam &tile, const TileParam &lastTile, bool isLastTile)
    {
        const uint32_t aRotL1BufferId = static_cast<uint32_t>(tileCount_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        if constexpr (MROPE_COMPACT_PIPELINE) {
            const uint32_t preprocessBufferId =
                static_cast<uint32_t>(tileCount_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));

            // Keep the current A path ahead of K postprocessing. Otherwise
            // StageMropeK(i - 1) delays A_READY(i) and stalls both AIC and AIV.
            vec_.ComputeTile(
                tile, aRotL1Nz,
                qkPreprocessDbPoolUb_[preprocessBufferId *
                                      QKV_K_SCALE_MROPE_COMPACT_QK_PREPROCESS_ONE_BUFFER_FLOAT_ELEMENTS],
                preprocessBufferId);
            SetAivMte3ToAicMte1AReady();

            // Retire K(i - 2) before saving K(i - 1) because pending metadata
            // has a single slot. K scatter follows A_READY and cannot delay
            // the current AIC tile.
            if (hasPendingKTile_) {
                vec_.ScatterMropeK(pendingKTile_, pendingKQuantBufferId_);
                hasPendingKTile_ = false;
            }
            if (tileCount_ > 0U) {
                const uint32_t kBufferId =
                    static_cast<uint32_t>((tileCount_ - 1U) & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
                WaitAicFixToAivMropeKReady();
                vec_.StageMropeK(
                    lastTile,
                    kRotationDbPoolUb_[kBufferId * QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS],
                    kBufferId);
                SetAivVToAicFixMropeKConsumed();
                if (lastTile.aivTokenSize != 0U) {
                    SavePendingKTile(lastTile, kBufferId);
                }
            }
        } else {
            const uint32_t outputBufferId = static_cast<uint32_t>(outputBufferUseId_ &
                                                                  (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
            WaitAicMte1ToAivMte3AConsumed();
            vec_.ComputeTile(
                tile, aRotL1Nz,
                outputDbPoolUb_[outputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS], outputBufferId);
            ++outputBufferUseId_;
            if (isLastTile && tileCount_ > 0U) {
                if constexpr (DIRECT_MROPE_Q_OUT) {
                    SetAivMte3ToAicFixMropeKConsumed();
                } else {
                    SetAivMte3ToAicFixCConsumed();
                }
            }
            SetAivMte3ToAicMte1AReady();
            if (tileCount_ > 0U) {
                const uint32_t lastOutputBufferId = static_cast<uint32_t>(outputBufferUseId_ &
                                                                          (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
                if constexpr (DIRECT_MROPE_Q_OUT) {
                    WaitAicFixToAivMropeKReady();
                } else {
                    WaitAicFixToAivVCReady();
                }
                vec_.PostprocessQk(
                    lastTile,
                    outputDbPoolUb_[lastOutputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                    lastOutputBufferId);
                if (!isLastTile) {
                    if constexpr (DIRECT_MROPE_Q_OUT) {
                        SetAivMte3ToAicFixMropeKConsumed();
                    } else {
                        SetAivMte3ToAicFixCConsumed();
                    }
                }
                ++outputBufferUseId_;
            }
        }
        ++tileCount_;
    }

    __aicore__ inline void SavePendingKTile(const TileParam &tile, uint32_t bufferId)
    {
        pendingKTile_.aivTokenSize = tile.aivTokenSize;
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            pendingKTile_.cacheBaseOffset[tokenIdx] = tile.cacheBaseOffset[tokenIdx];
            pendingKTile_.scaleCacheBaseOffset[tokenIdx] = tile.scaleCacheBaseOffset[tokenIdx];
        }
        pendingKQuantBufferId_ = bufferId;
        hasPendingKTile_ = true;
    }

    __aicore__ inline void EndAic()
    {
        for (uint32_t i = 0U; i < MROPE_K_CONSUMED_CREDITS; ++i) {
            if constexpr (DIRECT_MROPE_Q_OUT) {
                if constexpr (MROPE_COMPACT_PIPELINE) {
                    WaitAivVToAicFixMropeKConsumed();
                } else {
                    WaitAivMte3ToAicFixMropeKConsumed();
                }
            } else {
                WaitAivMte3ToAicFixCConsumed();
            }
        }
        tileCount_ = 0U;
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile)
    {
        if constexpr (MROPE_COMPACT_PIPELINE) {
            if (hasPendingKTile_) {
                vec_.ScatterMropeK(pendingKTile_, pendingKQuantBufferId_);
                hasPendingKTile_ = false;
            }
            if (tileCount_ > 0U) {
                const uint32_t kBufferId =
                    static_cast<uint32_t>((tileCount_ - 1U) & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
                WaitAicFixToAivMropeKReady();
                vec_.StageMropeK(
                    lastTile,
                    kRotationDbPoolUb_[kBufferId * QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS],
                    kBufferId);
                SetAivVToAicFixMropeKConsumed();
                if (lastTile.aivTokenSize != 0U) {
                    vec_.ScatterMropeK(lastTile, kBufferId);
                }
                tileCount_ = 0U;
            }
        } else if (tileCount_ > 0U) {
            const uint32_t lastOutputBufferId = static_cast<uint32_t>(outputBufferUseId_ &
                                                                      (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
            if constexpr (DIRECT_MROPE_Q_OUT) {
                WaitAicFixToAivMropeKReady();
            } else {
                WaitAicFixToAivVCReady();
            }
            vec_.PostprocessQk(lastTile,
                               outputDbPoolUb_[lastOutputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                               lastOutputBufferId);
            if constexpr (DIRECT_MROPE_Q_OUT) {
                SetAivMte3ToAicFixMropeKConsumed();
            } else {
                SetAivMte3ToAicFixCConsumed();
            }
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

    __aicore__ inline void SetAicFixToAivMropeKReady()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_READY +
                                                                                QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_READY);
    }

    __aicore__ inline void WaitAicFixToAivVCReady()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_C_READY);
    }

    __aicore__ inline void WaitAicFixToAivMropeKReady()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_MROPE_K_READY);
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

    __aicore__ inline void SetAivMte3ToAicFixMropeKConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_MROPE_K_CONSUMED);
    }

    __aicore__ inline void SetAivVToAicFixMropeKConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_MROPE_K_CONSUMED);
    }

    __aicore__ inline void WaitAivMte3ToAicFixMropeKConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_CONSUMED +
                                                                                 QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_CONSUMED);
    }

    __aicore__ inline void WaitAivVToAicFixMropeKConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_CONSUMED +
                                                                                 QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_CONSUMED);
    }

    static constexpr bool DIRECT_MROPE_Q_OUT =
        (QKV_LAYOUT == QKV_K_SCALE_LAYOUT_TND && ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE &&
         Q_OUT_LAYOUT == QKV_K_SCALE_LAYOUT_TND &&
         K_QUANT_MODE == QKV_K_SCALE_K_QUANT_MODE_INT8 &&
         Q_QUANT_MODE == QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT);
    static_assert(!MROPE_COMPACT_PIPELINE || DIRECT_MROPE_Q_OUT,
                  "compact M-RoPE pipeline requires the direct-Q scene");

    static constexpr uint32_t QKV_K_SCALE_AIV_OUTPUT_BUFFER_USES_PER_TILE = 2U;
    static constexpr uint32_t QKV_K_SCALE_C_OUTPUT_BUFFER_CREDITS = 1U;
    static constexpr uint32_t MROPE_K_CONSUMED_CREDITS =
        (DIRECT_MROPE_Q_OUT && MROPE_COMPACT_PIPELINE) ? QKV_K_SCALE_DOUBLE_BUFFER_NUM
                                                       : QKV_K_SCALE_C_OUTPUT_BUFFER_CREDITS;

    uint64_t tileCount_;
    uint64_t outputBufferUseId_;
    bool hasPendingKTile_;
    uint32_t pendingKQuantBufferId_;
    TileParam pendingKTile_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<float> outputDbPoolUb_;
    LocalTensor<float> qkPreprocessDbPoolUb_;
    LocalTensor<float> kRotationDbPoolUb_;
    QkvRmsNormRopeCacheWithKScaleCube<DIRECT_MROPE_Q_OUT, MROPE_COMPACT_PIPELINE> cube_;
    QkvRmsNormRopeCacheWithKScaleVec<QKV_LAYOUT, Q_OUT_LAYOUT, ROPE_MODE, K_QUANT_MODE, Q_QUANT_MODE,
                                    MROPE_COMPACT_PIPELINE>
        vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
