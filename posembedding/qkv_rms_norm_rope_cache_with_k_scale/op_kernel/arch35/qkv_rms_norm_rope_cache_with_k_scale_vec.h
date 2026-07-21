/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_vf.h"

namespace QkvRmsNormRopeCacheWithKScale {

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE, uint32_t K_QUANT_MODE,
          uint32_t Q_QUANT_MODE, bool MROPE_COMPACT_PIPELINE>
class QkvRmsNormRopeCacheWithKScaleVec {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch, uint64_t qHeadNum,
                                uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize, uint64_t tokenTile,
                                uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead, uint64_t kvCacheStrideToken,
                                uint64_t kScaleCacheStrideBlock, uint64_t kScaleCacheStrideHead,
                                uint64_t kScaleCacheStrideToken, float epsilon, uint64_t mropeSectionH,
                                uint64_t mropeSectionW)
    {
        totalTokens_ = totalTokens;
        batch_ = batch;
        qHeadNum_ = qHeadNum;
        kvHeadNum_ = kvHeadNum;
        headDim_ = headDim;
        blockSize_ = blockSize;
        kvCacheStrideBlock_ = kvCacheStrideBlock;
        kvCacheStrideHead_ = kvCacheStrideHead;
        kvCacheStrideToken_ = kvCacheStrideToken;
        kScaleCacheStrideBlock_ = kScaleCacheStrideBlock;
        kScaleCacheStrideHead_ = kScaleCacheStrideHead;
        kScaleCacheStrideToken_ = kScaleCacheStrideToken;
        epsilon_ = epsilon;
        mropeSectionH_ = mropeSectionH;
        mropeSectionW_ = mropeSectionW;
        inputBufferUseId_ = 0U;
        vOutBufferUseId_ = 0U;
        cosSinBatchIdx_ = 0U;
        metadataWindowBegin_ = 0U;
        metadataWindowEnd_ = 0U;
        metadataRangeEnd_ = 0U;
        const uint64_t tokenCapacity = CeilDiv(tokenTile, QKV_K_SCALE_MIX_AIV_PER_AIC);
        const uint64_t qPreprocessRows = tokenCapacity * qHeadNum_;
        const uint64_t kPreprocessRows = tokenCapacity * kvHeadNum_;
        qPreprocessRowStride_ =
            static_cast<uint32_t>(AlignUp(qPreprocessRows - 1U, QKV_K_SCALE_QK_PREPROCESS_UB_NZ_STRIDE_ALIGN) + 1U);
        kPreprocessRowStride_ =
            static_cast<uint32_t>(AlignUp(kPreprocessRows - 1U, QKV_K_SCALE_QK_PREPROCESS_UB_NZ_STRIDE_ALIGN) + 1U);
        qPreprocessElements_ = static_cast<uint32_t>(
            (((QKV_K_SCALE_HEAD_DIM_D128 / QKV_K_SCALE_NZ_C0) - 1U) * qPreprocessRowStride_ + qPreprocessRows) *
            QKV_K_SCALE_NZ_C0);
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        constexpr uint32_t inputBufferNum =
            OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_INPUT_BUFFER_NUM : QKV_K_SCALE_DOUBLE_BUFFER_NUM;
        for (uint32_t bufferId = 0U; bufferId < inputBufferNum; ++bufferId) {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
        }
        for (uint32_t bufferId = 0U; bufferId < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++bufferId) {
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
            if constexpr (OPTIMIZED_MROPE) {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_K_QUANT_UB_BASE + bufferId));
            }
        }
    }

    __aicore__ inline void PrepareBeforeLoop(uint64_t coreTokenBegin, uint64_t coreTokenEnd)
    {
        CopyPersistentInputs();
        BuildRopeNzScatterIndex();
        if constexpr (ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE) {
            BuildMropeGatherIndex();
        }
        if constexpr (OPTIMIZED_MROPE) {
            metadataRangeEnd_ = coreTokenEnd;
            if (coreTokenBegin < coreTokenEnd) {
                LoadMropeMetadataWindow(coreTokenBegin);
            }
        }
        SetFlag<HardEvent::S_V>(static_cast<event_t>(EVT_S_TO_PIPE_V_PERSISTENT_INDEX_READY));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(EVT_S_TO_PIPE_V_PERSISTENT_INDEX_READY));
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        constexpr uint32_t inputBufferNum =
            OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_INPUT_BUFFER_NUM : QKV_K_SCALE_DOUBLE_BUFFER_NUM;
        for (uint32_t bufferId = 0U; bufferId < inputBufferNum; ++bufferId) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
        }
        for (uint32_t bufferId = 0U; bufferId < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
            if constexpr (OPTIMIZED_MROPE) {
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_K_QUANT_UB_BASE + bufferId));
            }
        }
    }

    __aicore__ inline void ComputeTile(TileParam &tile, const LocalTensor<bfloat16_t> &aRotL1Nz,
                                       const LocalTensor<float> &qkPreprocessUb, uint32_t outputBufferId)
    {
        constexpr uint32_t inputBufferNum =
            OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_INPUT_BUFFER_NUM : QKV_K_SCALE_DOUBLE_BUFFER_NUM;
        const uint32_t inputBufferId = static_cast<uint32_t>(inputBufferUseId_ & (inputBufferNum - 1U));
        const uint32_t metadataBufferId = static_cast<uint32_t>(
            OPTIMIZED_MROPE ? inputBufferId : (inputBufferId & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U)));
        if (tile.aivTokenSize == 0U) {
            if constexpr (OPTIMIZED_MROPE) {
                WaitAicMte1ToAivMte3AConsumed();
            }
            ++inputBufferUseId_;
            return;
        }

        if constexpr (OPTIMIZED_MROPE) {
            EnsureMropeMetadataWindow(tile);
        }
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
        CopyPreprocessInputs(tile, inputBufferId, metadataBufferId);
        if constexpr (!OPTIMIZED_MROPE) {
            SetAndWaitMte2ToSSlotMappingReady();
        }
        if constexpr (ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE) {
            CopyMropeCosSinTile(tile, metadataBufferId);
        }
        BuildCacheOffsetsFromSlotMapping(tile, metadataBufferId);
        SetAndWaitMte2ToVInputReady();

        const uint32_t vOutBufferId = static_cast<uint32_t>(vOutBufferUseId_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        QuantVToFp8(tile, inputBufferId, vOutBufferId);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_V_CACHE_READY));

        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        if constexpr (ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE) {
            BuildMropeBf16NzUb(tile, inputBufferId, metadataBufferId, qkPreprocessUb);
        } else {
            BuildRopeBf16NzUb(tile, inputBufferId, qkPreprocessUb);
        }

        if constexpr (OPTIMIZED_MROPE) {
            // QKV and raw cos/sin are no longer read by VF after
            // preprocessing.  Release this slot before the L1 copy so the
            // next tile's MTE2 can overlap the current Cube/FIX work.
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
        }
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_V_CACHE_READY));
        ScatterVCache(tile, vOutBufferId);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        ++vOutBufferUseId_;

        if constexpr (OPTIMIZED_MROPE) {
            // Keep MTE2/VF of this tile ahead of the cross-core L1 reuse gate.
            // Only the MTE3 copy below needs the previous AIC consumer credit.
            WaitAicMte1ToAivMte3AConsumed();
        }
        SetAndWaitQkToMte3Ready();
        CopyRopeBf16NzUbToL1Nz(tile, aRotL1Nz, qkPreprocessUb);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));

        if constexpr (!OPTIMIZED_MROPE) {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
        }
        ++inputBufferUseId_;
    }

    __aicore__ inline void StageMropeK(const TileParam &tile, const LocalTensor<float> &kRotationUb,
                                       uint32_t kQuantBufferId)
    {
        if constexpr (!OPTIMIZED_MROPE) {
            return;
        }
        if (tile.aivTokenSize == 0U) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_K_QUANT_UB_BASE + kQuantBufferId));
        const LocalTensor<int8_t> kInt8 =
            kQuantDbPoolUb_[kQuantBufferId * QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_BYTES];
        const LocalTensor<float> kScale =
            kInt8[QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_K_BYTES].template ReinterpretCast<float>();
        QuantKMropeToInt8Compact(tile, kRotationUb, kInt8, kScale);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_K_QUANT_READY_BASE + kQuantBufferId));
    }

    __aicore__ inline void ScatterMropeK(const TileParam &tile, uint32_t kQuantBufferId)
    {
        if constexpr (!OPTIMIZED_MROPE) {
            return;
        }
        if (tile.aivTokenSize == 0U) {
            return;
        }
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_K_QUANT_READY_BASE + kQuantBufferId));
        const LocalTensor<int8_t> kInt8 =
            kQuantDbPoolUb_[kQuantBufferId * QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_BYTES];
        const LocalTensor<float> kScale =
            kInt8[QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_K_BYTES].template ReinterpretCast<float>();
        ScatterKMropeCompactOutputs(tile, kInt8, kScale);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_K_QUANT_UB_BASE + kQuantBufferId));
    }

    __aicore__ inline void PostprocessQk(const TileParam &tile, const LocalTensor<float> &qkAfterCubeUb,
                                         uint32_t outputBufferId)
    {
        if (tile.aivTokenSize == 0U) {
            return;
        }

        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        const uint32_t vOutBufferId = static_cast<uint32_t>(vOutBufferUseId_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        const LocalTensor<float> qkScaleStaging =
            vOutDbPoolUb_[vOutBufferId * V_OUT_ONE_BUFFER_ELEMENTS].template ReinterpretCast<float>();
        LocalTensor<float> kScale = qkScaleStaging;
        if constexpr (K_QUANT_MODE == QKV_K_SCALE_K_QUANT_MODE_FP8) {
            const uint64_t qScaleNtdHeadStride =
                AlignUp(tile.aivTokenSize, QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS);
            kScale = qkScaleStaging[qHeadNum_ * qScaleNtdHeadStride];
        }

        if constexpr (!DIRECT_MROPE_Q_OUT) {
            if constexpr (Q_QUANT_MODE == QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT) {
                QuantQMropeToBf16(tile, qkAfterCubeUb);
                SetAndWaitQkToMte3Ready();
                StoreQMropeOutputs(tile, qkAfterCubeUb);
            } else {
                QuantQToFp8(tile, qkAfterCubeUb, qkScaleStaging);
                SetAndWaitQkToMte3Ready();
                StoreQOutputs(tile, qkAfterCubeUb, qkScaleStaging);
            }
        }

        if constexpr (K_QUANT_MODE == QKV_K_SCALE_K_QUANT_MODE_INT8) {
            QuantKMropeToInt8(tile, qkAfterCubeUb, kScale);
            SetAndWaitQkToMte3Ready();
            ScatterKMropeOutputs(tile, qkAfterCubeUb, kScale);
        } else {
            QuantKToFp8(tile, qkAfterCubeUb, kScale);
            SetAndWaitQkToMte3Ready();
            ScatterKOutputs(tile, qkAfterCubeUb, kScale);
        }
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        ++vOutBufferUseId_;
    }

private:
    static constexpr uint32_t QK_PREPROCESS_SCATTER_INDEX_TABLE_ELEMENTS =
        QKV_K_SCALE_QK_NZ_SCATTER_INDEX_TABLE_ELEMENTS;
    static constexpr bool QKV_IS_TND = QKV_LAYOUT == QKV_K_SCALE_LAYOUT_TND;
    static constexpr bool Q_OUT_IS_TND = Q_OUT_LAYOUT == QKV_K_SCALE_LAYOUT_TND;
    static constexpr bool DIRECT_MROPE_Q_OUT =
        (QKV_IS_TND && ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE && Q_OUT_IS_TND &&
         K_QUANT_MODE == QKV_K_SCALE_K_QUANT_MODE_INT8 &&
         Q_QUANT_MODE == QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT);
    static constexpr bool OPTIMIZED_MROPE = DIRECT_MROPE_Q_OUT && MROPE_COMPACT_PIPELINE;
    static constexpr uint32_t SLOT_MAPPING_ONE_BUFFER_ELEMENTS =
        OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_ONE_BUFFER_ELEMENTS
                        : QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS;
    static constexpr uint32_t V_OUT_ONE_BUFFER_ELEMENTS =
        OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_V_OUT_ONE_BUFFER_ELEMENTS
                        : QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS;
    static constexpr uint32_t INPUT_ONE_BUFFER_ELEMENTS =
        OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_INPUT_ONE_BUFFER_ELEMENTS
                        : QKV_K_SCALE_INPUT_ONE_BUFFER_ELEMENTS;
    static constexpr uint32_t MROPE_POSITION_ONE_BUFFER_ELEMENTS =
        OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_POSITION_ONE_BUFFER_ELEMENTS
                        : QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_ELEMENTS;

    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_READY = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_K_QUANT_UB_BASE = 4U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_K_QUANT_READY_BASE = 4U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_V_CACHE_READY = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_QK_READY = 1U;
    static constexpr uint32_t EVT_MTE2_TO_S_SLOT_MAPPING_READY = 5U;
    static constexpr uint32_t EVT_S_TO_PIPE_V_PERSISTENT_INDEX_READY = 0U;

    __aicore__ inline void SetAndWaitMte2ToVInputReady()
    {
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_READY));
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_READY));
    }

    __aicore__ inline void SetAndWaitMte2ToSSlotMappingReady()
    {
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(EVT_MTE2_TO_S_SLOT_MAPPING_READY));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(EVT_MTE2_TO_S_SLOT_MAPPING_READY));
    }

    __aicore__ inline void SetAndWaitQkToMte3Ready()
    {
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_QK_READY));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_QK_READY));
    }

    __aicore__ inline void WaitAicMte1ToAivMte3AConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void CopyPersistentInputs()
    {
        DataCopyGmToUb2D(gammaUb_, qGammaGm_, 1U, headDim_, headDim_);
        DataCopyGmToUb2D(gammaUb_[QKV_K_SCALE_GAMMA_UB_ELEMENTS / 2U], kGammaGm_, 1U, headDim_, headDim_);
        DataCopyGmToUb2D(vScaleUb_, vScaleGm_, 1U, kvHeadNum_, kvHeadNum_);
        SetAndWaitMte2ToVInputReady();
    }

    __aicore__ inline void LoadMropeMetadataWindow(uint64_t windowBegin)
    {
        if (windowBegin >= metadataRangeEnd_) {
            return;
        }
        const uint64_t windowTokenSize =
            MinU64(QKV_K_SCALE_MROPE_METADATA_WINDOW_TOKEN_CAPACITY, metadataRangeEnd_ - windowBegin);
        DataCopyGmToUb2D(slotMappingDbPoolUb_, slotMappingGm_[windowBegin], 1U, windowTokenSize, windowTokenSize);
        DataCopyGmToUb2D(mropePositionDbPoolUb_, mropePositionGm_[windowBegin * 3U], 1U,
                         windowTokenSize * 3U, windowTokenSize * 3U);
        SetAndWaitMte2ToSSlotMappingReady();
        metadataWindowBegin_ = windowBegin;
        metadataWindowEnd_ = windowBegin + windowTokenSize;
    }

    __aicore__ inline void EnsureMropeMetadataWindow(const TileParam &tile)
    {
        const uint64_t tileEnd = tile.aivTokenOffset + tile.aivTokenSize;
        if (tile.aivTokenOffset >= metadataWindowBegin_ && tileEnd <= metadataWindowEnd_) {
            return;
        }
        LoadMropeMetadataWindow(tile.tokenOffset);
    }

    __aicore__ inline void CopyPreprocessInputs(const TileParam &tile, uint32_t inputBufferId,
                                                uint32_t metadataBufferId)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * INPUT_ONE_BUFFER_ELEMENTS];
        const uint64_t qkvHeadNum = qHeadNum_ + kvHeadNum_ + kvHeadNum_;
        if constexpr (QKV_IS_TND) {
            DataCopyGmToUb2D(input, qkvGm_[tile.aivTokenOffset * qkvHeadNum * headDim_], 1U,
                             tile.aivTokenSize * qkvHeadNum * headDim_, tile.aivTokenSize * qkvHeadNum * headDim_);
        } else {
            DataCopyGmToUb2D(input, qkvGm_[tile.aivTokenOffset * headDim_], qkvHeadNum, tile.aivTokenSize * headDim_,
                             totalTokens_ * headDim_);
        }
        if constexpr (!OPTIMIZED_MROPE) {
            const LocalTensor<int32_t> slotMapping =
                slotMappingDbPoolUb_[metadataBufferId * SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
            DataCopyGmToUb2D(slotMapping, slotMappingGm_[tile.aivTokenOffset], 1U, tile.aivTokenSize,
                             tile.aivTokenSize);
            if constexpr (ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE) {
                const LocalTensor<int32_t> mropePosition =
                    mropePositionDbPoolUb_[metadataBufferId * MROPE_POSITION_ONE_BUFFER_ELEMENTS];
                // Copy only valid token-major rows; the UB slot is padded to the
                // maximum tile size and must not expand the GM read at the tail.
                DataCopyGmToUb2D(mropePosition, mropePositionGm_[tile.aivTokenOffset * 3U], 1U,
                                 tile.aivTokenSize * 3U, tile.aivTokenSize * 3U);
            } else {
                CopyCosSinTile(tile, metadataBufferId);
            }
        }
    }

    __aicore__ inline void CopyMropeCosSinTile(const TileParam &tile, uint32_t metadataBufferId)
    {
        LocalTensor<int32_t> mropePosition = mropePositionDbPoolUb_;
        if constexpr (!OPTIMIZED_MROPE) {
            mropePosition = mropePositionDbPoolUb_[metadataBufferId * MROPE_POSITION_ONE_BUFFER_ELEMENTS];
        }
        const LocalTensor<float> cosSin =
            cosSinDbPoolUb_[metadataBufferId * QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS];
        const uint64_t metadataTokenOffset =
            OPTIMIZED_MROPE ? tile.aivTokenOffset - metadataWindowBegin_ : 0U;
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            for (uint32_t axis = 0U; axis < 3U; ++axis) {
                const int32_t position = mropePosition.GetValue((metadataTokenOffset + tokenIdx) * 3U + axis);
                const uint64_t sourceOffset = static_cast<uint64_t>(position) * headDim_;
                const uint64_t destinationOffset = (tokenIdx * 3U + axis) * headDim_;
                DataCopyGmToUb2D(cosSin[destinationOffset], cosSinGm_[sourceOffset], 1U, headDim_, headDim_);
            }
        }
    }

    __aicore__ inline void BuildCacheOffsetsFromSlotMapping(TileParam &tile, uint32_t inputBufferId)
    {
        LocalTensor<int32_t> slotMapping = slotMappingDbPoolUb_;
        if constexpr (!OPTIMIZED_MROPE) {
            slotMapping = slotMappingDbPoolUb_[inputBufferId * SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        }
        const uint64_t metadataTokenOffset =
            OPTIMIZED_MROPE ? tile.aivTokenOffset - metadataWindowBegin_ : 0U;
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t slot = static_cast<uint64_t>(slotMapping.GetValue(metadataTokenOffset + tokenIdx));
            const uint64_t blockId = slot / blockSize_;
            const uint64_t blockOffset = slot % blockSize_;
            tile.cacheBaseOffset[tokenIdx] = blockId * kvCacheStrideBlock_ + blockOffset * kvCacheStrideToken_;
            tile.scaleCacheBaseOffset[tokenIdx] =
                blockId * kScaleCacheStrideBlock_ + blockOffset * kScaleCacheStrideToken_;
        }
    }

    __aicore__ inline void CopyCosSinTile(const TileParam &tile, uint32_t inputBufferId)
    {
        const LocalTensor<float> cosSin = cosSinDbPoolUb_[inputBufferId * QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS];
        uint64_t runStart = 0U;
        while (runStart < tile.aivTokenSize) {
            const uint64_t tokenOffset = tile.aivTokenOffset + runStart;
            uint64_t runPosition = tokenOffset;
            uint64_t runSize = tile.aivTokenSize - runStart;
            uint64_t seqBegin = 0U;
            uint64_t seqEnd = 0U;
            while (cosSinBatchIdx_ < batch_) {
                seqBegin = static_cast<uint64_t>(queryStartLocGm_.GetValue(cosSinBatchIdx_));
                seqEnd = static_cast<uint64_t>(queryStartLocGm_.GetValue(cosSinBatchIdx_ + 1U));
                if (tokenOffset < seqEnd) {
                    break;
                }
                ++cosSinBatchIdx_;
            }
            if (cosSinBatchIdx_ < batch_ && tokenOffset >= seqBegin) {
                const uint64_t localLen = seqEnd - seqBegin;
                const uint64_t actualLen = static_cast<uint64_t>(seqLensGm_.GetValue(cosSinBatchIdx_));
                const uint64_t historyOffset = actualLen > localLen ? actualLen - localLen : 0U;
                runPosition = historyOffset + tokenOffset - seqBegin;
                runSize = MinU64(runSize, seqEnd - tokenOffset);
            }
            DataCopyGmToUb2D(cosSin[runStart * headDim_], cosSinGm_[runPosition * headDim_], runSize, headDim_,
                             headDim_);
            runStart += runSize;
        }
    }

    __aicore__ inline void BuildRopeBf16NzUb(const TileParam &tile, uint32_t inputBufferId,
                                             const LocalTensor<float> &qkPreprocessUb)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * QKV_K_SCALE_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> gamma = gammaUb_;
        const LocalTensor<float> cosSin = cosSinDbPoolUb_[inputBufferId * QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qRopeNz = qkPreprocessUb.template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[qPreprocessElements_];
        const LocalTensor<uint16_t> kNzScatterIndex = qkNzScatterIndexUb_[QK_PREPROCESS_SCATTER_INDEX_TABLE_ELEMENTS];
        const uint32_t tokenCapacity = static_cast<uint32_t>(tile.cubeHalfTokenSize);
        uint32_t inputTokenStride;
        uint32_t inputHeadStride;
        uint32_t qOutputTokenStride;
        uint32_t kOutputTokenStride;
        uint32_t outputHeadStride;
        if constexpr (QKV_IS_TND) {
            const uint32_t qkvHeadNum = static_cast<uint32_t>(qHeadNum_ + kvHeadNum_ + kvHeadNum_);
            inputTokenStride = qkvHeadNum * QKV_K_SCALE_D128_FULL_SIZE;
            inputHeadStride = QKV_K_SCALE_D128_FULL_SIZE;
        } else {
            inputTokenStride = QKV_K_SCALE_D128_FULL_SIZE;
            inputHeadStride = static_cast<uint32_t>(tile.aivTokenSize) * QKV_K_SCALE_D128_FULL_SIZE;
        }
        if constexpr (Q_OUT_IS_TND) {
            qOutputTokenStride = static_cast<uint32_t>(qHeadNum_);
            kOutputTokenStride = static_cast<uint32_t>(kvHeadNum_);
            outputHeadStride = 1U;
        } else {
            qOutputTokenStride = 1U;
            kOutputTokenStride = 1U;
            outputHeadStride = tokenCapacity;
        }
        AscendC::VF_CALL<QkRmsNormRopeD128SegmentNzVfImpl>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)gamma.GetPhyAddr(),
            (__ubuf__ float *)cosSin.GetPhyAddr(), (__ubuf__ bfloat16_t *)qRopeNz.GetPhyAddr(),
            (__ubuf__ uint16_t *)qkNzScatterIndexUb_.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
            static_cast<uint16_t>(qHeadNum_), inputTokenStride, inputHeadStride, qOutputTokenStride, outputHeadStride,
            qPreprocessRowStride_, epsilon_);
        const uint32_t kInputOffset = static_cast<uint32_t>(qHeadNum_) * inputHeadStride;
        AscendC::VF_CALL<QkRmsNormRopeD128SegmentNzVfImpl>(
            (__ubuf__ bfloat16_t *)input[kInputOffset].GetPhyAddr(),
            (__ubuf__ float *)(gamma[QKV_K_SCALE_HEAD_DIM_D128].GetPhyAddr()), (__ubuf__ float *)cosSin.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)kRopeNz.GetPhyAddr(), (__ubuf__ uint16_t *)kNzScatterIndex.GetPhyAddr(),
            static_cast<uint16_t>(tile.aivTokenSize), static_cast<uint16_t>(kvHeadNum_), inputTokenStride,
            inputHeadStride, kOutputTokenStride, outputHeadStride, kPreprocessRowStride_, epsilon_);
    }

    __aicore__ inline void BuildRopeNzScatterIndex()
    {
        BuildRopeNzScatterIndexTable(qkNzScatterIndexUb_, qPreprocessRowStride_);
        BuildRopeNzScatterIndexTable(qkNzScatterIndexUb_[QK_PREPROCESS_SCATTER_INDEX_TABLE_ELEMENTS],
                                     kPreprocessRowStride_);
    }

    __aicore__ inline void BuildMropeGatherIndex()
    {
        // BindLocalTensors selects the compact or generic reserve layout at
        // compile time.  Reusing that tensor is important: rebuilding it from
        // the generic offset would move compact mode outside its 36 KiB
        // reserve and alias the neighboring slots.
        const LocalTensor<uint32_t> gatherIndex = mropeGatherIndexUb_;
        for (uint32_t lane = 0U; lane < QKV_K_SCALE_D128_HALF_SIZE; ++lane) {
            const uint32_t repeat = lane / 3U;
            const uint32_t axisInGroup = lane % 3U;
            uint32_t axis = 0U;
            if (axisInGroup == 1U && repeat < mropeSectionH_) {
                axis = 1U;
            } else if (axisInGroup == 2U && repeat < mropeSectionW_) {
                axis = 2U;
            }
            gatherIndex.SetValue(lane, axis * QKV_K_SCALE_D128_FULL_SIZE + lane);
        }
    }

    __aicore__ inline void BuildMropeBf16NzUb(const TileParam &tile, uint32_t inputBufferId,
                                              uint32_t metadataBufferId,
                                              const LocalTensor<float> &qkPreprocessUb)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> gamma = gammaUb_;
        const LocalTensor<float> rawCosSin =
            cosSinDbPoolUb_[metadataBufferId * QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qRopeNz = qkPreprocessUb.template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[qPreprocessElements_];
        const LocalTensor<uint16_t> kNzScatterIndex = qkNzScatterIndexUb_[QK_PREPROCESS_SCATTER_INDEX_TABLE_ELEMENTS];
        const uint32_t qkvHeadNum = static_cast<uint32_t>(qHeadNum_ + kvHeadNum_ + kvHeadNum_);
        const uint32_t inputTokenStride = qkvHeadNum * QKV_K_SCALE_D128_FULL_SIZE;
        const uint32_t inputHeadStride = QKV_K_SCALE_D128_FULL_SIZE;
        const uint32_t outputTokenStride = static_cast<uint32_t>(qHeadNum_);
        const uint32_t kOutputTokenStride = static_cast<uint32_t>(kvHeadNum_);
        const uint32_t outputHeadStride = 1U;
        AscendC::VF_CALL<QkRmsNormMropeD128SegmentNzVfImpl>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)gamma.GetPhyAddr(),
            (__ubuf__ float *)rawCosSin.GetPhyAddr(), (__ubuf__ uint32_t *)mropeGatherIndexUb_.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)qRopeNz.GetPhyAddr(), (__ubuf__ uint16_t *)qkNzScatterIndexUb_.GetPhyAddr(),
            static_cast<uint16_t>(tile.aivTokenSize), static_cast<uint16_t>(qHeadNum_), inputTokenStride,
            inputHeadStride, outputTokenStride, outputHeadStride, qPreprocessRowStride_, epsilon_);
        const uint32_t kInputOffset = static_cast<uint32_t>(qHeadNum_) * inputHeadStride;
        AscendC::VF_CALL<QkRmsNormMropeD128SegmentNzVfImpl>(
            (__ubuf__ bfloat16_t *)input[kInputOffset].GetPhyAddr(),
            (__ubuf__ float *)(gamma[QKV_K_SCALE_HEAD_DIM_D128].GetPhyAddr()),
            (__ubuf__ float *)rawCosSin.GetPhyAddr(), (__ubuf__ uint32_t *)mropeGatherIndexUb_.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)kRopeNz.GetPhyAddr(), (__ubuf__ uint16_t *)kNzScatterIndex.GetPhyAddr(),
            static_cast<uint16_t>(tile.aivTokenSize), static_cast<uint16_t>(kvHeadNum_), inputTokenStride,
            inputHeadStride, kOutputTokenStride, outputHeadStride, kPreprocessRowStride_, epsilon_);
    }

    __aicore__ inline void BuildRopeNzScatterIndexTable(const LocalTensor<uint16_t> &scatterIndex, uint32_t rowStride)
    {
        for (uint32_t dim = 0U; dim < QKV_K_SCALE_D128_HALF_SIZE; ++dim) {
            const uint32_t dBlock = dim / QKV_K_SCALE_NZ_C0;
            const uint32_t dInner = dim % QKV_K_SCALE_NZ_C0;
            // BF16 values produced by f32->bf16 cast occupy one lane in each 32-bit slot.
            scatterIndex.SetValue(2U * dim, static_cast<uint16_t>(dBlock * rowStride * QKV_K_SCALE_NZ_C0 + dInner));
        }
    }

    __aicore__ inline void CopyRopeBf16NzUbToL1Nz(const TileParam &tile, const LocalTensor<bfloat16_t> &aRotL1,
                                                  const LocalTensor<float> &qkPreprocessUb)
    {
        if (tile.aivTokenSize == 0U) {
            return;
        }
        const uint64_t qTileRows = tile.cubeTokenSize * qHeadNum_;
        const uint64_t kTileRows = tile.cubeTokenSize * kvHeadNum_;
        const LocalTensor<bfloat16_t> qRopeNz = qkPreprocessUb.template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[qPreprocessElements_];
        CopyRopeSegmentBf16LayoutNzUbToL1Nz(aRotL1, qRopeNz, qTileRows, qHeadNum_, tile.cubeHalfTokenSize,
                                            tile.aivBlockTokenOffset, tile.aivTokenSize, qPreprocessRowStride_);
        CopyRopeSegmentBf16LayoutNzUbToL1Nz(aRotL1[NzMatrixElements(qTileRows)], kRopeNz, kTileRows, kvHeadNum_,
                                            tile.cubeHalfTokenSize, tile.aivBlockTokenOffset, tile.aivTokenSize,
                                            kPreprocessRowStride_);
    }

    __aicore__ inline void CopyRopeSegmentBf16LayoutNzUbToL1Nz(const LocalTensor<bfloat16_t> &dstSegment,
                                                               const LocalTensor<bfloat16_t> &srcSegment,
                                                               uint64_t tileSegmentRows, uint64_t headSize,
                                                               uint64_t tokenCapacity, uint64_t aivBlockTokenOffset,
                                                               uint64_t tokenSize, uint32_t rowStride)
    {
        if (tokenSize == 0U || headSize == 0U) {
            return;
        }
        const uint64_t copyRows = headSize * tokenCapacity;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(QKV_K_SCALE_HEAD_DIM_D128 / QKV_K_SCALE_NZ_C0);
        copyParams.blockLen = static_cast<uint16_t>(copyRows);
        copyParams.srcStride = static_cast<uint16_t>(rowStride - copyRows);
        copyParams.dstStride = static_cast<uint16_t>(AlignUp(tileSegmentRows, QKV_K_SCALE_NZ_C0) - copyRows);
        const uint64_t dstRowOffset = aivBlockTokenOffset * headSize;
        DataCopy(dstSegment[dstRowOffset * QKV_K_SCALE_NZ_C0], srcSegment, copyParams);
    }

    __aicore__ inline void QuantVToFp8(const TileParam &tile, uint32_t inputBufferId, uint32_t vOutBufferId)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> vScale = vScaleUb_;
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[vOutBufferId * V_OUT_ONE_BUFFER_ELEMENTS];
        const uint64_t vInputHeadBase = qHeadNum_ + kvHeadNum_;
        uint32_t inputTokenStride;
        uint32_t inputHeadStride;
        if constexpr (QKV_IS_TND) {
            const uint32_t qkvHeadNum = static_cast<uint32_t>(qHeadNum_ + kvHeadNum_ + kvHeadNum_);
            inputTokenStride = qkvHeadNum * QKV_K_SCALE_D128_FULL_SIZE;
            inputHeadStride = QKV_K_SCALE_D128_FULL_SIZE;
        } else {
            inputTokenStride = QKV_K_SCALE_D128_FULL_SIZE;
            inputHeadStride = static_cast<uint32_t>(tile.aivTokenSize) * QKV_K_SCALE_D128_FULL_SIZE;
        }
        const uint32_t vInputOffset = static_cast<uint32_t>(vInputHeadBase) * inputHeadStride;
        AscendC::VF_CALL<VScaleFp8D128ToNtdVfImpl>(
            (__ubuf__ bfloat16_t *)input[vInputOffset].GetPhyAddr(), (__ubuf__ float *)vScale.GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)vOut.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
            static_cast<uint16_t>(tile.vHeadSize), inputTokenStride, inputHeadStride);
    }

    __aicore__ inline void ScatterVCache(const TileParam &tile, uint32_t vOutBufferId)
    {
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[vOutBufferId * V_OUT_ONE_BUFFER_ELEMENTS];
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t ubOffset = tokenIdx * headDim_;
            DataCopyUbToGm2D(vCacheOutGm_[tile.cacheBaseOffset[tokenIdx]], vOut[ubOffset], tile.vHeadSize, headDim_,
                             tile.aivTokenSize * headDim_, kvCacheStrideHead_);
        }
    }

    __aicore__ inline void QuantQToFp8(const TileParam &tile, const LocalTensor<float> &qkAfterCube,
                                       const LocalTensor<float> &qScale)
    {
        const LocalTensor<fp8_e4m3fn_t> qFp8 = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        // Q dynamic quant keeps two VF entries. NTD writes compact contiguous head-major q/q_scale
        // from a padded cube buffer; reusing the TND address formula can make NTD stores overlap.
        if constexpr (Q_OUT_IS_TND) {
            AscendC::VF_CALL<QDynamicQuantD128TndVfImpl>(
                (__ubuf__ float *)qkAfterCube.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)qFp8.GetPhyAddr(),
                (__ubuf__ float *)qScale.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
                static_cast<uint16_t>(qHeadNum_));
        } else {
            AscendC::VF_CALL<QDynamicQuantD128NtdVfImpl>(
                (__ubuf__ float *)qkAfterCube.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)qFp8.GetPhyAddr(),
                (__ubuf__ float *)qScale.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
                static_cast<uint16_t>(qHeadNum_), static_cast<uint32_t>(tile.cubeHalfTokenSize));
        }
    }

    __aicore__ inline void QuantKToFp8(const TileParam &tile, const LocalTensor<float> &qkAfterCube,
                                       const LocalTensor<float> &kScale)
    {
        const uint64_t kRowOffset = tile.cubeHalfTokenSize * qHeadNum_ * headDim_;
        const LocalTensor<fp8_e4m3fn_t> kFp8 = qkAfterCube[kRowOffset].template ReinterpretCast<fp8_e4m3fn_t>();
        constexpr uint32_t fp32RowBytes = QKV_K_SCALE_D128_FULL_SIZE * sizeof(float);
        uint32_t inputHeadStride;
        uint32_t inputTokenStride;
        uint32_t outputHeadStrideBytes;
        uint32_t outputTokenStrideBytes;
        if constexpr (Q_OUT_IS_TND) {
            inputHeadStride = QKV_K_SCALE_D128_FULL_SIZE;
            inputTokenStride = static_cast<uint32_t>(kvHeadNum_) * QKV_K_SCALE_D128_FULL_SIZE;
            outputHeadStrideBytes = fp32RowBytes;
            outputTokenStrideBytes = static_cast<uint32_t>(kvHeadNum_) * fp32RowBytes;
        } else {
            inputHeadStride = static_cast<uint32_t>(tile.cubeHalfTokenSize) * QKV_K_SCALE_D128_FULL_SIZE;
            inputTokenStride = QKV_K_SCALE_D128_FULL_SIZE;
            outputHeadStrideBytes = static_cast<uint32_t>(tile.cubeHalfTokenSize) * fp32RowBytes;
            outputTokenStrideBytes = fp32RowBytes;
        }
        AscendC::VF_CALL<KDynamicQuantD128VfImpl>(
            (__ubuf__ float *)qkAfterCube[kRowOffset].GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)kFp8.GetPhyAddr(),
            (__ubuf__ float *)kScale.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
            static_cast<uint16_t>(kvHeadNum_), inputHeadStride, inputTokenStride, outputHeadStrideBytes,
            outputTokenStrideBytes);
    }

    __aicore__ inline void StoreQOutputs(const TileParam &tile, const LocalTensor<float> &qkAfterCube,
                                         const LocalTensor<float> &qScale)
    {
        const LocalTensor<fp8_e4m3fn_t> qFp8 = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        if constexpr (Q_OUT_IS_TND) {
            DataCopyUbToGm2D(qOutGm_[tile.aivTokenOffset * qHeadNum_ * headDim_], qFp8, tile.aivTokenSize * qHeadNum_,
                             headDim_, headDim_, headDim_);
            DataCopyUbToGm2D(qScaleGm_[tile.aivTokenOffset * qHeadNum_], qScale, 1U, tile.aivTokenSize * qHeadNum_,
                             tile.aivTokenSize * qHeadNum_, tile.aivTokenSize * qHeadNum_);
        } else {
            const uint64_t qScaleNtdHeadStride = AlignUp(tile.aivTokenSize, QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS);
            DataCopyUbToGm2D(qOutGm_[tile.aivTokenOffset * headDim_], qFp8, qHeadNum_, tile.aivTokenSize * headDim_,
                             tile.aivTokenSize * headDim_, totalTokens_ * headDim_);
            DataCopyUbToGm2D(qScaleGm_[tile.aivTokenOffset], qScale, qHeadNum_, tile.aivTokenSize, qScaleNtdHeadStride,
                             totalTokens_);
        }
    }

    __aicore__ inline void ScatterKOutputs(const TileParam &tile, const LocalTensor<float> &qkAfterCube,
                                           const LocalTensor<float> &kScale)
    {
        const uint64_t kRowOffset = tile.cubeHalfTokenSize * qHeadNum_ * headDim_;
        const LocalTensor<fp8_e4m3fn_t> kFp8 = qkAfterCube[kRowOffset].template ReinterpretCast<fp8_e4m3fn_t>();
        const uint64_t kFp8SparseRowStride = headDim_ * sizeof(float) / sizeof(fp8_e4m3fn_t);
        uint64_t kUbTokenStride;
        uint64_t kUbHeadStride;
        if constexpr (Q_OUT_IS_TND) {
            kUbTokenStride = kvHeadNum_ * kFp8SparseRowStride;
            kUbHeadStride = kFp8SparseRowStride;
        } else {
            kUbTokenStride = kFp8SparseRowStride;
            kUbHeadStride = tile.cubeHalfTokenSize * kFp8SparseRowStride;
        }
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t kUbOffset = tokenIdx * kUbTokenStride;
            const uint64_t kCacheOffset = tile.cacheBaseOffset[tokenIdx];
            const uint64_t kScaleOffset = tile.scaleCacheBaseOffset[tokenIdx];
            DataCopyUbToGm2D(kCacheOutGm_[kCacheOffset], kFp8[kUbOffset], kvHeadNum_, headDim_, kUbHeadStride,
                             kvCacheStrideHead_);
            const uint64_t kScaleUbOffset =
                tokenIdx * kvHeadNum_ * QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS;
            DataCopyUbToGm2D(kScaleCacheOutGm_[kScaleOffset], kScale[kScaleUbOffset], kvHeadNum_, 1U, 1U,
                             kScaleCacheStrideHead_);
        }
    }

    __aicore__ inline void QuantQMropeToBf16(const TileParam &tile, const LocalTensor<float> &qkAfterCube)
    {
        // The VF loads both FP32 halves before writing the compact BF16 row.
        // Rows are processed in order, so the 2-byte destination footprint
        // cannot overwrite a source row that has not been consumed yet.
        LocalTensor<bfloat16_t> qBf16 = qkAfterCube.template ReinterpretCast<bfloat16_t>();
        AscendC::VF_CALL<QCastBf16D128TndVfImpl>(
            (__ubuf__ float *)qkAfterCube.GetPhyAddr(), (__ubuf__ bfloat16_t *)qBf16.GetPhyAddr(),
            static_cast<uint16_t>(tile.aivTokenSize), static_cast<uint16_t>(qHeadNum_));
    }

    __aicore__ inline void StoreQMropeOutputs(const TileParam &tile, const LocalTensor<float> &qkAfterCube)
    {
        const LocalTensor<bfloat16_t> qBf16 = qkAfterCube.template ReinterpretCast<bfloat16_t>();
        DataCopyUbToGm2D(qOutBf16Gm_[tile.aivTokenOffset * qHeadNum_ * headDim_], qBf16,
                         tile.aivTokenSize * qHeadNum_, headDim_, headDim_, headDim_);
    }

    __aicore__ inline void QuantKMropeToInt8(const TileParam &tile, const LocalTensor<float> &qkAfterCube,
                                             const LocalTensor<float> &kScale)
    {
        const uint64_t kRowOffset = tile.cubeHalfTokenSize * qHeadNum_ * headDim_;
        const LocalTensor<int8_t> kInt8 = qkAfterCube[kRowOffset].template ReinterpretCast<int8_t>();
        constexpr uint32_t fp32RowBytes = QKV_K_SCALE_D128_FULL_SIZE * sizeof(float);
        const uint32_t inputHeadStride = QKV_K_SCALE_D128_FULL_SIZE;
        const uint32_t inputTokenStride = static_cast<uint32_t>(kvHeadNum_) * QKV_K_SCALE_D128_FULL_SIZE;
        const uint32_t outputHeadStrideBytes = fp32RowBytes;
        const uint32_t outputTokenStrideBytes = static_cast<uint32_t>(kvHeadNum_) * fp32RowBytes;
        AscendC::VF_CALL<KDynamicQuantInt8D128VfImpl>(
            (__ubuf__ float *)qkAfterCube[kRowOffset].GetPhyAddr(), (__ubuf__ int8_t *)kInt8.GetPhyAddr(),
            (__ubuf__ float *)kScale.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
            static_cast<uint16_t>(kvHeadNum_), inputHeadStride, inputTokenStride, outputHeadStrideBytes,
            outputTokenStrideBytes);
    }

    __aicore__ inline void QuantKMropeToInt8Compact(const TileParam &tile, const LocalTensor<float> &kRotationUb,
                                                   const LocalTensor<int8_t> &kInt8,
                                                   const LocalTensor<float> &kScale)
    {
        const uint32_t inputHeadStride = QKV_K_SCALE_D128_FULL_SIZE;
        const uint32_t inputTokenStride = static_cast<uint32_t>(kvHeadNum_) * QKV_K_SCALE_D128_FULL_SIZE;
        const uint32_t outputHeadStrideBytes = QKV_K_SCALE_HEAD_DIM_D128;
        const uint32_t outputTokenStrideBytes = static_cast<uint32_t>(kvHeadNum_) * QKV_K_SCALE_HEAD_DIM_D128;
        AscendC::VF_CALL<KDynamicQuantInt8D128VfImpl>(
            (__ubuf__ float *)kRotationUb.GetPhyAddr(), (__ubuf__ int8_t *)kInt8.GetPhyAddr(),
            (__ubuf__ float *)kScale.GetPhyAddr(), static_cast<uint16_t>(tile.aivTokenSize),
            static_cast<uint16_t>(kvHeadNum_), inputHeadStride, inputTokenStride, outputHeadStrideBytes,
            outputTokenStrideBytes);
    }

    __aicore__ inline void ScatterKMropeOutputs(const TileParam &tile, const LocalTensor<float> &qkAfterCube,
                                                const LocalTensor<float> &kScale)
    {
        const uint64_t kRowOffset = tile.cubeHalfTokenSize * qHeadNum_ * headDim_;
        const LocalTensor<int8_t> kInt8 = qkAfterCube[kRowOffset].template ReinterpretCast<int8_t>();
        const uint64_t kSparseRowStride = headDim_ * sizeof(float) / sizeof(int8_t);
        const uint64_t kUbTokenStride = kvHeadNum_ * kSparseRowStride;
        const uint64_t kUbHeadStride = kSparseRowStride;
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t kUbOffset = tokenIdx * kUbTokenStride;
            const uint64_t kCacheOffset = tile.cacheBaseOffset[tokenIdx];
            const uint64_t kScaleOffset = tile.scaleCacheBaseOffset[tokenIdx];
            DataCopyUbToGm2D(kCacheInt8OutGm_[kCacheOffset], kInt8[kUbOffset], kvHeadNum_, headDim_, kUbHeadStride,
                             kvCacheStrideHead_);
            const uint64_t kScaleUbOffset =
                tokenIdx * kvHeadNum_ * QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS;
            DataCopyUbToGm2D(kScaleCacheOutGm_[kScaleOffset], kScale[kScaleUbOffset], kvHeadNum_, 1U, 1U,
                             kScaleCacheStrideHead_);
        }
    }

    __aicore__ inline void ScatterKMropeCompactOutputs(const TileParam &tile, const LocalTensor<int8_t> &kInt8,
                                                       const LocalTensor<float> &kScale)
    {
        const uint64_t kUbTokenStride = kvHeadNum_ * QKV_K_SCALE_HEAD_DIM_D128;
        const uint64_t kUbHeadStride = QKV_K_SCALE_HEAD_DIM_D128;
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t kUbOffset = tokenIdx * kUbTokenStride;
            const uint64_t kCacheOffset = tile.cacheBaseOffset[tokenIdx];
            const uint64_t kScaleOffset = tile.scaleCacheBaseOffset[tokenIdx];
            DataCopyUbToGm2D(kCacheInt8OutGm_[kCacheOffset], kInt8[kUbOffset], kvHeadNum_, headDim_, kUbHeadStride,
                             kvCacheStrideHead_);
            const uint64_t kScaleUbOffset =
                tokenIdx * kvHeadNum_ * QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS;
            DataCopyUbToGm2D(kScaleCacheOutGm_[kScaleOffset], kScale[kScaleUbOffset], kvHeadNum_, 1U, 1U,
                             kScaleCacheStrideHead_);
        }
    }

    __aicore__ inline void BindLocalTensors()
    {
        constexpr uint32_t reserveOffset = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_OFFSET
                                                           : QKV_K_SCALE_RESERVE_UB_OFFSET;
        constexpr uint32_t gammaOffset = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_GAMMA_UB_OFFSET
                                                         : QKV_K_SCALE_GAMMA_UB_OFFSET;
        constexpr uint32_t gammaElements = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_GAMMA_UB_ELEMENTS
                                                           : QKV_K_SCALE_GAMMA_UB_ELEMENTS;
        constexpr uint32_t vScaleOffset = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_OFFSET
                                                          : QKV_K_SCALE_V_SCALE_UB_OFFSET;
        constexpr uint32_t vScaleElements = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_ELEMENTS
                                                             : QKV_K_SCALE_V_SCALE_UB_ELEMENTS;
        constexpr uint32_t indexOffset = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_OFFSET
                                                         : QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_OFFSET;
        constexpr uint32_t indexElements =
            OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_ELEMENTS
                            : QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_ELEMENTS;
        constexpr uint32_t cosSinOffset = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_OFFSET
                                                           : QKV_K_SCALE_COS_SIN_DB_POOL_OFFSET;
        constexpr uint32_t cosSinElements =
            OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_ELEMENTS
                            : QKV_K_SCALE_COS_SIN_DB_POOL_ELEMENTS;
        constexpr uint32_t vOutOffset = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_OFFSET
                                                        : QKV_K_SCALE_V_OUT_DB_POOL_OFFSET;
        constexpr uint32_t vOutElements = OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_ELEMENTS
                                                          : QKV_K_SCALE_V_OUT_DB_POOL_ELEMENTS;
        inputDbPoolUb_ = LocalTensor<bfloat16_t>(TPosition::LCM,
                                                 OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_INPUT_DB_POOL_OFFSET
                                                                 : QKV_K_SCALE_INPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_INPUT_DB_POOL_ELEMENTS);
        gammaUb_ = LocalTensor<float>(TPosition::LCM, reserveOffset + gammaOffset, gammaElements);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM, reserveOffset + vScaleOffset, vScaleElements);
        qkNzScatterIndexUb_ = LocalTensor<uint16_t>(TPosition::LCM, reserveOffset + indexOffset, indexElements);
        cosSinDbPoolUb_ = LocalTensor<float>(TPosition::LCM, reserveOffset + cosSinOffset, cosSinElements);
        if constexpr (OPTIMIZED_MROPE) {
            slotMappingDbPoolUb_ = LocalTensor<int32_t>(TPosition::LCM,
                                                        QKV_K_SCALE_MROPE_METADATA_SLOT_CACHE_OFFSET,
                                                        QKV_K_SCALE_MROPE_METADATA_SLOT_CACHE_ELEMENTS);
        } else {
            slotMappingDbPoolUb_ = LocalTensor<int32_t>(TPosition::LCM,
                                                        reserveOffset + QKV_K_SCALE_SLOT_MAPPING_DB_POOL_OFFSET,
                                                        QKV_K_SCALE_SLOT_MAPPING_DB_POOL_ELEMENTS);
        }
        vOutDbPoolUb_ = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, reserveOffset + vOutOffset, vOutElements);
        if constexpr (ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE) {
            if constexpr (OPTIMIZED_MROPE) {
                mropePositionDbPoolUb_ = LocalTensor<int32_t>(
                    TPosition::LCM, QKV_K_SCALE_MROPE_METADATA_POSITION_CACHE_OFFSET,
                    QKV_K_SCALE_MROPE_METADATA_POSITION_CACHE_ELEMENTS);
            } else {
                mropePositionDbPoolUb_ = LocalTensor<int32_t>(
                    TPosition::LCM, reserveOffset + QKV_K_SCALE_MROPE_POSITION_DB_POOL_OFFSET,
                    QKV_K_SCALE_MROPE_POSITION_DB_POOL_ELEMENTS);
            }
            mropeGatherIndexUb_ = LocalTensor<uint32_t>(
                TPosition::LCM,
                reserveOffset + (OPTIMIZED_MROPE ? QKV_K_SCALE_MROPE_COMPACT_GATHER_INDEX_UB_OFFSET
                                                : QKV_K_SCALE_MROPE_GATHER_INDEX_OFFSET),
                QKV_K_SCALE_MROPE_GATHER_INDEX_ELEMENTS);
        }
        if constexpr (OPTIMIZED_MROPE) {
            kQuantDbPoolUb_ = LocalTensor<int8_t>(
                TPosition::LCM,
                QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_OFFSET + QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_OFFSET,
                QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_ELEMENTS);
        }
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        qkvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qkv);
        qkvGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        qGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.qGamma);
        kGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.kGamma);
        cosSinGm_.SetGlobalBuffer((__gm__ float *)tensors.cosSin);
        slotMappingGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.slotMapping);
        if constexpr (ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE) {
            mropePositionGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.mropePosition);
        } else {
            queryStartLocGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.queryStartLoc);
            seqLensGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.seqLens);
        }
        if constexpr (Q_QUANT_MODE == QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT) {
            qOutBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qOut);
        } else {
            qOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
            qScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.qScale);
        }
        if constexpr (K_QUANT_MODE == QKV_K_SCALE_K_QUANT_MODE_INT8) {
            kCacheInt8OutGm_.SetGlobalBuffer((__gm__ int8_t *)tensors.kCacheOut);
        } else {
            kCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
        }
        vScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        vCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCacheOut);
        kScaleCacheOutGm_.SetGlobalBuffer((__gm__ float *)tensors.kScaleCacheOut);
    }

    uint64_t totalTokens_;
    uint64_t batch_;
    uint64_t qHeadNum_;
    uint64_t kvHeadNum_;
    uint64_t headDim_;
    uint64_t blockSize_;
    uint64_t kvCacheStrideBlock_;
    uint64_t kvCacheStrideHead_;
    uint64_t kvCacheStrideToken_;
    uint64_t kScaleCacheStrideBlock_;
    uint64_t kScaleCacheStrideHead_;
    uint64_t kScaleCacheStrideToken_;
    uint64_t mropeSectionH_;
    uint64_t mropeSectionW_;
    float epsilon_;
    uint64_t inputBufferUseId_;
    uint64_t vOutBufferUseId_;
    uint64_t cosSinBatchIdx_;
    uint64_t metadataWindowBegin_;
    uint64_t metadataWindowEnd_;
    uint64_t metadataRangeEnd_;
    uint32_t qPreprocessRowStride_;
    uint32_t kPreprocessRowStride_;
    uint32_t qPreprocessElements_;
    GlobalTensor<bfloat16_t> qkvGm_;
    GlobalTensor<float> qGammaGm_;
    GlobalTensor<float> kGammaGm_;
    GlobalTensor<float> cosSinGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<int32_t> mropePositionGm_;
    GlobalTensor<float> vScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> qOutGm_;
    GlobalTensor<bfloat16_t> qOutBf16Gm_;
    GlobalTensor<float> qScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> kCacheOutGm_;
    GlobalTensor<int8_t> kCacheInt8OutGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheOutGm_;
    GlobalTensor<float> kScaleCacheOutGm_;
    LocalTensor<bfloat16_t> inputDbPoolUb_;
    LocalTensor<float> gammaUb_;
    LocalTensor<float> cosSinDbPoolUb_;
    LocalTensor<int32_t> slotMappingDbPoolUb_;
    LocalTensor<float> vScaleUb_;
    LocalTensor<uint16_t> qkNzScatterIndexUb_;
    LocalTensor<fp8_e4m3fn_t> vOutDbPoolUb_;
    LocalTensor<int32_t> mropePositionDbPoolUb_;
    LocalTensor<uint32_t> mropeGatherIndexUb_;
    LocalTensor<int8_t> kQuantDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
