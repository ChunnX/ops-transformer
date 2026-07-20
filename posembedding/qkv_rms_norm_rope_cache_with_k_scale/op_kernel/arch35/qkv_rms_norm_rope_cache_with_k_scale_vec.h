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

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT>
class QkvRmsNormRopeCacheWithKScaleVec {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch, uint64_t qHeadNum,
                                uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize, uint64_t tokenTile,
                                uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead, uint64_t kvCacheStrideToken,
                                uint64_t kScaleCacheStrideBlock, uint64_t kScaleCacheStrideHead,
                                uint64_t kScaleCacheStrideToken, float epsilon)
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
        inputBufferUseId_ = 0U;
        vOutBufferUseId_ = 0U;
        cosSinBatchIdx_ = 0U;
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
        for (uint32_t bufferId = 0U; bufferId < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++bufferId) {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
        }
    }

    __aicore__ inline void PrepareBeforeLoop()
    {
        CopyPersistentInputs();
        BuildRopeNzScatterIndex();
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_K_SCALE_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
        }
    }

    __aicore__ inline void ComputeTile(TileParam &tile, const LocalTensor<bfloat16_t> &aRotL1Nz,
                                       const LocalTensor<float> &qkPreprocessUb, uint32_t outputBufferId)
    {
        const uint32_t inputBufferId = static_cast<uint32_t>(inputBufferUseId_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        if (tile.aivTokenSize == 0U) {
            ++inputBufferUseId_;
            return;
        }

        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
        CopyPreprocessInputs(tile, inputBufferId);
        SetAndWaitMte2ToSSlotMappingReady();
        BuildCacheOffsetsFromSlotMapping(tile, inputBufferId);
        SetAndWaitMte2ToVInputReady();

        const uint32_t vOutBufferId = static_cast<uint32_t>(vOutBufferUseId_ & (QKV_K_SCALE_DOUBLE_BUFFER_NUM - 1U));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        QuantVToFp8(tile, inputBufferId, vOutBufferId);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_V_CACHE_READY));

        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        BuildRopeBf16NzUb(tile, inputBufferId, qkPreprocessUb);

        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_V_CACHE_READY));
        ScatterVCache(tile, vOutBufferId);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        ++vOutBufferUseId_;

        SetAndWaitQkToMte3Ready();
        CopyRopeBf16NzUbToL1Nz(tile, aRotL1Nz, qkPreprocessUb);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));

        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
        ++inputBufferUseId_;
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
            vOutDbPoolUb_[vOutBufferId * QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS].template ReinterpretCast<float>();
        const uint64_t qScaleNtdHeadStride = AlignUp(tile.aivTokenSize, QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS);
        const LocalTensor<float> kScale = qkScaleStaging[qHeadNum_ * qScaleNtdHeadStride];
        QuantQToFp8(tile, qkAfterCubeUb, qkScaleStaging);
        SetAndWaitQkToMte3Ready();
        StoreQOutputs(tile, qkAfterCubeUb, qkScaleStaging);
        QuantKToFp8(tile, qkAfterCubeUb, kScale);
        SetAndWaitQkToMte3Ready();
        ScatterKOutputs(tile, qkAfterCubeUb, kScale);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        ++vOutBufferUseId_;
    }

private:
    static constexpr uint32_t QK_PREPROCESS_SCATTER_INDEX_TABLE_ELEMENTS =
        QKV_K_SCALE_QK_NZ_SCATTER_INDEX_TABLE_ELEMENTS;
    static constexpr bool QKV_IS_TND = QKV_LAYOUT == QKV_K_SCALE_LAYOUT_TND;
    static constexpr bool Q_OUT_IS_TND = Q_OUT_LAYOUT == QKV_K_SCALE_LAYOUT_TND;

    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_READY = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_V_CACHE_READY = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_QK_READY = 1U;
    static constexpr uint32_t EVT_MTE2_TO_S_SLOT_MAPPING_READY = 5U;

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

    __aicore__ inline void CopyPersistentInputs()
    {
        DataCopyGmToUb2D(gammaUb_, qGammaGm_, 1U, headDim_, headDim_);
        DataCopyGmToUb2D(gammaUb_[QKV_K_SCALE_GAMMA_UB_ELEMENTS / 2U], kGammaGm_, 1U, headDim_, headDim_);
        DataCopyGmToUb2D(vScaleUb_, vScaleGm_, 1U, kvHeadNum_, kvHeadNum_);
        SetAndWaitMte2ToVInputReady();
    }

    __aicore__ inline void CopyPreprocessInputs(const TileParam &tile, uint32_t inputBufferId)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * QKV_K_SCALE_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[inputBufferId * QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        const uint64_t qkvHeadNum = qHeadNum_ + kvHeadNum_ + kvHeadNum_;
        if constexpr (QKV_IS_TND) {
            DataCopyGmToUb2D(input, qkvGm_[tile.aivTokenOffset * qkvHeadNum * headDim_], 1U,
                             tile.aivTokenSize * qkvHeadNum * headDim_, tile.aivTokenSize * qkvHeadNum * headDim_);
        } else {
            DataCopyGmToUb2D(input, qkvGm_[tile.aivTokenOffset * headDim_], qkvHeadNum, tile.aivTokenSize * headDim_,
                             totalTokens_ * headDim_);
        }
        DataCopyGmToUb2D(slotMapping, slotMappingGm_[tile.aivTokenOffset], 1U, tile.aivTokenSize, tile.aivTokenSize);
        CopyCosSinTile(tile, inputBufferId);
    }

    __aicore__ inline void BuildCacheOffsetsFromSlotMapping(TileParam &tile, uint32_t inputBufferId)
    {
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[inputBufferId * QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t slot = static_cast<uint64_t>(slotMapping.GetValue(tokenIdx));
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
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * QKV_K_SCALE_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> vScale = vScaleUb_;
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[vOutBufferId * QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS];
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
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[vOutBufferId * QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS];
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
            const uint64_t kScaleUbOffset = tokenIdx * kvHeadNum_ * QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS;
            DataCopyUbToGm2D(kScaleCacheOutGm_[kScaleOffset], kScale[kScaleUbOffset], kvHeadNum_, 1U, 1U,
                             kScaleCacheStrideHead_);
        }
    }

    __aicore__ inline void BindLocalTensors()
    {
        inputDbPoolUb_ = LocalTensor<bfloat16_t>(TPosition::LCM, QKV_K_SCALE_INPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_INPUT_DB_POOL_ELEMENTS);
        gammaUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_RESERVE_UB_OFFSET + QKV_K_SCALE_GAMMA_UB_OFFSET,
                                      QKV_K_SCALE_GAMMA_UB_ELEMENTS);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_RESERVE_UB_OFFSET + QKV_K_SCALE_V_SCALE_UB_OFFSET,
                                       QKV_K_SCALE_V_SCALE_UB_ELEMENTS);
        qkNzScatterIndexUb_ = LocalTensor<uint16_t>(
            TPosition::LCM, QKV_K_SCALE_RESERVE_UB_OFFSET + QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_OFFSET,
            QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_ELEMENTS);
        cosSinDbPoolUb_ =
            LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_RESERVE_UB_OFFSET + QKV_K_SCALE_COS_SIN_DB_POOL_OFFSET,
                               QKV_K_SCALE_COS_SIN_DB_POOL_ELEMENTS);
        slotMappingDbPoolUb_ = LocalTensor<int32_t>(
            TPosition::LCM, QKV_K_SCALE_RESERVE_UB_OFFSET + QKV_K_SCALE_SLOT_MAPPING_DB_POOL_OFFSET,
            QKV_K_SCALE_SLOT_MAPPING_DB_POOL_ELEMENTS);
        vOutDbPoolUb_ =
            LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, QKV_K_SCALE_RESERVE_UB_OFFSET + QKV_K_SCALE_V_OUT_DB_POOL_OFFSET,
                                      QKV_K_SCALE_V_OUT_DB_POOL_ELEMENTS);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        qkvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qkv);
        qGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.qGamma);
        kGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.kGamma);
        cosSinGm_.SetGlobalBuffer((__gm__ float *)tensors.cosSin);
        slotMappingGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.slotMapping);
        queryStartLocGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.queryStartLoc);
        seqLensGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.seqLens);
        vScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        qOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
        qScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.qScale);
        kCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
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
    float epsilon_;
    uint64_t inputBufferUseId_;
    uint64_t vOutBufferUseId_;
    uint64_t cosSinBatchIdx_;
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
    GlobalTensor<float> vScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> qOutGm_;
    GlobalTensor<float> qScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> kCacheOutGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheOutGm_;
    GlobalTensor<float> kScaleCacheOutGm_;
    LocalTensor<bfloat16_t> inputDbPoolUb_;
    LocalTensor<float> gammaUb_;
    LocalTensor<float> cosSinDbPoolUb_;
    LocalTensor<int32_t> slotMappingDbPoolUb_;
    LocalTensor<float> vScaleUb_;
    LocalTensor<uint16_t> qkNzScatterIndexUb_;
    LocalTensor<fp8_e4m3fn_t> vOutDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
