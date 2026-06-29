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

template <uint32_t INPUT_LAYOUT>
class QkvRmsNormRopeCacheWithKScaleVec {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, uint64_t totalTokens, uint64_t batch,
                                uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim, uint64_t blockSize,
                                uint64_t kvCacheStrideBlock, uint64_t kvCacheStrideHead,
                                uint64_t kvCacheStrideToken, uint64_t kScaleCacheStrideBlock,
                                uint64_t kScaleCacheStrideHead, uint64_t kScaleCacheStrideToken, float epsilon)
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
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
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
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
        }
    }

    __aicore__ inline void ComputeTile(TileParam &tile, const LocalTensor<bfloat16_t> &aRotL1Nz,
                                       LocalTensor<float> qkPreprocessUb, uint32_t outputBufferId)
    {
        const uint32_t inputBufferId =
            static_cast<uint32_t>((inputBufferUseId_++) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        if (tile.aivTokenSize == 0U) {
            return;
        }

        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        CopyPreprocessInputs(tile, inputBufferId);
        SetAndWaitMte2ToSSlotMappingReady();
        BuildCacheOffsetsFromSlotMapping(tile, inputBufferId);
        SetAndWaitMte2ToVInputReady();
        BuildRopeBf16NzUb(tile, inputBufferId, qkPreprocessUb);
        SetAndWaitVToMte3Ready();
        CopyRopeBf16NzUbToL1Nz(tile, aRotL1Nz, qkPreprocessUb);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
    }

    __aicore__ inline void PostprocessV(const TileParam &tile)
    {
        if (tile.aivTokenSize == 0U || tile.vHeadSize == 0U) {
            return;
        }

        const uint32_t inputBufferId =
            static_cast<uint32_t>((inputBufferUseId_ - 1U) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        const uint32_t vOutBufferId =
            static_cast<uint32_t>((vOutBufferUseId_++) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        QuantVToFp8(tile, inputBufferId, vOutBufferId);
        SetAndWaitVToMte3Ready();
        ScatterVCache(tile, vOutBufferId);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
    }

    __aicore__ inline void PostprocessQk(const TileParam &tile, LocalTensor<float> qkAfterCubeUb,
                                         uint32_t outputBufferId)
    {
        if (tile.aivTokenSize == 0U) {
            return;
        }

        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
        const uint32_t vOutBufferId =
            static_cast<uint32_t>((vOutBufferUseId_++) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        const LocalTensor<float> qkScaleStaging =
            vOutDbPoolUb_[vOutBufferId * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS].template ReinterpretCast<float>();
        const uint64_t qScaleNtdHeadStride = AlignUp(tile.aivTokenSize, QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS);
        const LocalTensor<float> kScale = qkScaleStaging[qHeadNum_ * qScaleNtdHeadStride];
        QuantQToFp8(tile, qkAfterCubeUb, qkScaleStaging);
        SetAndWaitVToMte3Ready();
        ScatterQOutputs(tile, qkAfterCubeUb, qkScaleStaging);
        QuantKToFp8(tile, qkAfterCubeUb, kScale);
        SetAndWaitVToMte3Ready();
        ScatterKOutputs(tile, qkAfterCubeUb, kScale);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + vOutBufferId));
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId));
    }

    __aicore__ inline void WaitVToMte2InputBufferReusable()
    {
        const uint32_t inputBufferId = static_cast<uint32_t>(inputBufferUseId_ & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
    }

    __aicore__ inline void SetVToMte2InputBufferReusable()
    {
        const uint32_t inputBufferId =
            static_cast<uint32_t>((inputBufferUseId_ - 1U) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + inputBufferId));
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t QK_PREPROCESS_BF16_NZ_BYTES = QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / 2U;
    static constexpr uint32_t QK_PREPROCESS_BF16_NZ_ELEMENTS = QK_PREPROCESS_BF16_NZ_BYTES / sizeof(bfloat16_t);
    static constexpr uint32_t QK_PREPROCESS_UB_NZ_ROW_STRIDE = QKV_RRC_HEAD_DIM_D128;
    static constexpr uint32_t QK_NZ_SCATTER_INDEX_LOAD_ELEMENTS = QKV_RRC_HEAD_DIM_D128 + QKV_RRC_HEAD_DIM_D128 / 2U;
    static constexpr bool IS_TND = INPUT_LAYOUT == QKV_RRC_LAYOUT_TND;

    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_READY = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_READY = 0U;
    static constexpr uint32_t EVT_MTE2_TO_S_SLOT_MAPPING_READY = 5U;

    static_assert(QK_PREPROCESS_BF16_NZ_ELEMENTS >=
                      (QKV_RRC_HEAD_DIM_D128 / QKV_RRC_NZ_C0) * QK_PREPROCESS_UB_NZ_ROW_STRIDE * QKV_RRC_NZ_C0,
                  "QkvRmsNormRopeCacheWithKScale Q/K BF16 NZ staging buffer is too small.");
    static_assert(QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS >= QKV_RRC_MAX_TOKEN_TILE_PER_AIV,
                  "QkvRmsNormRopeCacheWithKScale slot mapping DB buffer is too small.");
    static_assert(QKV_RRC_QK_NZ_SCATTER_INDEX_UB_ELEMENTS >= QK_NZ_SCATTER_INDEX_LOAD_ELEMENTS,
                  "QkvRmsNormRopeCacheWithKScale Q/K NZ scatter index load buffer is too small.");
    static_assert(QKV_RRC_V_OUT_ONE_BUFFER_BYTES / sizeof(float) >= 2U * QKV_RRC_QK_SCALE_UB_ELEMENTS,
                  "QkvRmsNormRopeCacheWithKScale V output buffer is too small for Q/K scale staging.");
    static_assert(INPUT_LAYOUT == QKV_RRC_LAYOUT_NTD || INPUT_LAYOUT == QKV_RRC_LAYOUT_TND,
                  "QkvRmsNormRopeCacheWithKScale AIV only supports NTD/TND inputLayout.");
    static_assert(EVT_MTE2_TO_PIPE_V_INPUT_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE2->V input ready event exceeds event id range.");
    static_assert(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV V->MTE2 input UB events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE3->V V output events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE3->V QK output events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE3_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV V->MTE3 event exceeds event id range.");
    static_assert(EVT_MTE2_TO_S_SLOT_MAPPING_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE2->S slot mapping event exceeds event id range.");

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

    __aicore__ inline void SetAndWaitVToMte3Ready()
    {
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_READY));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_READY));
    }

    __aicore__ inline void CopyPersistentInputs()
    {
        DataCopyGmToUb2D(gammaUb_, qGammaGm_, {1U, headDim_, headDim_});
        DataCopyGmToUb2D(gammaUb_[QKV_RRC_GAMMA_UB_ELEMENTS / 2U], kGammaGm_, {1U, headDim_, headDim_});
        DataCopyGmToUb2D(vScaleUb_, vScaleGm_, {1U, kvHeadNum_, kvHeadNum_});
        SetAndWaitMte2ToVInputReady();
    }

    __aicore__ inline void CopyPreprocessInputs(const TileParam &tile, uint32_t inputBufferId)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[inputBufferId * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        const uint64_t qkvHeadNum = qHeadNum_ + kvHeadNum_ + kvHeadNum_;
        if constexpr (IS_TND) {
            DataCopyGmToUb2D(input, qkvGm_[tile.aivTokenOffset * qkvHeadNum * headDim_],
                             {1U, tile.aivTokenSize * qkvHeadNum * headDim_,
                              tile.aivTokenSize * qkvHeadNum * headDim_});
        } else {
            DataCopyGmToUb2D(input, qkvGm_[tile.aivTokenOffset * headDim_],
                             {qkvHeadNum, tile.aivTokenSize * headDim_,
                              totalTokens_ * headDim_});
        }
        DataCopyGmToUb2D(slotMapping, slotMappingGm_[tile.aivTokenOffset], {1U, tile.aivTokenSize, tile.aivTokenSize});
        CopyCosSinTile(tile, inputBufferId);
    }

    __aicore__ inline void BuildCacheOffsetsFromSlotMapping(TileParam &tile,
                                                            uint32_t inputBufferId)
    {
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[inputBufferId * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t slot = static_cast<uint64_t>(slotMapping.GetValue(tokenIdx));
            const uint64_t blockId = slot / blockSize_;
            const uint64_t blockOffset = slot % blockSize_;
            tile.cacheBaseOffset[tokenIdx] =
                blockId * kvCacheStrideBlock_ + blockOffset * kvCacheStrideToken_;
            tile.scaleCacheBaseOffset[tokenIdx] =
                blockId * kScaleCacheStrideBlock_ + blockOffset * kScaleCacheStrideToken_;
        }
    }

    __aicore__ inline uint64_t ResolveTokenPosition(uint64_t tokenOffset)
    {
        for (uint64_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
            const uint64_t seqBegin = static_cast<uint64_t>(cuSeqLenGm_.GetValue(batchIdx));
            const uint64_t seqEnd = static_cast<uint64_t>(cuSeqLenGm_.GetValue(batchIdx + 1U));
            if (tokenOffset >= seqBegin && tokenOffset < seqEnd) {
                const uint64_t localLen = seqEnd - seqBegin;
                const uint64_t actualLen = static_cast<uint64_t>(actualSeqLensGm_.GetValue(batchIdx));
                const uint64_t historyOffset = actualLen > localLen ? actualLen - localLen : 0U;
                return historyOffset + tokenOffset - seqBegin;
            }
        }
        return tokenOffset;
    }

    __aicore__ inline void CopyCosSinTile(const TileParam &tile, uint32_t inputBufferId)
    {
        const LocalTensor<float> cosSin = cosSinDbPoolUb_[inputBufferId * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        uint64_t runStart = 0U;
        while (runStart < tile.aivTokenSize) {
            const uint64_t runPosition = ResolveTokenPosition(tile.aivTokenOffset + runStart);
            uint64_t runSize = 1U;
            while (runStart + runSize < tile.aivTokenSize) {
                const uint64_t prevPosition = ResolveTokenPosition(tile.aivTokenOffset + runStart + runSize - 1U);
                const uint64_t nextPosition = ResolveTokenPosition(tile.aivTokenOffset + runStart + runSize);
                if (nextPosition != prevPosition + 1U) {
                    break;
                }
                ++runSize;
            }
            DataCopyGmToUb2D(cosSin[runStart * headDim_], cosSinGm_[runPosition * headDim_],
                             {runSize, headDim_, headDim_});
            runStart += runSize;
        }
    }

    __aicore__ inline void BuildRopeBf16NzUb(const TileParam &tile, uint32_t inputBufferId,
                                             LocalTensor<float> qkPreprocessUb)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> gamma = gammaUb_;
        const LocalTensor<float> cosSin = cosSinDbPoolUb_[inputBufferId * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qRopeNz = qkPreprocessUb.template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[QK_PREPROCESS_BF16_NZ_ELEMENTS];
        const uint32_t qkvHeadNum = static_cast<uint32_t>(qHeadNum_ + kvHeadNum_ + kvHeadNum_);
        const uint32_t tokenCapacity = static_cast<uint32_t>(tile.cubeHalfTokenSize);
        AscendC::VF_CALL<QkRmsNormRopeD128SegmentNzVfImpl<IS_TND>>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)gamma.GetPhyAddr(),
            (__ubuf__ float *)cosSin.GetPhyAddr(), (__ubuf__ bfloat16_t *)qRopeNz.GetPhyAddr(),
            (__ubuf__ uint16_t *)qkNzScatterIndexUb_.GetPhyAddr(), static_cast<uint32_t>(tile.aivTokenSize), 0U,
            static_cast<uint32_t>(qHeadNum_), qkvHeadNum, tokenCapacity, epsilon_);
        AscendC::VF_CALL<QkRmsNormRopeD128SegmentNzVfImpl<IS_TND>>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(),
            (__ubuf__ float *)(gamma[QKV_RRC_HEAD_DIM_D128].GetPhyAddr()), (__ubuf__ float *)cosSin.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)kRopeNz.GetPhyAddr(),
            (__ubuf__ uint16_t *)qkNzScatterIndexUb_.GetPhyAddr(), static_cast<uint32_t>(tile.aivTokenSize),
            static_cast<uint32_t>(qHeadNum_), static_cast<uint32_t>(kvHeadNum_), qkvHeadNum, tokenCapacity, epsilon_);
    }

    __aicore__ inline void BuildRopeNzScatterIndex()
    {
        for (uint32_t dim = 0U; dim < QKV_RRC_HEAD_DIM_D128; ++dim) {
            const uint32_t dBlock = dim / QKV_RRC_NZ_C0;
            const uint32_t dInner = dim % QKV_RRC_NZ_C0;
            qkNzScatterIndexUb_.SetValue(
                dim, static_cast<uint16_t>(dBlock * QK_PREPROCESS_UB_NZ_ROW_STRIDE * QKV_RRC_NZ_C0 + dInner));
        }
    }

    __aicore__ inline void CopyRopeBf16NzUbToL1Nz(const TileParam &tile,
                                                  const LocalTensor<bfloat16_t> &aRotL1,
                                                  LocalTensor<float> qkPreprocessUb)
    {
        if (tile.aivTokenSize == 0U) {
            return;
        }
        const uint64_t qTileRows = tile.cubeTokenSize * qHeadNum_;
        const uint64_t kTileRows = tile.cubeTokenSize * kvHeadNum_;
        const LocalTensor<bfloat16_t> qRopeNz = qkPreprocessUb.template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[QK_PREPROCESS_BF16_NZ_ELEMENTS];
        CopyRopeSegmentBf16LayoutNzUbToL1Nz(aRotL1, qRopeNz, qTileRows, qHeadNum_, tile.cubeHalfTokenSize,
                                            tile.aivBlockTokenOffset, tile.aivTokenSize);
        CopyRopeSegmentBf16LayoutNzUbToL1Nz(aRotL1[NzMatrixElements(qTileRows)], kRopeNz, kTileRows,
                                            kvHeadNum_, tile.cubeHalfTokenSize, tile.aivBlockTokenOffset,
                                            tile.aivTokenSize);
    }

    __aicore__ inline void CopyRopeSegmentBf16LayoutNzUbToL1Nz(const LocalTensor<bfloat16_t> &dstSegment,
                                                               const LocalTensor<bfloat16_t> &srcSegment,
                                                               uint64_t tileSegmentRows, uint64_t headSize,
                                                               uint64_t tokenCapacity, uint64_t aivBlockTokenOffset,
                                                               uint64_t tokenSize)
    {
        if (tokenSize == 0U || headSize == 0U) {
            return;
        }
        const uint64_t copyRows = headSize * tokenCapacity;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(QKV_RRC_HEAD_DIM_D128 / QKV_RRC_NZ_C0);
        copyParams.blockLen = static_cast<uint16_t>(copyRows);
        copyParams.srcStride = static_cast<uint16_t>(QK_PREPROCESS_UB_NZ_ROW_STRIDE - copyRows);
        copyParams.dstStride = static_cast<uint16_t>(NzRowAligned(tileSegmentRows) - copyRows);
        const uint64_t dstRowOffset = aivBlockTokenOffset * headSize;
        DataCopy(dstSegment[dstRowOffset * QKV_RRC_NZ_C0], srcSegment, copyParams);
    }

    __aicore__ inline void QuantVToFp8(const TileParam &tile, uint32_t inputBufferId,
                                       uint32_t vOutBufferId)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[inputBufferId * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> vScale = vScaleUb_;
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[vOutBufferId * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        const uint64_t vHeadOffset = qHeadNum_ + kvHeadNum_;
        const uint64_t qkvHeadNum = qHeadNum_ + kvHeadNum_ + kvHeadNum_;
        AscendC::VF_CALL<VScaleFp8D128ToNtdVfImpl<IS_TND>>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)vScale.GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)vOut.GetPhyAddr(), static_cast<uint32_t>(tile.aivTokenSize),
            static_cast<uint32_t>(tile.vHeadOffset), static_cast<uint32_t>(tile.vHeadSize),
            static_cast<uint32_t>(qkvHeadNum), static_cast<uint32_t>(vHeadOffset));
    }

    __aicore__ inline void ScatterVCache(const TileParam &tile, uint32_t vOutBufferId)
    {
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[vOutBufferId * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        for (uint64_t tokenIdx = 0U; tokenIdx < tile.aivTokenSize; ++tokenIdx) {
            const uint64_t ubOffset = tokenIdx * headDim_;
            const uint64_t gmOffset = tile.cacheBaseOffset[tokenIdx] + tile.vHeadOffset * kvCacheStrideHead_;
            DataCopyUbToGm2D(vCacheOutGm_[gmOffset], vOut[ubOffset],
                             {tile.vHeadSize, headDim_, tile.aivTokenSize * headDim_,
                              kvCacheStrideHead_});
        }
    }

    __aicore__ inline void QuantQToFp8(const TileParam &tile,
                                       LocalTensor<float> qkAfterCube, LocalTensor<float> qScale)
    {
        const LocalTensor<fp8_e4m3fn_t> qFp8 = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        AscendC::VF_CALL<QDynamicQuantD128VfImpl<IS_TND>>(
            (__ubuf__ float *)qkAfterCube.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)qFp8.GetPhyAddr(),
            (__ubuf__ float *)qScale.GetPhyAddr(), static_cast<uint32_t>(tile.aivTokenSize),
            static_cast<uint32_t>(qHeadNum_), static_cast<uint32_t>(tile.cubeHalfTokenSize));
    }

    __aicore__ inline void QuantKToFp8(const TileParam &tile,
                                       LocalTensor<float> qkAfterCube, LocalTensor<float> kScale)
    {
        const uint64_t kRowOffset = tile.cubeHalfTokenSize * qHeadNum_ * headDim_;
        const LocalTensor<fp8_e4m3fn_t> kFp8 = qkAfterCube[kRowOffset].template ReinterpretCast<fp8_e4m3fn_t>();
        AscendC::VF_CALL<KDynamicQuantD128VfImpl<IS_TND>>(
            (__ubuf__ float *)qkAfterCube[kRowOffset].GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)kFp8.GetPhyAddr(),
            (__ubuf__ float *)kScale.GetPhyAddr(), static_cast<uint32_t>(tile.aivTokenSize),
            static_cast<uint32_t>(kvHeadNum_), static_cast<uint32_t>(tile.cubeHalfTokenSize));
    }

    __aicore__ inline void ScatterQOutputs(const TileParam &tile, LocalTensor<float> qkAfterCube,
                                           LocalTensor<float> qScale)
    {
        const LocalTensor<fp8_e4m3fn_t> qFp8 = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        const uint64_t qFp8SparseRowStride = headDim_ * sizeof(float) / sizeof(fp8_e4m3fn_t);
        if constexpr (IS_TND) {
            DataCopyUbToGm2D(qOutGm_[tile.aivTokenOffset * qHeadNum_ * headDim_], qFp8,
                             {tile.aivTokenSize * qHeadNum_, headDim_, qFp8SparseRowStride, headDim_});
            DataCopyUbToGm2D(qScaleGm_[tile.aivTokenOffset * qHeadNum_], qScale,
                             {1U, tile.aivTokenSize * qHeadNum_,
                              tile.aivTokenSize * qHeadNum_, tile.aivTokenSize * qHeadNum_});
        } else {
            for (uint64_t qHeadIdx = 0U; qHeadIdx < qHeadNum_; ++qHeadIdx) {
                const uint64_t qUbOffset = qHeadIdx * tile.cubeHalfTokenSize * qFp8SparseRowStride;
                const uint64_t qGmOffset = qHeadIdx * totalTokens_ * headDim_ + tile.aivTokenOffset * headDim_;
                DataCopyUbToGm2D(qOutGm_[qGmOffset], qFp8[qUbOffset],
                                 {tile.aivTokenSize, headDim_, qFp8SparseRowStride, headDim_});
            }
            DataCopyUbToGm2D(qScaleGm_[tile.aivTokenOffset], qScale,
                             {qHeadNum_, tile.aivTokenSize, tile.aivTokenSize, totalTokens_});
        }
    }

    __aicore__ inline void ScatterKOutputs(const TileParam &tile,
                                           LocalTensor<float> qkAfterCube, LocalTensor<float> kScale)
    {
        const uint64_t kRowOffset = tile.cubeHalfTokenSize * qHeadNum_ * headDim_;
        const LocalTensor<fp8_e4m3fn_t> kFp8 = qkAfterCube[kRowOffset].template ReinterpretCast<fp8_e4m3fn_t>();
        const uint64_t kFp8SparseRowStride = headDim_ * sizeof(float) / sizeof(fp8_e4m3fn_t);
        uint64_t kUbTokenStride;
        uint64_t kUbHeadStride;
        if constexpr (IS_TND) {
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
            DataCopyUbToGm2D(kCacheOutGm_[kCacheOffset], kFp8[kUbOffset],
                             {kvHeadNum_, headDim_, kUbHeadStride, kvCacheStrideHead_});
            const uint64_t kScaleUbOffset = tokenIdx * kvHeadNum_ * QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS;
            DataCopyUbToGm2D(kScaleCacheOutGm_[kScaleOffset], kScale[kScaleUbOffset],
                             {kvHeadNum_, 1U, 1U, kScaleCacheStrideHead_});
        }
    }

    __aicore__ inline void BindLocalTensors()
    {
        inputDbPoolUb_ =
            LocalTensor<bfloat16_t>(TPosition::LCM, QKV_RRC_INPUT_DB_POOL_OFFSET, QKV_RRC_INPUT_DB_POOL_ELEMENTS);
        gammaUb_ = LocalTensor<float>(TPosition::LCM, QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_GAMMA_UB_OFFSET,
                                      QKV_RRC_GAMMA_UB_ELEMENTS);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM, QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_SCALE_UB_OFFSET,
                                       QKV_RRC_V_SCALE_UB_ELEMENTS);
        qkNzScatterIndexUb_ =
            LocalTensor<uint16_t>(TPosition::LCM, QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_QK_NZ_SCATTER_INDEX_UB_OFFSET,
                                  QKV_RRC_QK_NZ_SCATTER_INDEX_UB_ELEMENTS);
        cosSinDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_COS_SIN_DB_POOL_OFFSET,
                                             QKV_RRC_COS_SIN_DB_POOL_ELEMENTS);
        slotMappingDbPoolUb_ =
            LocalTensor<int32_t>(TPosition::LCM, QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET,
                                 QKV_RRC_SLOT_MAPPING_DB_POOL_ELEMENTS);
        vOutDbPoolUb_ = LocalTensor<fp8_e4m3fn_t>(
            TPosition::LCM, QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_OUT_DB_POOL_OFFSET, QKV_RRC_V_OUT_DB_POOL_ELEMENTS);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        qkvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qkv);
        qGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.qGamma);
        kGammaGm_.SetGlobalBuffer((__gm__ float *)tensors.kGamma);
        cosSinGm_.SetGlobalBuffer((__gm__ float *)tensors.cosSin);
        slotMappingGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.slotMapping);
        cuSeqLenGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.cuSeqLen);
        actualSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.actualSeqLens);
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
    GlobalTensor<bfloat16_t> qkvGm_;
    GlobalTensor<float> qGammaGm_;
    GlobalTensor<float> kGammaGm_;
    GlobalTensor<float> cosSinGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<int32_t> cuSeqLenGm_;
    GlobalTensor<int32_t> actualSeqLensGm_;
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
