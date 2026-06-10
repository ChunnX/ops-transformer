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

class QkvRmsNormRopeCacheWithKScaleVec {
public:
    __aicore__ inline void Init(const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId,
                                const GlobalTensors &tensors)
    {
        tilingData_ = tilingData;
        aivLocalId_ = aivLocalId;
        activeDataBufferId_ = 0U;
        activeOutputBufferId_ = 0U;
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            SetVToMte2InputBufferReusable(bufferId);
            SetMte3ToVVOutBufferReusable(bufferId);
            SetMte3ToVQkOutBufferReusableByBufferId(bufferId);
        }
        CopyPersistentInputs();
        BuildRopeNzScatterIndex(qkNzScatterIndexUb_);
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitVToMte2InputBufferReusable(bufferId);
            WaitMte3ToVVOutBufferReusable(bufferId);
            WaitMte3ToVQkOutBufferReusableByBufferId(bufferId);
        }
    }

    __aicore__ inline void ComputeTile(TileParam &tile, uint64_t outBufferUseId)
    {
        const AivSlice slice = MakeAivSlice(tile, QkHeadTile(), aivLocalId_);
        activeDataBufferId_ = MakeDataBufferId(tile.bufferId);
        activeOutputBufferId_ = MakeOutputBufferId(outBufferUseId);
        PreprocessQkv(tile, slice);
    }

    __aicore__ inline void PostprocessV(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, QkHeadTile(), aivLocalId_);
        activeDataBufferId_ = MakeDataBufferId(tile.bufferId);
        QuantAndScatterV(tile, slice);
    }

    __aicore__ inline void WaitVToMte2InputBufferReusable(const TileParam &tile)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        WaitVToMte2InputBufferReusable(bufferId);
    }

    __aicore__ inline void SetVToMte2InputBufferReusable(const TileParam &tile)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        SetVToMte2InputBufferReusable(bufferId);
    }

    __aicore__ inline void WaitMte3ToVQkOutBufferReusable(uint64_t outBufferUseId)
    {
        const uint32_t bufferId = MakeOutputBufferId(outBufferUseId);
        WaitMte3ToVQkOutBufferReusableByBufferId(bufferId);
    }

    __aicore__ inline void SetMte3ToVQkOutBufferReusable(uint64_t outBufferUseId)
    {
        const uint32_t bufferId = MakeOutputBufferId(outBufferUseId);
        SetMte3ToVQkOutBufferReusableByBufferId(bufferId);
    }

    __aicore__ inline void PostprocessQk(const TileParam &tile, uint64_t outBufferUseId)
    {
        const AivSlice slice = MakeAivSlice(tile, QkHeadTile(), aivLocalId_);
        activeDataBufferId_ = MakeDataBufferId(tile.bufferId);
        activeOutputBufferId_ = MakeOutputBufferId(outBufferUseId);
        QuantAndScatterQk(tile, slice);
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t QK_PREPROCESS_BF16_NZ_BYTES = QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / 2U;
    static constexpr uint32_t QK_PREPROCESS_BF16_NZ_ELEMENTS = QK_PREPROCESS_BF16_NZ_BYTES / sizeof(bfloat16_t);
    static constexpr uint32_t QK_PREPROCESS_UB_NZ_ROW_STRIDE = QKV_RRC_HEAD_DIM_D128;
    static constexpr uint32_t QK_NZ_SCATTER_INDEX_LOAD_ELEMENTS = QKV_RRC_HEAD_DIM_D128 + QKV_RRC_HEAD_DIM_D128 / 2U;
    static constexpr float QKV_RRC_HEAD_DIM_D128_RECIP = 1.0F / static_cast<float>(QKV_RRC_HEAD_DIM_D128);

    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_A_L1_READY = 4U;
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
    static_assert(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE2->V input UB events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV V->MTE2 input UB events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE3->V V output events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE3->V QK output events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE3_A_L1_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV V->MTE3 AL1 ready event exceeds event id range.");
    static_assert(EVT_MTE2_TO_S_SLOT_MAPPING_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIV MTE2->S slot mapping event exceeds event id range.");

    __aicore__ inline uint64_t QkHeadTile() const
    {
        return tilingData_->numQHeads + tilingData_->numKHeads;
    }

    __aicore__ inline void WaitVToMte2InputBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetVToMte2InputBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetMte2ToVInputReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void WaitMte2ToVInputReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetAndWaitMte2ToSSlotMappingReady()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(EVT_MTE2_TO_S_SLOT_MAPPING_READY));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(EVT_MTE2_TO_S_SLOT_MAPPING_READY));
#endif
    }

    __aicore__ inline void WaitMte3ToVVOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetMte3ToVVOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetVToMte3VOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void WaitVToMte3VOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void WaitMte3ToVQkOutBufferReusableByBufferId(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetMte3ToVQkOutBufferReusableByBufferId(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetVToMte3QkOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void WaitVToMte3QkOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#endif
    }

    __aicore__ inline void SetAndWaitVToMte3Al1StageReady()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_A_L1_READY));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_A_L1_READY));
#endif
    }

    __aicore__ inline void CopyPersistentInputs()
    {
        const LocalTensor<float> gamma = gammaUb_;
        const LocalTensor<float> vScale = vScaleUb_;
        DataCopyGmToUb2D(gamma, qGammaGm_, {1U, tilingData_->headDim, tilingData_->headDim});
        DataCopyGmToUb2D(gamma[QKV_RRC_GAMMA_UB_ELEMENTS / 2U], kGammaGm_,
                         {1U, tilingData_->headDim, tilingData_->headDim});
        if (tilingData_->numVHeads > 0U) {
            DataCopyGmToUb2D(vScale, vScaleGm_, {1U, tilingData_->numVHeads, tilingData_->numVHeads});
        }
        SetMte2ToVInputReady(activeDataBufferId_);
        WaitMte2ToVInputReady(activeDataBufferId_);
    }

    __aicore__ inline void PreprocessQkv(TileParam &tile, const AivSlice &slice)
    {
        if (slice.tokenSize == 0U || slice.rowSize == 0U) {
            return;
        }

        CopyPreprocessInputs(slice);
        SetAndWaitMte2ToSSlotMappingReady();
        BuildCacheOffsetsFromSlotMapping(tile, slice);
        SetMte2ToVInputReady(activeDataBufferId_);
        WaitMte2ToVInputReady(activeDataBufferId_);
        BuildRopeBf16NzUb(tile, slice);
        SetAndWaitVToMte3Al1StageReady();
        CopyRopeBf16NzUbToL1Nz(tile, slice);
    }

    __aicore__ inline void CopyPreprocessInputs(const AivSlice &slice)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[activeDataBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[activeDataBufferId_ * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        DataCopyGmToUb2D(input, qkvGm_[slice.tokenOffset * tilingData_->headDim],
                         {QkHeadTile() + tilingData_->numVHeads, slice.tokenSize * tilingData_->headDim,
                          tilingData_->totalTokens * tilingData_->headDim});
        if (slice.tokenSize > 0U) {
            DataCopyGmToUb2D(slotMapping, slotMappingGm_[slice.tokenOffset], {1U, slice.tokenSize, slice.tokenSize});
            CopyCosSinTile(slice);
        }
    }

    __aicore__ inline void BuildCacheOffsetsFromSlotMapping(TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[activeDataBufferId_ * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        for (uint64_t tokenIdx = 0U; tokenIdx < slice.tokenSize; ++tokenIdx) {
            const uint64_t slot = static_cast<uint64_t>(slotMapping.GetValue(tokenIdx));
            const uint64_t blockId = slot / tilingData_->blockSize;
            const uint64_t blockOffset = slot % tilingData_->blockSize;
            tile.cacheBaseOffset[tokenIdx] =
                blockId * tilingData_->kCacheStrideBlock + blockOffset * tilingData_->kCacheStrideToken;
            tile.scaleCacheBaseOffset[tokenIdx] =
                blockId * tilingData_->kScaleCacheStrideBlock + blockOffset * tilingData_->kScaleCacheStrideToken;
        }
    }

    __aicore__ inline uint64_t ResolveTokenPosition(uint64_t tokenOffset)
    {
        for (uint64_t batchIdx = 0U; batchIdx < tilingData_->batch; ++batchIdx) {
            const uint64_t seqBegin = static_cast<uint64_t>(cuSeqLenGm_.GetValue(batchIdx));
            const uint64_t seqEnd = static_cast<uint64_t>(cuSeqLenGm_.GetValue(batchIdx + 1U));
            if (tokenOffset >= seqBegin && tokenOffset < seqEnd) {
                return tokenOffset - seqBegin;
            }
        }
        return tokenOffset;
    }

    __aicore__ inline void CopyCosSinTile(const AivSlice &slice)
    {
        const LocalTensor<float> cosSin = cosSinDbPoolUb_[activeDataBufferId_ * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        uint64_t runStart = 0U;
        while (runStart < slice.tokenSize) {
            const uint64_t runPosition = ResolveTokenPosition(slice.tokenOffset + runStart);
            uint64_t runSize = 1U;
            while (runStart + runSize < slice.tokenSize) {
                const uint64_t prevPosition = ResolveTokenPosition(slice.tokenOffset + runStart + runSize - 1U);
                const uint64_t nextPosition = ResolveTokenPosition(slice.tokenOffset + runStart + runSize);
                if (nextPosition != prevPosition + 1U) {
                    break;
                }
                ++runSize;
            }
            DataCopyGmToUb2D(cosSin[runStart * tilingData_->headDim], cosSinGm_[runPosition * tilingData_->headDim],
                             {runSize, tilingData_->headDim, tilingData_->headDim});
            runStart += runSize;
        }
    }

    __aicore__ inline void BuildRopeBf16NzUb(const TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[activeDataBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> gamma = gammaUb_;
        const LocalTensor<float> cosSin = cosSinDbPoolUb_[activeDataBufferId_ * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qRopeNz =
            outputDbPoolUb_[activeOutputBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS]
                .template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[QK_PREPROCESS_BF16_NZ_ELEMENTS];
        const uint64_t qkvUbStrideN = slice.tokenSize * tilingData_->headDim;
        const uint64_t qkvUbStrideT = tilingData_->headDim;
        const uint64_t outTokenStride = MakeCubeHalfTokenSize(tile);
        AscendC::VF_CALL<QkRmsNormRopeD128SegmentNtdNzVf>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)gamma.GetPhyAddr(),
            (__ubuf__ float *)cosSin.GetPhyAddr(), (__ubuf__ bfloat16_t *)qRopeNz.GetPhyAddr(),
            (__ubuf__ uint16_t *)qkNzScatterIndexUb_.GetPhyAddr(), static_cast<uint32_t>(slice.tokenSize), 0U,
            static_cast<uint32_t>(tilingData_->numQHeads), static_cast<uint32_t>(qkvUbStrideN),
            static_cast<uint32_t>(qkvUbStrideT), static_cast<uint32_t>(outTokenStride),
            QKV_RRC_HEAD_DIM_D128_RECIP, tilingData_->epsilon);
        AscendC::VF_CALL<QkRmsNormRopeD128SegmentNtdNzVf>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)(gamma[QKV_RRC_HEAD_DIM_D128].GetPhyAddr()),
            (__ubuf__ float *)cosSin.GetPhyAddr(), (__ubuf__ bfloat16_t *)kRopeNz.GetPhyAddr(),
            (__ubuf__ uint16_t *)qkNzScatterIndexUb_.GetPhyAddr(), static_cast<uint32_t>(slice.tokenSize),
            static_cast<uint32_t>(tilingData_->numQHeads), static_cast<uint32_t>(tilingData_->numKHeads),
            static_cast<uint32_t>(qkvUbStrideN), static_cast<uint32_t>(qkvUbStrideT),
            static_cast<uint32_t>(outTokenStride), QKV_RRC_HEAD_DIM_D128_RECIP, tilingData_->epsilon);
    }

    __aicore__ inline void BuildRopeNzScatterIndex(const LocalTensor<uint16_t> &nzScatterIndex)
    {
        for (uint32_t dim = 0U; dim < QKV_RRC_HEAD_DIM_D128; ++dim) {
            const uint32_t dBlock = dim / QKV_RRC_NZ_C0;
            const uint32_t dInner = dim % QKV_RRC_NZ_C0;
            nzScatterIndex.SetValue(
                dim, static_cast<uint16_t>(dBlock * QK_PREPROCESS_UB_NZ_ROW_STRIDE * QKV_RRC_NZ_C0 + dInner));
        }
    }

    __aicore__ inline void CopyRopeBf16NzUbToL1Nz(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.rowSize == 0U) {
            return;
        }
        const uint64_t cubeTokenSize = MakeCubeTokenSize(tile);
        const uint64_t cubeHalfTokenSize = MakeCubeHalfTokenSize(tile);
        const uint64_t aivBlockTokenOffset = aivLocalId_ * cubeHalfTokenSize;
        const uint64_t qTileRows = cubeTokenSize * tilingData_->numQHeads;
        const uint64_t kTileRows = cubeTokenSize * tilingData_->numKHeads;
        const LocalTensor<bfloat16_t> aRotL1 =
            aRotL1Pool_[activeDataBufferId_ * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qRopeNz =
            outputDbPoolUb_[activeOutputBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS]
                .template ReinterpretCast<bfloat16_t>();
        const LocalTensor<bfloat16_t> kRopeNz = qRopeNz[QK_PREPROCESS_BF16_NZ_ELEMENTS];
        CopyRopeSegmentBf16NtdNzUbToL1Nz(aRotL1, qRopeNz, qTileRows, tilingData_->numQHeads, cubeHalfTokenSize,
                                         aivBlockTokenOffset, slice.tokenSize);
        CopyRopeSegmentBf16NtdNzUbToL1Nz(aRotL1[NzMatrixElements(qTileRows)], kRopeNz, kTileRows,
                                         tilingData_->numKHeads, cubeHalfTokenSize, aivBlockTokenOffset,
                                         slice.tokenSize);
    }

    __aicore__ inline void CopyRopeSegmentBf16NtdNzUbToL1Nz(const LocalTensor<bfloat16_t> &dstSegment,
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

    __aicore__ inline void QuantAndScatterV(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.tokenSize == 0U || tile.vHeadSize == 0U) {
            return;
        }

        WaitMte3ToVVOutBufferReusable(activeDataBufferId_);
        QuantVToFp8(tile, slice);
        SetVToMte3VOutReady(activeDataBufferId_);
        WaitVToMte3VOutReady(activeDataBufferId_);
        ScatterVCache(tile, slice);
        SetMte3ToVVOutBufferReusable(activeDataBufferId_);
    }

    __aicore__ inline void QuantVToFp8(const TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<bfloat16_t> input = inputDbPoolUb_[activeDataBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> vScale = vScaleUb_;
        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[activeDataBufferId_ * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        const uint64_t qkvUbStrideN = slice.tokenSize * tilingData_->headDim;
        const uint64_t qkvUbStrideT = tilingData_->headDim;
        AscendC::VF_CALL<VScaleFp8D128ToNtdVf>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(), (__ubuf__ float *)vScale.GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)vOut.GetPhyAddr(), static_cast<uint32_t>(slice.tokenSize),
            static_cast<uint32_t>(tile.vHeadOffset), static_cast<uint32_t>(tile.vHeadSize),
            static_cast<uint32_t>(qkvUbStrideN), static_cast<uint32_t>(qkvUbStrideT),
            static_cast<uint32_t>(QkHeadTile()));
    }

    __aicore__ inline void ScatterVCache(const TileParam &tile, const AivSlice &slice)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(tile.vHeadSize);
        copyParams.blockLen = tilingData_->headDim * sizeof(fp8_e4m3fn_t);
        copyParams.srcStride = static_cast<decltype(copyParams.srcStride)>((slice.tokenSize - 1U) *
            tilingData_->headDim * sizeof(fp8_e4m3fn_t) / QKV_RRC_BLOCK_BYTES);
        copyParams.dstStride = static_cast<decltype(copyParams.dstStride)>((tilingData_->vCacheStrideHead -
            tilingData_->headDim) * sizeof(fp8_e4m3fn_t));
        copyParams.rsv = 0U;

        const LocalTensor<fp8_e4m3fn_t> vOut = vOutDbPoolUb_[activeDataBufferId_ * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        for (uint64_t tokenIdx = 0U; tokenIdx < slice.tokenSize; ++tokenIdx) {
            const uint64_t ubOffset = tokenIdx * tilingData_->headDim;
            const uint64_t gmOffset = tile.cacheBaseOffset[tokenIdx] + tile.vHeadOffset * tilingData_->vCacheStrideHead;
            DataCopyPad(vCacheOutGm_[gmOffset], vOut[ubOffset], copyParams);
        }
    }

    __aicore__ inline void QuantAndScatterQk(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.rowSize == 0U) {
            return;
        }

        WaitMte3ToVVOutBufferReusable(activeDataBufferId_);
        QuantQToFp8Ntd(tile, slice);
        SetVToMte3QkOutReady(activeOutputBufferId_);
        WaitVToMte3QkOutReady(activeOutputBufferId_);
        ScatterQOutputs(slice);
        QuantKToFp8Ntd(tile, slice);
        SetVToMte3QkOutReady(activeOutputBufferId_);
        WaitVToMte3QkOutReady(activeOutputBufferId_);
        ScatterKOutputs(tile, slice);
        SetMte3ToVVOutBufferReusable(activeDataBufferId_);
    }

    __aicore__ inline uint64_t QScaleNtdHeadStride(const AivSlice &slice) const
    {
        return AlignUp(slice.tokenSize, QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS);
    }

    __aicore__ inline uint64_t KScaleStagingOffset(uint64_t tokenIdx, uint64_t kHeadIdx) const
    {
        return (tokenIdx * tilingData_->numKHeads + kHeadIdx) * QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS;
    }

    __aicore__ inline uint64_t KScaleStagingBaseOffset(const AivSlice &slice) const
    {
        return tilingData_->numQHeads * QScaleNtdHeadStride(slice);
    }

    __aicore__ inline uint64_t QkLocalKBaseRow(const TileParam &tile) const
    {
        return MakeCubeHalfTokenSize(tile) * tilingData_->numQHeads;
    }

    __aicore__ inline LocalTensor<float> QkScaleStagingDbUb()
    {
        return vOutDbPoolUb_[activeDataBufferId_ * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS]
            .template ReinterpretCast<float>();
    }

    __aicore__ inline void QuantQToFp8Ntd(const TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<float> qkAfterCube =
            outputDbPoolUb_[activeOutputBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        const LocalTensor<fp8_e4m3fn_t> qFp8Ntd = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        const LocalTensor<float> qkScale = QkScaleStagingDbUb();
        const uint64_t qkUbStrideN = MakeCubeHalfTokenSize(tile) * tilingData_->headDim;
        AscendC::VF_CALL<QDynamicQuantD128NtdToNtdVf>(
            (__ubuf__ float *)qkAfterCube.GetPhyAddr(), (__ubuf__ fp8_e4m3fn_t *)qFp8Ntd.GetPhyAddr(),
            (__ubuf__ float *)qkScale.GetPhyAddr(), static_cast<uint32_t>(slice.tokenSize),
            static_cast<uint32_t>(tilingData_->numQHeads), static_cast<uint32_t>(qkUbStrideN),
            static_cast<uint32_t>(tilingData_->headDim));
    }

    __aicore__ inline void QuantKToFp8Ntd(const TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<float> qkAfterCube =
            outputDbPoolUb_[activeOutputBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        const uint64_t kRowOffset = QkLocalKBaseRow(tile) * tilingData_->headDim;
        const LocalTensor<fp8_e4m3fn_t> kFp8Ntd =
            qkAfterCube[kRowOffset].template ReinterpretCast<fp8_e4m3fn_t>();
        const LocalTensor<float> qkScale = QkScaleStagingDbUb()[KScaleStagingBaseOffset(slice)];
        const uint64_t qkUbStrideN = MakeCubeHalfTokenSize(tile) * tilingData_->headDim;
        AscendC::VF_CALL<KDynamicQuantD128NtdToNtdVf>(
            (__ubuf__ float *)qkAfterCube[kRowOffset].GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)kFp8Ntd.GetPhyAddr(), (__ubuf__ float *)qkScale.GetPhyAddr(),
            static_cast<uint32_t>(slice.tokenSize), static_cast<uint32_t>(tilingData_->numKHeads),
            static_cast<uint32_t>(qkUbStrideN), static_cast<uint32_t>(tilingData_->headDim));
    }

    __aicore__ inline void ScatterQOutputs(const AivSlice &slice)
    {
        DataCopyExtParams qCopyParams;
        qCopyParams.blockCount = static_cast<uint16_t>(tilingData_->numQHeads);
        qCopyParams.blockLen =
            static_cast<uint32_t>(slice.tokenSize * tilingData_->headDim * sizeof(fp8_e4m3fn_t));
        qCopyParams.srcStride = 0U;
        qCopyParams.dstStride = static_cast<decltype(qCopyParams.dstStride)>(((tilingData_->totalTokens -
            slice.tokenSize) * tilingData_->headDim) * sizeof(fp8_e4m3fn_t));
        qCopyParams.rsv = 0U;

        DataCopyExtParams qScaleCopyParams;
        qScaleCopyParams.blockCount = static_cast<uint16_t>(tilingData_->numQHeads);
        qScaleCopyParams.blockLen = static_cast<uint32_t>(slice.tokenSize * sizeof(float));
        qScaleCopyParams.srcStride = 0U;
        qScaleCopyParams.dstStride =
            static_cast<decltype(qScaleCopyParams.dstStride)>((tilingData_->totalTokens - slice.tokenSize) *
                sizeof(float));
        qScaleCopyParams.rsv = 0U;

        const LocalTensor<float> qkAfterCube =
            outputDbPoolUb_[activeOutputBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        const LocalTensor<fp8_e4m3fn_t> qFp8Ntd = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        const LocalTensor<float> qkScale = QkScaleStagingDbUb();
        DataCopyPad(qOutGm_[slice.tokenOffset * tilingData_->headDim], qFp8Ntd, qCopyParams);
        DataCopyPad(qScaleGm_[slice.tokenOffset], qkScale, qScaleCopyParams);
    }

    __aicore__ inline void ScatterKOutputs(const TileParam &tile, const AivSlice &slice)
    {
        DataCopyExtParams kCopyParams;
        kCopyParams.blockCount = static_cast<uint16_t>(tilingData_->numKHeads);
        kCopyParams.blockLen = static_cast<uint32_t>(tilingData_->headDim * sizeof(fp8_e4m3fn_t));
        kCopyParams.srcStride = static_cast<decltype(kCopyParams.srcStride)>((slice.tokenSize - 1U) *
            tilingData_->headDim * sizeof(fp8_e4m3fn_t) / QKV_RRC_BLOCK_BYTES);
        kCopyParams.dstStride = static_cast<decltype(kCopyParams.dstStride)>((tilingData_->kCacheStrideHead -
            tilingData_->headDim) * sizeof(fp8_e4m3fn_t));
        kCopyParams.rsv = 0U;

        DataCopyExtParams kScaleCopyParams;
        kScaleCopyParams.blockCount = static_cast<uint16_t>(tilingData_->numKHeads);
        kScaleCopyParams.blockLen = static_cast<uint32_t>(sizeof(float));
        kScaleCopyParams.srcStride = 0U;
        kScaleCopyParams.dstStride = static_cast<decltype(kScaleCopyParams.dstStride)>((tilingData_->kScaleCacheStrideHead -
            1U) * sizeof(float));
        kScaleCopyParams.rsv = 0U;

        const LocalTensor<float> qkAfterCube =
            outputDbPoolUb_[activeOutputBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        const uint64_t kRowOffset = QkLocalKBaseRow(tile) * tilingData_->headDim;
        const LocalTensor<fp8_e4m3fn_t> kFp8Ntd =
            qkAfterCube[kRowOffset].template ReinterpretCast<fp8_e4m3fn_t>();
        const LocalTensor<float> qkScale = QkScaleStagingDbUb()[KScaleStagingBaseOffset(slice)];
        for (uint64_t tokenIdx = 0U; tokenIdx < slice.tokenSize; ++tokenIdx) {
            const uint64_t kUbOffset = tokenIdx * tilingData_->headDim;
            const uint64_t kCacheOffset = tile.cacheBaseOffset[tokenIdx];
            const uint64_t kScaleOffset = tile.scaleCacheBaseOffset[tokenIdx];
            DataCopyPad(kCacheOutGm_[kCacheOffset], kFp8Ntd[kUbOffset], kCopyParams);
            DataCopyPad(kScaleCacheOutGm_[kScaleOffset], qkScale[KScaleStagingOffset(tokenIdx, 0U)],
                        kScaleCopyParams);
        }
    }

    __aicore__ inline void BindLocalTensors()
    {
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        inputDbPoolUb_ =
            LocalTensor<bfloat16_t>(TPosition::LCM, QKV_RRC_INPUT_DB_POOL_OFFSET, QKV_RRC_INPUT_DB_POOL_ELEMENTS);
        outputDbPoolUb_ =
            LocalTensor<float>(TPosition::LCM, QKV_RRC_OUTPUT_DB_POOL_OFFSET, QKV_RRC_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        BindReserveUbTensors();
    }

    __aicore__ inline void BindReserveUbTensors()
    {
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
        kCacheInGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCache);
        vCacheInGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCache);
        kScaleCacheInGm_.SetGlobalBuffer((__gm__ float *)tensors.kScaleCache);
        cuSeqLenGm_.SetGlobalBuffer((__gm__ int32_t *)tensors.cuSeqLen);
        vScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        qOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
        qScaleGm_.SetGlobalBuffer((__gm__ float *)tensors.qScale);
        kCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
        vCacheOutGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCacheOut);
        kScaleCacheOutGm_.SetGlobalBuffer((__gm__ float *)tensors.kScaleCacheOut);
        workspaceGm_.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
    uint32_t aivLocalId_;
    uint32_t activeDataBufferId_;
    uint32_t activeOutputBufferId_;
    GlobalTensor<bfloat16_t> qkvGm_;
    GlobalTensor<float> qGammaGm_;
    GlobalTensor<float> kGammaGm_;
    GlobalTensor<float> cosSinGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<fp8_e4m3fn_t> kCacheInGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheInGm_;
    GlobalTensor<float> kScaleCacheInGm_;
    GlobalTensor<int32_t> cuSeqLenGm_;
    GlobalTensor<float> vScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> qOutGm_;
    GlobalTensor<float> qScaleGm_;
    GlobalTensor<fp8_e4m3fn_t> kCacheOutGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheOutGm_;
    GlobalTensor<float> kScaleCacheOutGm_;
    GlobalTensor<uint8_t> workspaceGm_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<bfloat16_t> inputDbPoolUb_;
    LocalTensor<float> outputDbPoolUb_;
    LocalTensor<float> gammaUb_;
    LocalTensor<float> cosSinDbPoolUb_;
    LocalTensor<int32_t> slotMappingDbPoolUb_;
    LocalTensor<float> vScaleUb_;
    LocalTensor<uint16_t> qkNzScatterIndexUb_;
    LocalTensor<fp8_e4m3fn_t> vOutDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
