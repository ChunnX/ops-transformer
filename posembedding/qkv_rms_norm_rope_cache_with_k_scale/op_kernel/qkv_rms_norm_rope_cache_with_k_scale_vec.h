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

class D128FastVec {
public:
    __aicore__ inline void Init(const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId,
        const GlobalTensors &tensors)
    {
        tilingData_ = tilingData;
        aivLocalId_ = aivLocalId;
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            SetVToMte2InputBufferReusable(bufferId);
            SetMte3ToVVOutBufferReusable(bufferId);
            SetMte3ToVQkOutBufferReusable(bufferId);
        }
        CopyPersistentInputs();
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitVToMte2InputBufferReusable(bufferId);
            WaitMte3ToVVOutBufferReusable(bufferId);
            WaitMte3ToVQkOutBufferReusable(bufferId);
        }
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        activeBufferId_ = MakeDataBufferId(tile.bufferId);
        PreprocessQkv(tile, slice);
    }

    __aicore__ inline void PostprocessV(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        activeBufferId_ = MakeDataBufferId(tile.bufferId);
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

    __aicore__ inline void PostprocessQk(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        activeBufferId_ = MakeDataBufferId(tile.bufferId);
        QuantAndScatterQk(tile, slice);
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t QK_PREPROCESS_BF16_NZ_BYTES = QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / 2U;
    static constexpr uint32_t QK_PREPROCESS_BF16_NZ_ELEMENTS =
        QK_PREPROCESS_BF16_NZ_BYTES / sizeof(bfloat16_t);
    static constexpr uint32_t QK_NZ_SCATTER_INDEX_LOAD_ELEMENTS =
        QKV_RRC_HEAD_DIM_D128 + QKV_RRC_HEAD_DIM_D128 / 2U;
    static constexpr float QKV_RRC_HEAD_DIM_D128_RECIP = 1.0F / static_cast<float>(QKV_RRC_HEAD_DIM_D128);

    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_A_L1_READY = 4U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_A_L1_DONE = 4U;

    static_assert(QK_PREPROCESS_BF16_NZ_ELEMENTS >= QKV_RRC_HEAD_DIM_D128 * QKV_RRC_HEAD_DIM_D128,
        "QkvRmsNormRopeCacheWithKScale Q/K BF16 NZ staging buffer is too small.");
    static_assert(QKV_RRC_QK_SCALE_UB_BYTES >= QK_NZ_SCATTER_INDEX_LOAD_ELEMENTS * sizeof(uint16_t),
        "QkvRmsNormRopeCacheWithKScale Q/K NZ scatter index load buffer is too small.");
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
    static_assert(EVT_MTE3_TO_PIPE_V_A_L1_DONE <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV MTE3->V AL1 done event exceeds event id range.");

    __aicore__ inline void WaitVToMte2InputBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetVToMte2InputBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetMte2ToVInputReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitMte2ToVInputReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitMte3ToVVOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetMte3ToVVOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetVToMte3VOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitVToMte3VOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitMte3ToVQkOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetMte3ToVQkOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetVToMte3QkOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitVToMte3QkOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetAndWaitMte3ToVAl1StageDone()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_A_L1_DONE));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_A_L1_DONE));
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
        DataCopyGmToUb2D(
            gamma[QKV_RRC_GAMMA_UB_ELEMENTS / 2U], kGammaGm_, {1U, tilingData_->headDim, tilingData_->headDim});
        if (tilingData_->numVHeads > 0U) {
            DataCopyGmToUb2D(vScale, vScaleGm_, {1U, tilingData_->numVHeads, tilingData_->numVHeads});
        }
        SetMte2ToVInputReady(activeBufferId_);
        WaitMte2ToVInputReady(activeBufferId_);
    }

    __aicore__ inline void PreprocessQkv(const TileParam &tile, const AivSlice &slice)
    {
        WaitMte3ToVQkOutBufferReusable(activeBufferId_);
        if (slice.tokenSize == 0U || slice.rowSize == 0U) {
            return;
        }

        CopyPreprocessInputs(slice);
        SetMte2ToVInputReady(activeBufferId_);
        WaitMte2ToVInputReady(activeBufferId_);
        BuildRopeBf16NzUb(tile, slice);
        SetAndWaitVToMte3Al1StageReady();
        CopyRopeBf16NzUbToL1Nz(tile, slice);
        SetAndWaitMte3ToVAl1StageDone();
    }

    __aicore__ inline void CopyPreprocessInputs(const AivSlice &slice)
    {
        const LocalTensor<bfloat16_t> input =
            inputDbPoolUb_[activeBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[activeBufferId_ * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        DataCopyGmToUb2D(input, qkvGm_[slice.tokenOffset * tilingData_->qkvStrideT],
            {slice.tokenSize, tilingData_->qkvStrideT, tilingData_->qkvStrideT});
        if (slice.tokenSize > 0U) {
            DataCopyGmToUb2D(
                slotMapping, slotMappingGm_[slice.tokenOffset],
                {1U, slice.tokenSize, slice.tokenSize});
            CopyCosSinTile(slice);
        }
    }

    __aicore__ inline uint32_t ResolveTokenPosition(uint32_t tokenOffset)
    {
        for (uint32_t batchIdx = 0U; batchIdx < tilingData_->batch; ++batchIdx) {
            const uint32_t seqBegin = static_cast<uint32_t>(cuSeqLenGm_.GetValue(batchIdx));
            const uint32_t seqEnd = static_cast<uint32_t>(cuSeqLenGm_.GetValue(batchIdx + 1U));
            if (tokenOffset >= seqBegin && tokenOffset < seqEnd) {
                return tokenOffset - seqBegin;
            }
        }
        return tokenOffset;
    }

    __aicore__ inline void CopyCosSinTile(const AivSlice &slice)
    {
        const LocalTensor<float> cosSin =
            cosSinDbPoolUb_[activeBufferId_ * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        uint32_t runStart = 0U;
        while (runStart < slice.tokenSize) {
            const uint32_t runPosition = ResolveTokenPosition(slice.tokenOffset + runStart);
            uint32_t runSize = 1U;
            while (runStart + runSize < slice.tokenSize) {
                const uint32_t prevPosition = ResolveTokenPosition(slice.tokenOffset + runStart + runSize - 1U);
                const uint32_t nextPosition = ResolveTokenPosition(slice.tokenOffset + runStart + runSize);
                if (nextPosition != prevPosition + 1U) {
                    break;
                }
                ++runSize;
            }
            DataCopyGmToUb2D(cosSin[runStart * tilingData_->headDim],
                cosSinGm_[runPosition * tilingData_->headDim],
                {runSize, tilingData_->headDim, tilingData_->headDim});
            runStart += runSize;
        }
    }

    __aicore__ inline void BuildRopeBf16NzUb(const TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<bfloat16_t> input =
            inputDbPoolUb_[activeBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> gamma = gammaUb_;
        const LocalTensor<float> cosSin =
            cosSinDbPoolUb_[activeBufferId_ * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qkRopeNz = outputDbPoolUb_[
            activeBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS].template ReinterpretCast<bfloat16_t>();
        const LocalTensor<uint16_t> nzScatterIndex = qkScaleStagingUb_.template ReinterpretCast<uint16_t>();
        BuildRopeNzScatterIndex(nzScatterIndex, AlignUp(slice.rowSize, QKV_RRC_NZ_C0));
        AscendC::VF_CALL<QkRmsNormRopeD128NzVf>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(),
            (__ubuf__ float *)gamma.GetPhyAddr(),
            (__ubuf__ float *)cosSin.GetPhyAddr(),
            (__ubuf__ bfloat16_t *)qkRopeNz.GetPhyAddr(),
            (__ubuf__ uint16_t *)nzScatterIndex.GetPhyAddr(),
            slice.tokenSize, tile.qkHeadOffset, tile.qkHeadSize, tilingData_->qkvStrideT,
            tilingData_->numQHeads, QKV_RRC_HEAD_DIM_D128_RECIP, tilingData_->epsilon);
    }

    __aicore__ inline void BuildRopeNzScatterIndex(const LocalTensor<uint16_t> &nzScatterIndex,
        uint32_t rowSizeAligned)
    {
        for (uint32_t dim = 0U; dim < QKV_RRC_HEAD_DIM_D128; ++dim) {
            const uint32_t dBlock = dim / QKV_RRC_NZ_C0;
            const uint32_t dInner = dim % QKV_RRC_NZ_C0;
            nzScatterIndex.SetValue(dim, static_cast<uint16_t>(dBlock * rowSizeAligned * QKV_RRC_NZ_C0 + dInner));
        }
    }

    __aicore__ inline void CopyRopeBf16NzUbToL1Nz(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.rowSize == 0U) {
            return;
        }
        const LocalTensor<bfloat16_t> aRotL1 =
            aRotL1Pool_[activeBufferId_ * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        const LocalTensor<bfloat16_t> qkRopeNz = outputDbPoolUb_[
            activeBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS].template ReinterpretCast<bfloat16_t>();
        DataCopyParams copyParams;
        copyParams.blockCount = QKV_RRC_HEAD_DIM_D128 / QKV_RRC_NZ_C0;
        copyParams.blockLen = slice.rowSize;
        copyParams.srcStride = AlignUp(slice.rowSize, QKV_RRC_NZ_C0) - slice.rowSize;
        copyParams.dstStride = AlignUp(tile.rowSize, QKV_RRC_NZ_C0) - slice.rowSize;
        DataCopy(aRotL1[slice.rowOffset * QKV_RRC_NZ_C0], qkRopeNz, copyParams);
    }

    __aicore__ inline void QuantAndScatterV(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.tokenSize == 0U || tile.vHeadSize == 0U) {
            return;
        }

        WaitMte3ToVVOutBufferReusable(activeBufferId_);
        QuantVToFp8(tile, slice);
        SetVToMte3VOutReady(activeBufferId_);
        WaitVToMte3VOutReady(activeBufferId_);
        ScatterVCache(tile, slice);
        SetMte3ToVVOutBufferReusable(activeBufferId_);
    }

    __aicore__ inline void QuantVToFp8(const TileParam &tile, const AivSlice &slice)
    {
        const LocalTensor<bfloat16_t> input =
            inputDbPoolUb_[activeBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<float> vScale = vScaleUb_;
        const LocalTensor<fp8_e4m3fn_t> vOut =
            vOutDbPoolUb_[activeBufferId_ * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        AscendC::VF_CALL<VScaleFp8D128Vf>(
            (__ubuf__ bfloat16_t *)input.GetPhyAddr(),
            (__ubuf__ float *)vScale.GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)vOut.GetPhyAddr(),
            slice.tokenSize, tile.vHeadOffset, tile.vHeadSize, tilingData_->qkvStrideT,
            tilingData_->numQHeads + tilingData_->numKHeads);
    }

    __aicore__ inline void ScatterVCache(const TileParam &tile, const AivSlice &slice)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1U;
        copyParams.blockLen = tilingData_->headDim * sizeof(fp8_e4m3fn_t);
        copyParams.srcStride = 0U;
        copyParams.dstStride = 0U;
        copyParams.rsv = 0U;

        const LocalTensor<fp8_e4m3fn_t> vOut =
            vOutDbPoolUb_[activeBufferId_ * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[activeBufferId_ * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        for (uint32_t tokenIdx = 0U; tokenIdx < slice.tokenSize; ++tokenIdx) {
            const uint32_t slot = static_cast<uint32_t>(slotMapping.GetValue(tokenIdx));
            const uint32_t blockId = slot / tilingData_->blockSize;
            const uint32_t blockOffset = slot % tilingData_->blockSize;
            for (uint32_t headIdx = 0U; headIdx < tile.vHeadSize; ++headIdx) {
                const uint32_t vHead = tile.vHeadOffset + headIdx;
                const uint32_t ubOffset = (tokenIdx * tile.vHeadSize + headIdx) * tilingData_->headDim;
                const uint32_t gmOffset = blockId * tilingData_->cacheStrideBlock +
                    vHead * tilingData_->cacheStrideHead + blockOffset * tilingData_->cacheStrideToken;
                DataCopyPad(vCacheOutGm_[gmOffset], vOut[ubOffset], copyParams);
            }
        }
    }

    __aicore__ inline void QuantAndScatterQk(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.rowSize == 0U) {
            SetMte3ToVQkOutBufferReusable(activeBufferId_);
            return;
        }

        QuantQkToFp8(slice);
        SetVToMte3QkOutReady(activeBufferId_);
        WaitVToMte3QkOutReady(activeBufferId_);
        ScatterQkOutputs(tile, slice);
        SetMte3ToVQkOutBufferReusable(activeBufferId_);
    }

    __aicore__ inline void QuantQkToFp8(const AivSlice &slice)
    {
        const LocalTensor<float> qkAfterCube =
            outputDbPoolUb_[activeBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        const LocalTensor<fp8_e4m3fn_t> qkFp8 = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        const LocalTensor<float> qkScale = qkScaleStagingUb_;
        AscendC::VF_CALL<QkDynamicQuantD128Vf>(
            (__ubuf__ float *)qkAfterCube.GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)qkFp8.GetPhyAddr(),
            (__ubuf__ float *)qkScale.GetPhyAddr(),
            slice.rowSize);
    }

    __aicore__ inline void ScatterQkOutputs(const TileParam &tile, const AivSlice &slice)
    {
        DataCopyExtParams qkCopyParams;
        qkCopyParams.blockCount = 1U;
        qkCopyParams.blockLen = tilingData_->headDim * sizeof(fp8_e4m3fn_t);
        qkCopyParams.srcStride = 0U;
        qkCopyParams.dstStride = 0U;
        qkCopyParams.rsv = 0U;

        DataCopyExtParams scaleCopyParams;
        scaleCopyParams.blockCount = 1U;
        scaleCopyParams.blockLen = sizeof(float);
        scaleCopyParams.srcStride = 0U;
        scaleCopyParams.dstStride = 0U;
        scaleCopyParams.rsv = 0U;

        const LocalTensor<float> qkAfterCube =
            outputDbPoolUb_[activeBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        const LocalTensor<fp8_e4m3fn_t> qkFp8 = qkAfterCube.template ReinterpretCast<fp8_e4m3fn_t>();
        const LocalTensor<float> qkScale = qkScaleStagingUb_;
        const LocalTensor<int32_t> slotMapping =
            slotMappingDbPoolUb_[activeBufferId_ * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        for (uint32_t tokenIdx = 0U; tokenIdx < slice.tokenSize; ++tokenIdx) {
            const uint32_t slot = static_cast<uint32_t>(slotMapping.GetValue(tokenIdx));
            const uint32_t blockId = slot / tilingData_->blockSize;
            const uint32_t blockOffset = slot % tilingData_->blockSize;
            for (uint32_t headIdx = 0U; headIdx < tile.qkHeadSize; ++headIdx) {
                const uint32_t rowIdx = tokenIdx * tile.qkHeadSize + headIdx;
                const uint32_t ubOffset = rowIdx * tilingData_->headDim;
                if (headIdx < tilingData_->numQHeads) {
                    const uint32_t qGmOffset =
                        (slice.tokenOffset + tokenIdx) * tilingData_->numQHeads * tilingData_->headDim +
                        headIdx * tilingData_->headDim;
                    const uint32_t qScaleOffset =
                        (slice.tokenOffset + tokenIdx) * tilingData_->numQHeads + headIdx;
                    DataCopyPad(qOutGm_[qGmOffset], qkFp8[ubOffset], qkCopyParams);
                    DataCopyPad(qScaleGm_[qScaleOffset], qkScale[rowIdx], scaleCopyParams);
                } else {
                    const uint32_t kHead = headIdx - tilingData_->numQHeads;
                    const uint32_t kCacheOffset = blockId * tilingData_->cacheStrideBlock +
                        kHead * tilingData_->cacheStrideHead + blockOffset * tilingData_->cacheStrideToken;
                    const uint32_t kScaleOffset = blockId * tilingData_->numKHeads * tilingData_->blockSize +
                        kHead * tilingData_->blockSize + blockOffset;
                    DataCopyPad(kCacheOutGm_[kCacheOffset], qkFp8[ubOffset], qkCopyParams);
                    DataCopyPad(kScaleCacheOutGm_[kScaleOffset], qkScale[rowIdx], scaleCopyParams);
                }
            }
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
        gammaUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_GAMMA_UB_OFFSET,
            QKV_RRC_GAMMA_UB_ELEMENTS);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_SCALE_UB_OFFSET,
            QKV_RRC_V_SCALE_UB_ELEMENTS);
        qkScaleStagingUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_QK_SCALE_UB_OFFSET,
            QKV_RRC_QK_SCALE_UB_ELEMENTS);
        cosSinDbPoolUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_COS_SIN_DB_POOL_OFFSET,
            QKV_RRC_COS_SIN_DB_POOL_ELEMENTS);
        slotMappingDbPoolUb_ = LocalTensor<int32_t>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET,
            QKV_RRC_SLOT_MAPPING_DB_POOL_ELEMENTS);
        vOutDbPoolUb_ = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_OUT_DB_POOL_OFFSET,
            QKV_RRC_V_OUT_DB_POOL_ELEMENTS);
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
    uint32_t activeBufferId_;
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
    LocalTensor<float> qkScaleStagingUb_;
    LocalTensor<fp8_e4m3fn_t> vOutDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
