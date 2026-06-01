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
            ReleaseInputBuffer(bufferId);
            ReleaseVOutBuffer(bufferId);
            ReleaseQkOutBuffer(bufferId);
        }
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitInputBufferReusable(bufferId);
            WaitVOutBufferReusable(bufferId);
            WaitQkOutBufferReusable(bufferId);
        }
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        activeBufferId_ = MakeDataBufferId(tile.bufferId);
        PreprocessQkv(tile, slice);
        (void)QkAfterCubeUb();
        (void)QkFp8StagingUb();
    }

    __aicore__ inline void PostprocessV(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        activeBufferId_ = MakeDataBufferId(tile.bufferId);
        QuantAndScatterV(tile, slice);
        ReleaseInputBuffer(activeBufferId_);
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t QK_PREPROCESS_BF16_ND_BYTES = QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / 2U;
    static constexpr uint32_t QK_PREPROCESS_BF16_ND_ELEMENTS =
        QK_PREPROCESS_BF16_ND_BYTES / sizeof(bfloat16_t);
    static constexpr float QKV_RRC_HEAD_DIM_D128_RECIP = 1.0F / static_cast<float>(QKV_RRC_HEAD_DIM_D128);

    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_A_L1_READY = 4U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_A_L1_DONE = 4U;

    static_assert(QK_PREPROCESS_BF16_ND_ELEMENTS >= QKV_RRC_HEAD_DIM_D128 * QKV_RRC_HEAD_DIM_D128,
        "QkvRmsNormRopeCacheWithKScale Q/K BF16 ND staging buffer is too small.");
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

    __aicore__ inline void WaitInputBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void ReleaseInputBuffer(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void NotifyInputReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitInputReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitVOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void ReleaseVOutBuffer(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void NotifyVOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitVOutReady(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitQkOutBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void ReleaseQkOutBuffer(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitAl1StageDone()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_A_L1_DONE));
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EVT_MTE3_TO_PIPE_V_A_L1_DONE));
#endif
    }

    __aicore__ inline void WaitAl1StageReady()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_A_L1_READY));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVT_PIPE_V_TO_MTE3_A_L1_READY));
#endif
    }

    __aicore__ inline void PreprocessQkv(const TileParam &tile, const AivSlice &slice)
    {
        WaitInputBufferReusable(activeBufferId_);
        CopyPreprocessInputs(slice);
        NotifyInputReady(activeBufferId_);
        WaitInputReady(activeBufferId_);
        BuildRopeBf16Nd(tile, slice);
        WaitAl1StageReady();
        WriteRopeBf16NdToL1Nz(tile, slice);
        WaitAl1StageDone();
    }

    __aicore__ inline void CopyPreprocessInputs(const AivSlice &slice)
    {
        DataCopyGmToUb2D(InputUb(), qkvGm_[slice.tokenOffset * tilingData_->qkvStrideT],
            {slice.tokenSize, tilingData_->qkvStrideT, tilingData_->qkvStrideT});
        if (tilingData_->headDim > 0U) {
            DataCopyGmToUb2D(gammaUb_, qGammaGm_, {1U, tilingData_->headDim, tilingData_->headDim});
            DataCopyGmToUb2D(
                gammaUb_[tilingData_->headDim], kGammaGm_, {1U, tilingData_->headDim, tilingData_->headDim});
        }
        if (tilingData_->numVHeads > 0U) {
            DataCopyGmToUb2D(vScaleUb_, vScaleGm_, {1U, tilingData_->numVHeads, tilingData_->numVHeads});
        }
        if (slice.tokenSize > 0U) {
            DataCopyGmToUb2D(
                SlotMappingUb(), slotMappingGm_[slice.tokenOffset],
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
            DataCopyGmToUb2D(CosSinUb()[runStart * tilingData_->headDim],
                cosSinGm_[runPosition * tilingData_->headDim],
                {runSize, tilingData_->headDim, tilingData_->headDim});
            runStart += runSize;
        }
    }

    __aicore__ inline void BuildRopeBf16Nd(const TileParam &tile, const AivSlice &slice)
    {
        AscendC::VF_CALL<QkRmsNormRopeD128Vf>(
            (__ubuf__ bfloat16_t *)InputUb().GetPhyAddr(),
            (__ubuf__ float *)gammaUb_.GetPhyAddr(),
            (__ubuf__ float *)CosSinUb().GetPhyAddr(),
            (__ubuf__ bfloat16_t *)QkRopeBf16NdUb().GetPhyAddr(),
            slice.tokenSize, tile.qkHeadOffset, tile.qkHeadSize, tilingData_->qkvStrideT,
            tilingData_->numQHeads, QKV_RRC_HEAD_DIM_D128_RECIP, tilingData_->epsilon);
    }

    __aicore__ inline void WriteRopeBf16NdToL1Nz(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.rowSize == 0U) {
            return;
        }
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1U;
        nd2nzParams.nValue = slice.rowSize;
        nd2nzParams.dValue = tilingData_->headDim;
        nd2nzParams.srcNdMatrixStride = 0U;
        nd2nzParams.srcDValue = tilingData_->headDim;
        nd2nzParams.dstNzC0Stride = AlignUp(tile.rowSize, QKV_RRC_NZ_C0);
        nd2nzParams.dstNzNStride = 1U;
        nd2nzParams.dstNzMatrixStride = 0U;
        DataCopy(ARotL1Nz()[slice.rowOffset * QKV_RRC_NZ_C0], QkRopeBf16NdUb(), nd2nzParams);
    }

    __aicore__ inline void QuantAndScatterV(const TileParam &tile, const AivSlice &slice)
    {
        if (slice.tokenSize == 0U || tile.vHeadSize == 0U) {
            return;
        }

        WaitVOutBufferReusable(activeBufferId_);
        QuantVToFp8(tile, slice);
        NotifyVOutReady(activeBufferId_);
        WaitVOutReady(activeBufferId_);
        ScatterVCache(tile, slice);
        ReleaseVOutBuffer(activeBufferId_);
    }

    __aicore__ inline void QuantVToFp8(const TileParam &tile, const AivSlice &slice)
    {
        AscendC::VF_CALL<VScaleFp8D128Vf>(
            (__ubuf__ bfloat16_t *)InputUb().GetPhyAddr(),
            (__ubuf__ float *)vScaleUb_.GetPhyAddr(),
            (__ubuf__ fp8_e4m3fn_t *)VOutUb().GetPhyAddr(),
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

        const LocalTensor<fp8_e4m3fn_t> vOut = VOutUb();
        const LocalTensor<int32_t> slotMapping = SlotMappingUb();
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

    __aicore__ inline LocalTensor<bfloat16_t> ARotL1Nz()
    {
        return aRotL1Pool_[activeBufferId_ * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
    }

    __aicore__ inline LocalTensor<bfloat16_t> InputUb()
    {
        return inputDbPoolUb_[activeBufferId_ * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
    }

    __aicore__ inline LocalTensor<float> QkAfterCubeUb()
    {
        return outputDbPoolUb_[activeBufferId_ * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
    }

    __aicore__ inline LocalTensor<fp8_e4m3fn_t> QkFp8StagingUb()
    {
        return QkAfterCubeUb().template ReinterpretCast<fp8_e4m3fn_t>();
    }

    __aicore__ inline LocalTensor<bfloat16_t> QkRopeBf16NdUb()
    {
        return QkAfterCubeUb().template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline LocalTensor<float> CosSinUb()
    {
        return cosSinDbPoolUb_[activeBufferId_ * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
    }

    __aicore__ inline LocalTensor<int32_t> SlotMappingUb()
    {
        return slotMappingDbPoolUb_[activeBufferId_ * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
    }

    __aicore__ inline LocalTensor<fp8_e4m3fn_t> VOutUb()
    {
        return vOutDbPoolUb_[activeBufferId_ * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
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
    LocalTensor<fp8_e4m3fn_t> vOutDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
