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
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        LocalTensor<bfloat16_t> aRotL1Nz = aRotL1Pool_[bufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        LocalTensor<bfloat16_t> vUb = inputUb_[bufferId * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        LocalTensor<float> qkAfterCubeUb = outputUb_[bufferId * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        LocalTensor<fp8_e4m3fn_t> qkFp8StagingUb = qkAfterCubeUb.template ReinterpretCast<fp8_e4m3fn_t>();
        LocalTensor<float> cosSinUb = cosSinUb_[bufferId * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        LocalTensor<int32_t> slotMappingUb = slotMappingUb_[bufferId * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        LocalTensor<fp8_e4m3fn_t> vOutUb = vOutUb_[bufferId * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        PreprocessQkv(tile, slice, bufferId, vUb, aRotL1Nz, cosSinUb, slotMappingUb);
        (void)qkAfterCubeUb;
        (void)qkFp8StagingUb;
        (void)vScaleUb_;
        (void)vOutUb;
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2U;
    static constexpr uint32_t EVT_PIPE_V_TO_MTE3_A_L1_BASE = 4U;
    static constexpr uint32_t EVT_MTE3_TO_PIPE_V_A_L1_BASE = 4U;

    static_assert(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV MTE2->V input UB events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV V->MTE2 input UB events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE3_V_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV V->MTE3 V output events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV MTE3->V V output events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE3_QK_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV V->MTE3 QK output events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV MTE3->V QK output events exceed event id range.");
    static_assert(EVT_PIPE_V_TO_MTE3_A_L1_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV V->MTE3 AL1 events exceed event id range.");
    static_assert(EVT_MTE3_TO_PIPE_V_A_L1_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIV MTE3->V AL1 events exceed event id range.");

    __aicore__ inline void WaitInputBufferReusable(uint32_t bufferId)
    {
        WaitIntraCoreEvent<HardEvent::V_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId);
    }

    __aicore__ inline void ReleaseInputBuffer(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::V_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId);
    }

    __aicore__ inline void NotifyInputReady(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::MTE2_V>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId);
    }

    __aicore__ inline void WaitInputReady(uint32_t bufferId)
    {
        WaitIntraCoreEvent<HardEvent::MTE2_V>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufferId);
    }

    __aicore__ inline void WaitVOutBufferReusable(uint32_t bufferId)
    {
        WaitIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId);
    }

    __aicore__ inline void ReleaseVOutBuffer(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId);
    }

    __aicore__ inline void WaitQkOutBufferReusable(uint32_t bufferId)
    {
        WaitIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId);
    }

    __aicore__ inline void ReleaseQkOutBuffer(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId);
    }

    __aicore__ inline void WaitAl1StageDone(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_A_L1_BASE + bufferId);
        WaitIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_A_L1_BASE + bufferId);
    }

    __aicore__ inline void WaitAl1StageReady(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::V_MTE3>(EVT_PIPE_V_TO_MTE3_A_L1_BASE + bufferId);
        WaitIntraCoreEvent<HardEvent::V_MTE3>(EVT_PIPE_V_TO_MTE3_A_L1_BASE + bufferId);
    }

    __aicore__ inline void PreprocessQkv(const TileParam &tile, const AivSlice &slice, uint32_t bufferId,
        LocalTensor<bfloat16_t> &inputUb, LocalTensor<bfloat16_t> &aRotL1Nz, LocalTensor<float> &cosSinUb,
        LocalTensor<int32_t> &slotMappingUb)
    {
        WaitInputBufferReusable(bufferId);
        CopyPreprocessInputs(slice, inputUb, cosSinUb, slotMappingUb);
        NotifyInputReady(bufferId);
        WaitInputReady(bufferId);
        WaitAl1StageReady(bufferId);
        StageAForCube(tile, slice, inputUb, aRotL1Nz);
        WaitAl1StageDone(bufferId);
        ReleaseInputBuffer(bufferId);
        NotifyAReadyToAic();
    }

    __aicore__ inline void CopyPreprocessInputs(const AivSlice &slice, LocalTensor<bfloat16_t> &inputUb,
        LocalTensor<float> &cosSinUb, LocalTensor<int32_t> &slotMappingUb)
    {
        const uint32_t qkvElements = slice.tokenSize * tilingData_->qkvStrideT;
        if (qkvElements > 0U) {
            DataCopy(inputUb, globalTensors_.qkv[slice.tokenOffset * tilingData_->qkvStrideT], qkvElements);
        }
        if (tilingData_->headDim > 0U) {
            DataCopy(gammaUb_, globalTensors_.qGamma, tilingData_->headDim);
            DataCopy(gammaUb_[tilingData_->headDim], globalTensors_.kGamma, tilingData_->headDim);
        }
        if (slice.tokenSize > 0U) {
            DataCopy(slotMappingUb, globalTensors_.slotMapping[slice.tokenOffset], slice.tokenSize);
        }
        (void)cosSinUb;
    }

    __aicore__ inline void StageAForCube(const TileParam &tile, const AivSlice &slice,
        LocalTensor<bfloat16_t> &inputUb, LocalTensor<bfloat16_t> &aRotL1Nz)
    {
        const uint32_t qkElementsPerToken = tile.qkHeadSize * tilingData_->headDim;
        for (uint32_t tokenIdx = 0U; tokenIdx < slice.tokenSize; ++tokenIdx) {
            const uint32_t srcOffset = tokenIdx * tilingData_->qkvStrideT;
            const uint32_t dstOffset = (slice.rowOffset + tokenIdx * tile.qkHeadSize) * tilingData_->headDim;
            DataCopy(aRotL1Nz[dstOffset], inputUb[srcOffset], qkElementsPerToken);
        }
    }

    __aicore__ inline void BindLocalTensors()
    {
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        inputUb_ =
            LocalTensor<bfloat16_t>(TPosition::LCM, QKV_RRC_INPUT_DB_POOL_OFFSET, QKV_RRC_INPUT_DB_POOL_ELEMENTS);
        outputUb_ =
            LocalTensor<float>(TPosition::LCM, QKV_RRC_OUTPUT_DB_POOL_OFFSET, QKV_RRC_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        BindReserveUbTensors();
    }

    __aicore__ inline void BindReserveUbTensors()
    {
        gammaUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_GAMMA_UB_OFFSET,
            QKV_RRC_GAMMA_UB_ELEMENTS);
        cosSinUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_COS_SIN_DB_POOL_OFFSET,
            QKV_RRC_COS_SIN_DB_POOL_ELEMENTS);
        slotMappingUb_ = LocalTensor<int32_t>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET,
            QKV_RRC_SLOT_MAPPING_DB_POOL_ELEMENTS);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_SCALE_UB_OFFSET,
            QKV_RRC_V_SCALE_UB_ELEMENTS);
        vOutUb_ = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_OUT_DB_POOL_OFFSET,
            QKV_RRC_V_OUT_DB_POOL_ELEMENTS);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        globalTensors_.qkv.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qkv);
        globalTensors_.qGamma.SetGlobalBuffer((__gm__ float *)tensors.qGamma);
        globalTensors_.kGamma.SetGlobalBuffer((__gm__ float *)tensors.kGamma);
        globalTensors_.cosSin.SetGlobalBuffer((__gm__ float *)tensors.cosSin);
        globalTensors_.slotMapping.SetGlobalBuffer((__gm__ int32_t *)tensors.slotMapping);
        globalTensors_.kCacheIn.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCache);
        globalTensors_.vCacheIn.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCache);
        globalTensors_.kScaleCacheIn.SetGlobalBuffer((__gm__ float *)tensors.kScaleCache);
        globalTensors_.cuSeqLen.SetGlobalBuffer((__gm__ int32_t *)tensors.cuSeqLen);
        globalTensors_.vScale.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        globalTensors_.qOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
        globalTensors_.qScale.SetGlobalBuffer((__gm__ float *)tensors.qScale);
        globalTensors_.kCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
        globalTensors_.vCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCacheOut);
        globalTensors_.kScaleCacheOut.SetGlobalBuffer((__gm__ float *)tensors.kScaleCacheOut);
        globalTensors_.workspace.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
    uint32_t aivLocalId_;
    VecGlobalTensors globalTensors_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<bfloat16_t> inputUb_;
    LocalTensor<float> outputUb_;
    LocalTensor<float> gammaUb_;
    LocalTensor<float> cosSinUb_;
    LocalTensor<int32_t> slotMappingUb_;
    LocalTensor<float> vScaleUb_;
    LocalTensor<fp8_e4m3fn_t> vOutUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
