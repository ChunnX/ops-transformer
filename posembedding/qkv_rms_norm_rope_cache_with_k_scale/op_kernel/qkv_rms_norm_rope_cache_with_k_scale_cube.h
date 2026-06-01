/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"

namespace QkvRmsNormRopeCacheWithKScale {

class D128FastCube {
public:
    __aicore__ inline void Init(uint32_t cubeIndex, const GlobalTensors &tensors)
    {
        cubeIndex_ = cubeIndex;
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < AIC_L0_BUFFER_NUM; ++bufferId) {
            ReleaseL0Buffer(bufferId);
        }
        for (uint32_t bufferId = 0U; bufferId < AIC_L0C_BUFFER_NUM; ++bufferId) {
            ReleaseL0CBuffer(bufferId);
        }
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < AIC_L0_BUFFER_NUM; ++bufferId) {
            WaitL0BufferReusable(bufferId);
        }
        for (uint32_t bufferId = 0U; bufferId < AIC_L0C_BUFFER_NUM; ++bufferId) {
            WaitL0CBufferReusable(bufferId);
        }
    }

    __aicore__ inline void PrepareRotationBeforeLoop()
    {
        CopyRotationGmToL1Nz();
        NotifyRotationReady();
        WaitRotationReady();
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        LocalTensor<bfloat16_t> aRotL1Nz = aRotL1Pool_[bufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        WaitAReadyFromAiv();
        ConsumePreprocessedA(tile, aRotL1Nz);
        (void)workspaceGm_;
        (void)cubeIndex_;
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t AIC_L0_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM;
    static constexpr uint32_t AIC_M_L0_TILE = 128U;
    static constexpr uint32_t AIC_L0A_BUFFER_ELEMENTS = 64U * QKV_RRC_KIB / sizeof(bfloat16_t);
    static constexpr uint32_t AIC_L0B_BUFFER_ELEMENTS = 64U * QKV_RRC_KIB / sizeof(bfloat16_t);
    static constexpr uint32_t AIC_L0C_BUFFER_ELEMENTS = 256U * QKV_RRC_KIB / sizeof(float);
    static constexpr uint32_t AIC_L0A_ONE_BUFFER_ELEMENTS = AIC_L0A_BUFFER_ELEMENTS / AIC_L0_BUFFER_NUM;
    static constexpr uint32_t AIC_L0B_ONE_BUFFER_ELEMENTS = AIC_L0B_BUFFER_ELEMENTS / AIC_L0_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_ONE_BUFFER_ELEMENTS = AIC_L0C_BUFFER_ELEMENTS / AIC_L0C_BUFFER_NUM;
    static constexpr uint32_t EVT_MTE2_TO_MTE1_ROTATION_READY = 5U;
    static constexpr uint32_t EVT_MTE1_TO_MMAD_L0_BASE = 3U;
    static constexpr uint32_t EVT_MMAD_TO_MTE1_L0_BASE = 3U;
    static constexpr uint32_t EVT_MMAD_TO_FIXP_L0C_BASE = 0U;
    static constexpr uint32_t EVT_FIXP_TO_MMAD_L0C_BASE = 0U;

    static_assert(EVT_MTE2_TO_MTE1_ROTATION_READY <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC rotation MTE2->MTE1 event exceeds event id range.");
    static_assert(EVT_MTE1_TO_MMAD_L0_BASE + AIC_L0_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MTE1->MMAD L0 events exceed event id range.");
    static_assert(EVT_MMAD_TO_MTE1_L0_BASE + AIC_L0_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MMAD->MTE1 L0 events exceed event id range.");
    static_assert(EVT_MMAD_TO_FIXP_L0C_BASE + AIC_L0C_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MMAD->FIXP L0C events exceed event id range.");
    static_assert(EVT_FIXP_TO_MMAD_L0C_BASE + AIC_L0C_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC FIXP->MMAD L0C events exceed event id range.");

    __aicore__ inline void NotifyRotationReady()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
#endif
    }

    __aicore__ inline void WaitRotationReady()
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
#endif
    }

    __aicore__ inline void WaitL0BufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void ReleaseL0Buffer(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void NotifyL0ReadyForMmad(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitL0ReadyForMmad(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitL0CBufferReusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(EVT_FIXP_TO_MMAD_L0C_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void ReleaseL0CBuffer(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(EVT_FIXP_TO_MMAD_L0C_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void NotifyL0CReadyForFixpipe(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::M_FIX>(static_cast<event_t>(EVT_MMAD_TO_FIXP_L0C_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitL0CReadyForFixpipe(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(EVT_MMAD_TO_FIXP_L0C_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void CopyRotationGmToL1Nz()
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1U;
        nd2nzParams.nValue = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.dValue = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.srcNdMatrixStride = 0U;
        nd2nzParams.srcDValue = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.dstNzC0Stride = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.dstNzNStride = 1U;
        nd2nzParams.dstNzMatrixStride = 0U;
        DataCopy(rotationL1Nz_, rotationGm_, nd2nzParams);
    }

    __aicore__ inline void ConsumePreprocessedA(const TileParam &tile, LocalTensor<bfloat16_t> &aRotL1Nz)
    {
        if (tile.rowSize == 0U) {
            return;
        }

        for (uint32_t mStart = 0U; mStart < tile.rowSize; mStart += AIC_M_L0_TILE) {
            const uint32_t mSize = MinU32(AIC_M_L0_TILE, tile.rowSize - mStart);
            const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
            WaitL0BufferReusable(bufferId);
            WaitL0CBufferReusable(bufferId);
            LoadAToL0(tile, mStart, mSize, bufferId, aRotL1Nz);
            LoadRotationToL0(bufferId);
            NotifyL0ReadyForMmad(bufferId);
            WaitL0ReadyForMmad(bufferId);
            ComputeMmad(mSize, bufferId);
            NotifyL0CReadyForFixpipe(bufferId);
            WaitL0CReadyForFixpipe(bufferId);
            ReleaseL0CBuffer(bufferId);
            ReleaseL0Buffer(bufferId);
        }
    }

    __aicore__ inline void LoadAToL0(const TileParam &tile, uint32_t mStart, uint32_t mSize, uint32_t bufferId,
        LocalTensor<bfloat16_t> &aRotL1Nz)
    {
        const uint32_t rowSizeAligned = AlignUp(tile.rowSize, QKV_RRC_NZ_C0);
        const uint32_t mSizeAligned = AlignUp(mSize, QKV_RRC_NZ_C0);
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = 0U;
        loadDataParams.mStep = CeilDiv(mSize, QKV_RRC_NZ_C0);
        loadDataParams.kStep = QKV_RRC_HEAD_DIM_D128 / (QKV_RRC_BLOCK_BYTES / sizeof(bfloat16_t));
        loadDataParams.srcStride = CeilDiv(rowSizeAligned, QKV_RRC_NZ_C0);
        loadDataParams.dstStride = CeilDiv(mSizeAligned, QKV_RRC_NZ_C0);
        loadDataParams.ifTranspose = false;
        LoadData(l0a_[bufferId * AIC_L0A_ONE_BUFFER_ELEMENTS], aRotL1Nz[mStart * QKV_RRC_NZ_C0],
            loadDataParams);
    }

    __aicore__ inline void LoadRotationToL0(uint32_t bufferId)
    {
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = 0U;
        loadDataParams.mStep = CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0);
        loadDataParams.kStep = QKV_RRC_HEAD_DIM_D128 / (QKV_RRC_BLOCK_BYTES / sizeof(bfloat16_t));
        loadDataParams.srcStride = CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0);
        loadDataParams.dstStride = CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0);
        loadDataParams.ifTranspose = true;
        LoadData(l0b_[bufferId * AIC_L0B_ONE_BUFFER_ELEMENTS], rotationL1Nz_, loadDataParams);
    }

    __aicore__ inline void ComputeMmad(uint32_t mSize, uint32_t bufferId)
    {
        MmadParams params;
        params.m = AlignUp(mSize, QKV_RRC_NZ_C0);
        params.n = QKV_RRC_HEAD_DIM_D128;
        params.k = QKV_RRC_HEAD_DIM_D128;
        params.unitFlag = 0b11;
        params.cmatrixInitVal = true;
        params.cmatrixSource = false;
        Mmad(l0c_[bufferId * AIC_L0C_ONE_BUFFER_ELEMENTS],
            l0a_[bufferId * AIC_L0A_ONE_BUFFER_ELEMENTS],
            l0b_[bufferId * AIC_L0B_ONE_BUFFER_ELEMENTS],
            params);
    }

    __aicore__ inline void BindLocalTensors()
    {
        rotationL1Nz_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_L1_ELEMENTS);
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        l0a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, AIC_L0A_BUFFER_ELEMENTS);
        l0b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, AIC_L0B_BUFFER_ELEMENTS);
        l0c_ = LocalTensor<float>(TPosition::CO1, 0, AIC_L0C_BUFFER_ELEMENTS);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        rotationGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.rotation);
        workspaceGm_.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    uint32_t cubeIndex_;
    GlobalTensor<bfloat16_t> rotationGm_;
    GlobalTensor<uint8_t> workspaceGm_;
    LocalTensor<bfloat16_t> rotationL1Nz_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<bfloat16_t> l0a_;
    LocalTensor<bfloat16_t> l0b_;
    LocalTensor<float> l0c_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
