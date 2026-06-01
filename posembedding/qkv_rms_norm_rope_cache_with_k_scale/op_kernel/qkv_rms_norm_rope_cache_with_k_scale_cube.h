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

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        LocalTensor<bfloat16_t> aRotL1Nz = aRotL1Pool_[bufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        WaitAReadyFromAiv();
        ConsumePreprocessedA(tile, aRotL1Nz);
        (void)globalTensors_;
        (void)rotationL1Nz_;
        (void)cubeIndex_;
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t AIC_L0_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM;
    static constexpr uint32_t EVT_MTE2_TO_MTE1_L1_BASE = 3U;
    static constexpr uint32_t EVT_MTE1_TO_MTE2_L1_BASE = 3U;
    static constexpr uint32_t EVT_MTE1_TO_MMAD_L0_BASE = 3U;
    static constexpr uint32_t EVT_MMAD_TO_MTE1_L0_BASE = 3U;
    static constexpr uint32_t EVT_MMAD_TO_FIXP_L0C_BASE = 0U;
    static constexpr uint32_t EVT_FIXP_TO_MMAD_L0C_BASE = 0U;

    static_assert(EVT_MTE2_TO_MTE1_L1_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MTE2->MTE1 L1 events exceed event id range.");
    static_assert(EVT_MTE1_TO_MTE2_L1_BASE + QKV_RRC_DOUBLE_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MTE1->MTE2 L1 events exceed event id range.");
    static_assert(EVT_MTE1_TO_MMAD_L0_BASE + AIC_L0_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MTE1->MMAD L0 events exceed event id range.");
    static_assert(EVT_MMAD_TO_MTE1_L0_BASE + AIC_L0_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MMAD->MTE1 L0 events exceed event id range.");
    static_assert(EVT_MMAD_TO_FIXP_L0C_BASE + AIC_L0C_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC MMAD->FIXP L0C events exceed event id range.");
    static_assert(EVT_FIXP_TO_MMAD_L0C_BASE + AIC_L0C_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
        "QkvRmsNormRopeCacheWithKScale AIC FIXP->MMAD L0C events exceed event id range.");

    __aicore__ inline void WaitL0BufferReusable(uint32_t bufferId)
    {
        WaitIntraCoreEvent<HardEvent::M_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId);
    }

    __aicore__ inline void ReleaseL0Buffer(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::M_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId);
    }

    __aicore__ inline void WaitL0CBufferReusable(uint32_t bufferId)
    {
        WaitIntraCoreEvent<HardEvent::FIX_M>(EVT_FIXP_TO_MMAD_L0C_BASE + bufferId);
    }

    __aicore__ inline void ReleaseL0CBuffer(uint32_t bufferId)
    {
        SetIntraCoreEvent<HardEvent::FIX_M>(EVT_FIXP_TO_MMAD_L0C_BASE + bufferId);
    }

    __aicore__ inline void ConsumePreprocessedA(const TileParam &tile, LocalTensor<bfloat16_t> &aRotL1Nz)
    {
        (void)tile;
        (void)aRotL1Nz;
    }

    __aicore__ inline void BindLocalTensors()
    {
        rotationL1Nz_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_L1_ELEMENTS);
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        globalTensors_.rotation.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.rotation);
        globalTensors_.workspace.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    uint32_t cubeIndex_;
    CubeGlobalTensors globalTensors_;
    LocalTensor<bfloat16_t> rotationL1Nz_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
