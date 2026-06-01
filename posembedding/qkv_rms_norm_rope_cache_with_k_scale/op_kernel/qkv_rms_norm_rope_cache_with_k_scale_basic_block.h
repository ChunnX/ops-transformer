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

class D128FastBasicBlock {
public:
    __aicore__ inline void InitAic(const GlobalTensors &tensors, uint32_t cubeIndex)
    {
        isAiv_ = false;
        cube_.Init(cubeIndex, tensors);
        InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId)
    {
        isAiv_ = true;
        vec_.Init(tilingData, aivLocalId, tensors);
        InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile)
    {
        cube_.ComputeTile(tile);
    }

    __aicore__ inline void ComputeTileAiv(const TileParam &tile)
    {
        vec_.ComputeTile(tile);
    }

    __aicore__ inline void End()
    {
        EndIntraCoreEvents();
    }

private:
    __aicore__ inline void InitIntraCoreEvents()
    {
        if (isAiv_) {
            InitAivIntraCoreEvents();
        } else {
            InitAicIntraCoreEvents();
        }
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        if (isAiv_) {
            EndAivIntraCoreEvents();
        } else {
            EndAicIntraCoreEvents();
        }
    }

    __aicore__ inline void InitAivIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            SetIntraCoreEvent<HardEvent::V_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId);
            SetIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId);
            SetIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId);
        }
    }

    __aicore__ inline void EndAivIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_DOUBLE_BUFFER_NUM; ++bufferId) {
            WaitIntraCoreEvent<HardEvent::V_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufferId);
            WaitIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufferId);
            WaitIntraCoreEvent<HardEvent::MTE3_V>(EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + bufferId);
        }
    }

    __aicore__ inline void InitAicIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_AIC_L0_BUFFER_NUM; ++bufferId) {
            SetIntraCoreEvent<HardEvent::M_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId);
        }
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_AIC_L0C_BUFFER_NUM; ++bufferId) {
            SetIntraCoreEvent<HardEvent::FIX_M>(EVT_FIXP_TO_MMAD_L0C_BASE + bufferId);
        }
    }

    __aicore__ inline void EndAicIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_AIC_L0_BUFFER_NUM; ++bufferId) {
            WaitIntraCoreEvent<HardEvent::M_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId);
        }
        for (uint32_t bufferId = 0U; bufferId < QKV_RRC_AIC_L0C_BUFFER_NUM; ++bufferId) {
            WaitIntraCoreEvent<HardEvent::FIX_M>(EVT_FIXP_TO_MMAD_L0C_BASE + bufferId);
        }
    }

    bool isAiv_;
    D128FastCube cube_;
    D128FastVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
