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

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        LocalTensor<bfloat16_t> aRotL1Nz = aRotL1Pool_[bufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        (void)globalTensors_;
        (void)rotationL1Nz_;
        (void)aRotL1Nz;
        (void)tile;
        (void)cubeIndex_;
    }

private:
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
