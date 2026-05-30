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
    __aicore__ inline void Init(uint32_t cubeIndex)
    {
        cubeIndex_ = cubeIndex;
    }

    __aicore__ inline void ComputeTile(const TileParam &tile, const BoundGlobalTensors &globalTensors,
        const StaticLocalTensors &localTensors)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        (void)globalTensors;
        (void)localTensors.rotationL1Nz;
        (void)localTensors.aRotL1Nz[bufferId];
        (void)tile;
        (void)cubeIndex_;
    }

private:
    uint32_t cubeIndex_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
