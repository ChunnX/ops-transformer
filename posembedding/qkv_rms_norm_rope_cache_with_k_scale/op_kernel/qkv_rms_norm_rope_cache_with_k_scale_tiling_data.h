/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_DATA_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_DATA_H_

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "kernel_tiling/kernel_tiling.h"

namespace QkvRmsNormRopeCacheWithKScaleKernelTiling {

#pragma pack(push, 8)
struct QkvRmsNormRopeCacheWithKScaleTilingData {
    uint32_t totalTokens = 0;
    uint32_t batch = 0;
    uint32_t numQHeads = 0;
    uint32_t numKHeads = 0;
    uint32_t numVHeads = 0;
    uint32_t headDim = 0;
    uint32_t maxSeqLen = 0;
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t coreTokenTile = 0;
    uint32_t coreGroupNum = 0;
    uint32_t qkvStrideT = 0;
    uint32_t qkvStrideH = 0;
    uint32_t cacheStrideBlock = 0;
    uint32_t cacheStrideHead = 0;
    uint32_t cacheStrideToken = 0;
    uint32_t tokenTile = 0;
    uint32_t tokenTilePerAiv = 0;
    uint32_t rowTile = 0;
    uint32_t rowTileAligned = 0;
    uint32_t qkHeadTile = 0;
    uint32_t vHeadTile = 0;
    uint32_t dimTile = 0;
    uint32_t flags = 0;
    float epsilon = 0.0F;
    uint32_t reserved = 0;
};
#pragma pack(pop)

} // namespace QkvRmsNormRopeCacheWithKScaleKernelTiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_DATA_H_
