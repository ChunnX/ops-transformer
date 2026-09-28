/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_KERNEL_TURBO_QUANT_ATTENTION_SCORE_TILING_DEF_H_
#define OPS_OP_KERNEL_TURBO_QUANT_ATTENTION_SCORE_TILING_DEF_H_

#include "adv_api/kernel_tiling.h"

namespace optiling {
constexpr uint64_t TQAS_TILING_KEY_MSE2 = 0;
constexpr uint64_t TQAS_TILING_KEY_MSE3 = 1;
constexpr uint64_t TQAS_TILING_KEY_MSE4 = 2;

// Device-side POD mirror. Field order and types must match the host TilingData.
struct TurboQuantAttentionScoreTilingData {
    uint32_t numTokens = 0;
    uint32_t numQHeads = 0;
    uint32_t numKvHeads = 0;
    uint32_t headDim = 0;
    uint32_t blockSize = 0;
    uint32_t numBlocks = 0;
    uint32_t idxPackedBytes = 0;
    uint32_t qjlPackedBytes = 0;
    uint32_t maxBatch = 0;
    uint32_t maxBlocksPerSeq = 0;
    uint32_t maxKvLen = 0;
    uint32_t kvTile = 0;
    uint32_t usedCoreNum = 0;
    uint32_t mseBits = 0;
    uint32_t groups = 0;
    uint32_t ubBlockSize = 0;
    float qjlCoeff = 0.0F;
};
} // namespace optiling
#endif // OPS_OP_KERNEL_TURBO_QUANT_ATTENTION_SCORE_TILING_DEF_H_
