/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_OP_KERNEL_SCALED_COSINE_ATTENTION_SCORE_TILING_DEF_H_
#define OPS_OP_KERNEL_SCALED_COSINE_ATTENTION_SCORE_TILING_DEF_H_

#include "adv_api/kernel_tiling.h"

namespace optiling {
constexpr uint64_t SCAS_TILING_KEY_FP16 = 0;
constexpr uint64_t SCAS_TILING_KEY_BF16 = 1;
constexpr uint64_t SCAS_TILING_KEY_FP32 = 2;

// Device-side POD mirror. Field order and types must match the host TilingData.
struct ScaledCosineAttentionScoreTilingData {
    uint32_t batch;
    uint32_t heads;
    uint32_t seqLen;
    uint32_t headDim;
    uint32_t alignedHeadDim;
    uint32_t keyTileRows;
    uint32_t usedCoreNum;
    uint32_t reserved;
    uint64_t totalQueryRows;
    float clampMax;
    float eps;
};
}  // namespace optiling
#endif  // OPS_OP_KERNEL_SCALED_COSINE_ATTENTION_SCORE_TILING_DEF_H_

