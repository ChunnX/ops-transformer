/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_HOST_TURBO_QUANT_ATTENTION_SCORE_TILING_H_
#define OPS_OP_HOST_TURBO_QUANT_ATTENTION_SCORE_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

// Tiling keys: mse_bits attribute (not dtype), keep binary count small.
constexpr uint64_t TQAS_TILING_KEY_MSE2 = 0;
constexpr uint64_t TQAS_TILING_KEY_MSE3 = 1;
constexpr uint64_t TQAS_TILING_KEY_MSE4 = 2;
// Paged KV cache block size (product constraint with vllm-ascend), not UB align size.
constexpr uint32_t TQAS_PAGED_BLOCK_SIZE = 128;

BEGIN_TILING_DATA_DEF(TurboQuantAttentionScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, numQHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, idxPackedBytes);
TILING_DATA_FIELD_DEF(uint32_t, qjlPackedBytes);
TILING_DATA_FIELD_DEF(uint32_t, maxBatch);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq);
TILING_DATA_FIELD_DEF(uint32_t, maxKvLen);
TILING_DATA_FIELD_DEF(uint32_t, kvTile);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, mseBits);
TILING_DATA_FIELD_DEF(uint32_t, groups);
TILING_DATA_FIELD_DEF(uint32_t, ubBlockSize);
TILING_DATA_FIELD_DEF(float, qjlCoeff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuantAttentionScore, TurboQuantAttentionScoreTilingData)

struct TurboQuantAttentionScoreCompileInfo {
    int64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
    uint32_t ubBlockSize = 0;
};

} // namespace optiling
#endif // OPS_OP_HOST_TURBO_QUANT_ATTENTION_SCORE_TILING_H_
