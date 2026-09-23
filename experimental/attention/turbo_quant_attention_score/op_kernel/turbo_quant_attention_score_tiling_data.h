/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_attention_score_tiling_data.h
 * \brief Tiling data shared between host-side tiling and device-side kernel (plain POD, layout must match).
 */

#ifndef TURBO_QUANT_ATTENTION_SCORE_TILING_DATA_H
#define TURBO_QUANT_ATTENTION_SCORE_TILING_DATA_H

#include <cstdint>

namespace TurboQuantAttentionScore {

struct TurboQuantAttentionScoreTilingData {
    int64_t numTokens = 0;        // num_q_tokens
    int64_t totalItems = 0;       // numTokens * numKvHeads * maxBlocksPerSeq
    uint32_t numQHeads = 0;       // 32
    uint32_t numKvHeads = 0;      // 8
    uint32_t groupSize = 0;       // numQHeads / numKvHeads (<= 8)
    uint32_t batch = 0;           // block_table.shape[0]
    uint32_t maxBlocksPerSeq = 0; // block_table.shape[1]
    uint32_t maxKvLen = 0;        // maxBlocksPerSeq * 128
    uint32_t usedCoreNum = 0;     // 实际启用核数
    uint32_t formerCoreNum = 0;   // 前 formerCoreNum 个核多处理 1 个 work item
    uint32_t itemsPerCore = 0;    // 每核基础 work item 数
    uint32_t mseBits = 3;         // 主量化位宽 2/3/4（冗余存一份，tilingkey 为主）
};

} // namespace TurboQuantAttentionScore

#endif // TURBO_QUANT_ATTENTION_SCORE_TILING_DATA_H
