/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPARSE_SPLIT_KV_MERGE_H
#define SPARSE_SPLIT_KV_MERGE_H

#include "kernel_operator.h"

namespace AttentionCommon {
using namespace AscendC;

// Arch22 MLA decode contract: G=8 or 64, D=512, at most 16 partitions.
template <typename OUT_T>
__aicore__ inline void MergeSparseSplitKV(
    uint32_t queryCount, uint32_t queryIdx, uint32_t splitCount, uint32_t heads, bool returnLse,
    GlobalTensor<OUT_T> &partialOut, GlobalTensor<float> &partialMax, GlobalTensor<float> &partialSum,
    GlobalTensor<OUT_T> &finalOut, GlobalTensor<float> &finalMax, GlobalTensor<float> &finalSum,
    LocalTensor<OUT_T> partial, LocalTensor<float> converted, LocalTensor<float> accum,
    LocalTensor<float> stats, LocalTensor<float> work, LocalTensor<OUT_T> result)
{
    constexpr uint32_t DIM = 512;
    constexpr uint32_t HEADS_PER_CHUNK = 16; // 32 KiB float accumulation buffer.
    constexpr uint32_t SUM_ROWS = 16;
    constexpr uint32_t WEIGHT_ROWS = 32;
    LocalTensor<float> max = work;
    LocalTensor<float> sum = work[heads];
    for (uint32_t split = 0; split < splitCount; ++split) {
        const uint64_t offset = (split * queryCount + queryIdx) * heads;
        DataCopy(stats[split * heads], partialMax[offset], heads);
        DataCopy(stats[(SUM_ROWS + split) * heads], partialSum[offset], heads);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    DataCopy(max, stats, heads);
    Duplicate(sum, 0.0f, heads);
    PipeBarrier<PIPE_V>();
    for (uint32_t split = 1; split < splitCount; ++split) {
        Max(max, max, stats[split * heads], heads);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t split = 0; split < splitCount; ++split) {
        LocalTensor<float> weight = stats[(WEIGHT_ROWS + split) * heads];
        Sub(weight, stats[split * heads], max, heads);
        PipeBarrier<PIPE_V>();
        Exp(weight, weight, heads);
        PipeBarrier<PIPE_V>();
        Mul(weight, weight, stats[(SUM_ROWS + split) * heads], heads);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, weight, heads);
        PipeBarrier<PIPE_V>();
    }
    if (returnLse && GetSubBlockIdx() == 0) {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(finalMax[queryIdx * heads], max, heads);
        DataCopy(finalSum[queryIdx * heads], sum, heads);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    for (uint32_t split = 0; split < splitCount; ++split) {
        LocalTensor<float> weight = stats[(WEIGHT_ROWS + split) * heads];
        Div(weight, weight, sum, heads);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    const uint32_t firstHead = GetSubBlockIdx() * heads / 2;
    const uint32_t endHead = firstHead + heads / 2;
    for (uint32_t head = firstHead; head < endHead; head += HEADS_PER_CHUNK) {
        const uint32_t chunkHeads = Min(HEADS_PER_CHUNK, endHead - head);
        const uint32_t elements = chunkHeads * DIM;
        Duplicate(accum, 0.0f, elements);
        PipeBarrier<PIPE_V>();
        for (uint32_t split = 0; split < splitCount; ++split) {
            const uint64_t offset = ((split * queryCount + queryIdx) * heads + head) * DIM;
            DataCopy(partial, partialOut[offset], elements);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Cast(converted, partial, RoundMode::CAST_NONE, elements);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < chunkHeads; ++row) {
                const float weight = stats.GetValue((WEIGHT_ROWS + split) * heads + head + row);
                Muls(converted[row * DIM], converted[row * DIM], weight, DIM);
            }
            PipeBarrier<PIPE_V>();
            Add(accum, accum, converted, elements);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) {
            Cast(result, accum, RoundMode::CAST_RINT, elements);
        } else {
            Cast(result, accum, RoundMode::CAST_ROUND, elements);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(finalOut[(queryIdx * heads + head) * DIM], result, elements);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}
} // namespace AttentionCommon
#endif
