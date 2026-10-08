/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SFA_ARCH22_SPLIT_KV_MERGE_H
#define SFA_ARCH22_SPLIT_KV_MERGE_H

#include "kernel_operator.h"

namespace AttentionCommon {
using namespace AscendC;

constexpr uint32_t SPLIT_KV_HEAD_DIM = 512;
constexpr uint32_t SPLIT_KV_HEADS_PER_CHUNK = 16;

// Arch22 MLA decode contract: native G in [1, 128], D=512, at most 16 partitions.
template <typename OUT_T>
__aicore__ inline void MergeSparseSplitKV(uint32_t queryCount, uint32_t queryIdx, uint32_t splitCount, uint32_t heads,
                                          bool returnLse, GlobalTensor<OUT_T>& partialOut,
                                          GlobalTensor<float>& partialMax, GlobalTensor<float>& partialSum,
                                          GlobalTensor<OUT_T>& finalOut, GlobalTensor<float>& finalMax,
                                          GlobalTensor<float>& finalSum, LocalTensor<OUT_T> partial,
                                          LocalTensor<float> converted, LocalTensor<float> accum,
                                          LocalTensor<float> stats, LocalTensor<float> work, LocalTensor<OUT_T> result)
{
    constexpr uint32_t DIM = SPLIT_KV_HEAD_DIM;
    constexpr uint32_t HEADS_PER_CHUNK = SPLIT_KV_HEADS_PER_CHUNK;
    constexpr uint32_t SUM_ROWS = 16;
    constexpr uint32_t WEIGHT_ROWS = 32;
    const uint32_t statsStride = (heads + 7) / 8 * 8;
    const DataCopyExtParams statsCopy{1, static_cast<uint32_t>(heads * sizeof(float)), 0, 0, 0};
    const DataCopyPadExtParams<float> statsPad{false, 0, 0, 0};
    LocalTensor<float> max = work;
    LocalTensor<float> sum = work[statsStride];
    for (uint32_t split = 0; split < splitCount; ++split) {
        const uint64_t offset = (split * queryCount + queryIdx) * heads;
        DataCopyPad(stats[split * statsStride], partialMax[offset], statsCopy, statsPad);
        DataCopyPad(stats[(SUM_ROWS + split) * statsStride], partialSum[offset], statsCopy, statsPad);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    DataCopy(max, stats, statsStride);
    Duplicate(sum, 0.0f, heads);
    PipeBarrier<PIPE_V>();
    for (uint32_t split = 1; split < splitCount; ++split) {
        Max(max, max, stats[split * statsStride], heads);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t split = 0; split < splitCount; ++split) {
        LocalTensor<float> weight = stats[(WEIGHT_ROWS + split) * statsStride];
        Sub(weight, stats[split * statsStride], max, heads);
        PipeBarrier<PIPE_V>();
        Exp(weight, weight, heads);
        PipeBarrier<PIPE_V>();
        Mul(weight, weight, stats[(SUM_ROWS + split) * statsStride], heads);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, weight, heads);
        PipeBarrier<PIPE_V>();
    }
    if (returnLse && GetSubBlockIdx() == 0) {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(finalMax[queryIdx * heads], max, statsCopy);
        DataCopyPad(finalSum[queryIdx * heads], sum, statsCopy);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    for (uint32_t split = 0; split < splitCount; ++split) {
        LocalTensor<float> weight = stats[(WEIGHT_ROWS + split) * statsStride];
        Div(weight, weight, sum, heads);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    const uint32_t headsPerSubBlock = (heads + 1) / 2;
    const uint32_t firstHead = GetSubBlockIdx() * headsPerSubBlock;
    const uint32_t endHead = Min(heads, firstHead + headsPerSubBlock);
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
                const float weight = stats.GetValue((WEIGHT_ROWS + split) * statsStride + head + row);
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
