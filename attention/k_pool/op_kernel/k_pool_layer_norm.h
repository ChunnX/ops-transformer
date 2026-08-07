/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef K_POOL_LAYER_NORM_H
#define K_POOL_LAYER_NORM_H

#include "k_pool_tools.h"

namespace KPool {

__aicore__ inline void KPoolLayerNorm(const LocalTensor<float> &out, const LocalTensor<float> &in,
                                      const LocalTensor<float> &mean, const LocalTensor<float> &gamma,
                                      const LocalTensor<float> &beta, float eps, uint32_t count)
{
    const float reciprocal = 1.0f / static_cast<float>(count);

    Muls(mean, in, reciprocal, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(mean, mean, mean, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float meanValue = mean.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(mean, meanValue, count);
    PipeBarrier<PIPE_V>();

    Sub(in, in, mean, count);
    PipeBarrier<PIPE_V>();
    Mul(out, in, in, count);
    PipeBarrier<PIPE_V>();
    Muls(out, out, reciprocal, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(out, out, out, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float variance = out.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, variance, count);
    PipeBarrier<PIPE_V>();
    Adds(out, out, eps, count);
    PipeBarrier<PIPE_V>();
    Sqrt(out, out, count);
    PipeBarrier<PIPE_V>();

    Div(out, in, out, count);
    PipeBarrier<PIPE_V>();
    Mul(out, out, gamma, count);
    PipeBarrier<PIPE_V>();
    Add(out, out, beta, count);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void ApplyKPoolLayerNormToCache(
    const GlobalTensor<float> &stateCache, const GlobalTensor<int32_t> &blockTable,
    const GlobalTensor<float> &normWeight, const GlobalTensor<float> &normBias, KPoolTools<COMP> &tools,
    const ConstInfo &constInfo, uint32_t coff, const LocalTensor<float> &input, const LocalTensor<float> &output,
    const LocalTensor<float> &mean, const LocalTensor<float> &gamma, const LocalTensor<float> &beta)
{
    const uint32_t headDim = constInfo.headDim;
    const uint32_t hiddenDim = coff * headDim;
    const uint32_t vectorId = constInfo.aiCoreIdx * 2 + GetSubBlockIdx();
    const uint32_t vectorCount = constInfo.usedCoreNum * 2;

    DataCopy(gamma, normWeight, hiddenDim);
    DataCopy(beta, normBias, hiddenDim);
    PipeBarrier<PIPE_MTE2>();

    for (uint32_t bIdx = vectorId; bIdx < constInfo.batchSize; bIdx += vectorCount) {
        const uint32_t seqLength = tools.GetSeqLength(bIdx);
        const uint64_t startPos = tools.GetStartPos(bIdx);
        const uint64_t blockTableOffset = static_cast<uint64_t>(bIdx) * constInfo.maxBlockNumPerBatch;

        for (uint32_t sIdx = 0; sIdx < seqLength; ++sIdx) {
            const uint64_t logicalPos = startPos + sIdx;
            const uint64_t logicalBlock = logicalPos / constInfo.blockSize;
            const uint64_t blockOffset = logicalPos % constInfo.blockSize;
            const uint64_t physicalBlock = blockTable.GetValue(blockTableOffset + logicalBlock);
            const uint64_t stateOffset =
                physicalBlock * constInfo.stateCacheStrideDim0 +
                blockOffset * STATE_INTERLEAVE_FACTOR * hiddenDim;

            DataCopy(input, stateCache[stateOffset], headDim);
            if (coff == 2) {
                DataCopy(input[headDim], stateCache[stateOffset + headDim], headDim);
            }
            PipeBarrier<PIPE_MTE2>();

            KPoolLayerNorm(output, input, mean, gamma, beta, constInfo.normEps, hiddenDim);

            DataCopy(stateCache[stateOffset], output, headDim);
            if (coff == 2) {
                DataCopy(stateCache[stateOffset + headDim], output[headDim], headDim);
            }
            PipeBarrier<PIPE_MTE3>();
        }
    }
}

} // namespace KPool

#endif // K_POOL_LAYER_NORM_H
