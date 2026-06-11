/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_indices_pad_multi_core_gm.h
 * \brief 路径 A：masked select 流 GM GetValue 直读。A/B 测试后若劣于 UB 路径则删除本文件。
 */
#ifndef MOE_INDICES_PAD_MULTI_CORE_GM_H
#define MOE_INDICES_PAD_MULTI_CORE_GM_H

#include "kernel_tiling/kernel_tiling.h"
#include "moe_common.h"

namespace MoeTokenPermute {
using namespace AscendC;

class MoeIndicesPadMultiCoreGm
{
public:
    __aicore__ inline MoeIndicesPadMultiCoreGm() {}

    __aicore__ inline void Init(
        GM_ADDR msTokenIdsGm, GM_ADDR sortedIndicesOut, int32_t actualOutTokensIn,
        const MoeTokenPermuteWithRoutingMapTilingData* tilingData, GM_ADDR workspace, TPipe* tPipeIn);

    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t PadUpAlign(int64_t bytes);
    __aicore__ inline int64_t PadCeilDiv(int64_t a, int64_t b);
    __aicore__ inline void InitTokenBatch(
        int64_t batchTokenSize, LocalTensor<int32_t>& outBatch, LocalTensor<int32_t>& slotCounter);
    __aicore__ inline void TryAssignToken(
        int32_t tokenId, int32_t pos, int64_t batchTokenStart, int64_t batchTokenSize,
        LocalTensor<int32_t>& outBatch, LocalTensor<int32_t>& slotCounter);
    __aicore__ inline void FlushBatchToGm(
        int64_t batchTokenStart, int64_t batchTokenSize, LocalTensor<int32_t>& outBatch);

private:
    static constexpr int64_t PAD_ONE_BLOCK_BYTE = 32;
    static constexpr int64_t PAD_INT32_SIZE = 4;
    static constexpr int64_t PAD_TPIPE_RESERVE = 256;
    static constexpr int64_t PAD_AIV_UB_SIZE = 196608;
    static constexpr int32_t PAD_MINUS_ONE = -1;

    TPipe* pipe;
    TBuf<TPosition::VECCALC> outBatchBuf;
    TBuf<TPosition::VECCALC> slotCounterBuf;

    GlobalTensor<int32_t> msGm;
    GlobalTensor<int32_t> outGm;

    int64_t blockIdx;
    int32_t actualOutTokens;
    int64_t numTokens;
    int64_t topK;
    int64_t needCoreNum;
    int64_t frontCoreNum;
    int64_t coreCalcNum;
    int64_t coreCalcTail;
    int64_t tokenStart;
    int64_t tokenCount;
    int64_t onceToken;
    int64_t outBatchBytes;
    int64_t slotCounterBytes;
};

__aicore__ inline int64_t MoeIndicesPadMultiCoreGm::PadUpAlign(int64_t bytes)
{
    return (bytes + PAD_ONE_BLOCK_BYTE - 1) / PAD_ONE_BLOCK_BYTE * PAD_ONE_BLOCK_BYTE;
}

__aicore__ inline int64_t MoeIndicesPadMultiCoreGm::PadCeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline void MoeIndicesPadMultiCoreGm::Init(
    GM_ADDR msTokenIdsGm, GM_ADDR sortedIndicesOut, int32_t actualOutTokensIn,
    const MoeTokenPermuteWithRoutingMapTilingData* tilingData, GM_ADDR workspace, TPipe* tPipeIn)
{
    (void)workspace;
    pipe = tPipeIn;
    blockIdx = GetBlockIdx();
    actualOutTokens = actualOutTokensIn;
    numTokens = tilingData->n;
    topK = tilingData->topK;

    const IndexCopyComputeRMTilingData* idxTiling = &(tilingData->indexCopyComputeParamsOp);
    needCoreNum = idxTiling->needCoreNum;
    frontCoreNum = idxTiling->frontCoreNum;
    coreCalcNum = idxTiling->coreCalcNum;
    coreCalcTail = idxTiling->coreCalcTail;

    tokenStart = 0;
    tokenCount = 0;
    if (blockIdx < frontCoreNum) {
        tokenStart = blockIdx * coreCalcNum;
        tokenCount = coreCalcNum;
    } else if (blockIdx < needCoreNum) {
        tokenStart = frontCoreNum * coreCalcNum + (blockIdx - frontCoreNum) * coreCalcTail;
        tokenCount = coreCalcTail;
    }

    msGm.SetGlobalBuffer((__gm__ int32_t*)msTokenIdsGm, actualOutTokens);
    outGm.SetGlobalBuffer((__gm__ int32_t*)sortedIndicesOut, numTokens * topK);

    int64_t ubAvail = PAD_AIV_UB_SIZE - PAD_TPIPE_RESERVE;
    int64_t perTokenUb = topK * PAD_INT32_SIZE + PAD_INT32_SIZE;
    onceToken = ubAvail / perTokenUb;
    onceToken = Max(onceToken, static_cast<int64_t>(1));
    if (tokenCount > 0) {
        onceToken = Min(onceToken, tokenCount);
    }

    outBatchBytes = PadUpAlign(onceToken * topK * PAD_INT32_SIZE);
    slotCounterBytes = PadUpAlign(onceToken * PAD_INT32_SIZE);
}

__aicore__ inline void MoeIndicesPadMultiCoreGm::InitTokenBatch(
    int64_t batchTokenSize, LocalTensor<int32_t>& outBatch, LocalTensor<int32_t>& slotCounter)
{
    Duplicate(outBatch, static_cast<int32_t>(PAD_MINUS_ONE), onceToken * topK);
    for (int64_t i = 0; i < batchTokenSize; ++i) {
        slotCounter.SetValue(i, 0);
    }
}

__aicore__ inline void MoeIndicesPadMultiCoreGm::TryAssignToken(
    int32_t tokenId, int32_t pos, int64_t batchTokenStart, int64_t batchTokenSize,
    LocalTensor<int32_t>& outBatch, LocalTensor<int32_t>& slotCounter)
{
    if (tokenId < batchTokenStart || tokenId >= batchTokenStart + batchTokenSize) {
        return;
    }
    if (tokenId < 0 || tokenId >= numTokens) {
        return;
    }
    int64_t localIdx = tokenId - batchTokenStart;
    int32_t slot = slotCounter.GetValue(localIdx);
    if (slot >= topK) {
        return;
    }
    outBatch.SetValue(localIdx * topK + slot, pos);
    slotCounter.SetValue(localIdx, slot + 1);
}

__aicore__ inline void MoeIndicesPadMultiCoreGm::FlushBatchToGm(
    int64_t batchTokenStart, int64_t batchTokenSize, LocalTensor<int32_t>& outBatch)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(batchTokenSize * topK * PAD_INT32_SIZE), 0, 0, 0};
    DataCopyPadCustom(outGm[batchTokenStart * topK], outBatch, copyParams);
}

__aicore__ inline void MoeIndicesPadMultiCoreGm::Process()
{
    if (blockIdx >= needCoreNum || tokenCount <= 0) {
        return;
    }

    pipe->InitBuffer(outBatchBuf, outBatchBytes);
    pipe->InitBuffer(slotCounterBuf, slotCounterBytes);
    LocalTensor<int32_t> outBatch = outBatchBuf.Get<int32_t>();
    LocalTensor<int32_t> slotCounter = slotCounterBuf.Get<int32_t>();

    int64_t numBatches = PadCeilDiv(tokenCount, onceToken);
    for (int64_t batch = 0; batch < numBatches; ++batch) {
        int64_t batchTokenStart = tokenStart + batch * onceToken;
        int64_t batchTokenSize = Min(onceToken, tokenCount - batch * onceToken);
        InitTokenBatch(batchTokenSize, outBatch, slotCounter);

        for (int32_t pos = 0; pos < actualOutTokens; ++pos) {
            int32_t tokenId = msGm.GetValue(pos);
            TryAssignToken(tokenId, pos, batchTokenStart, batchTokenSize, outBatch, slotCounter);
        }

        FlushBatchToGm(batchTokenStart, batchTokenSize, outBatch);
    }
}

} // namespace MoeTokenPermute

#endif // MOE_INDICES_PAD_MULTI_CORE_GM_H
