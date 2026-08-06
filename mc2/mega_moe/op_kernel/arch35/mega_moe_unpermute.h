/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_UNPERMUTE_H
#define MEGA_MOE_UNPERMUTE_H

#include "aiv_compute/mega_moe_token_unpermute.h"

namespace MegaMoeImpl {

// Waits for the shared-expert GMM2 tile group consumed by one Unpermute token.
template <uint32_t Gmm1TileM>
__aicore__ inline void WaitSharedExpertGmm2Ready(const TokenUnpermuteContext &context,
                                                 const TokenUnpermuteArgs &args, int32_t tokenIdx,
                                                 uint32_t sharedExpertIdx)
{
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / Gmm1TileM;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(context.tokenNum, Gmm1TileM);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    __gm__ int32_t *counterAddr = GetCombineSyncCounterAddress(
        reinterpret_cast<__gm__ int32_t *>(args.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedExpertIdx) * sharedExpertStride,
        tokenGroupIndex);
    while (AscendC::ReadGmByPassDCache(counterAddr) !=
           static_cast<int32_t>(context.gmm2NTilesPerGroup)) {
        int64_t waitStartCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
        }
    }
}

// Prototype: MegaMoe::Unpermute. Controls the full Unpermute pipeline for the assigned token partition.
template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch,
          uint32_t Gmm1TileM>
__aicore__ inline void UnpermuteTokens(const TokenUnpermuteContext &context,
                                       const TokenUnpermuteArgs &args,
                                       TokenUnpermuteScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t jobLen = 0;
    int32_t jobOffset = 0;
    TilingByJobContext(static_cast<int32_t>(context.tokenNum), jobLen, jobOffset,
                       context.job.jobIndex, context.job.totalJobs, 1);
    if (jobLen == 0) {
        return;
    }
    MegaMoeUnpermuteBufferConfig bufferConfig =
        InitTokenUnpermuteBuffers<CombineMode, TopkWeightsType>(context, scratch);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.expandedXPtr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.outputPtr));

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen;
         batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount =
            batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ?
                jobLen - batchTokenOffset :
                bufferConfig.tokensPerBatch;
        if constexpr (!TopkWeightsPrefetch) {
            LoadTokenUnpermuteWeights<TopkWeightsType>(
                context, args, scratch, jobOffset, batchTokenOffset, batchTokenCount);
        }
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
        for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
            int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
            AccumulateRoutedExpertsForToken<CombineMode, TopkWeightsPrefetch>(
                context, scratch, tokenIdx, localIdx, expandedX, bufferConfig);
            for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < context.sharedExpertNum;
                 ++sharedExpertIdx) {
                WaitSharedExpertGmm2Ready<Gmm1TileM>(context, args, tokenIdx, sharedExpertIdx);
                AccumulateSharedExpertForToken(
                    context, args, scratch, tokenIdx, localIdx, sharedExpertIdx, bufferConfig);
            }
            WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
            Cast(scratch.dataResTensor, scratch.dataResFp32Tensor, RoundMode::CAST_RINT,
                 context.tokenHiddenDim);
            SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * context.tokenHiddenDim],
                     scratch.dataResTensor, context.tokenHiddenDim);
            SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
        }
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
    }
    WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UNPERMUTE_H
