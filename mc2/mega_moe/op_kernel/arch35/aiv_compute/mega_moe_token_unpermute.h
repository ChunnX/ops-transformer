/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_UNPERMUTE_H
#define MEGA_MOE_TOKEN_UNPERMUTE_H

#include "mega_moe_mxfp8_dequant.h"
#include "../mega_moe_impl_base.h"
#include "../mega_moe_job_context.h"

namespace MegaMoeImpl {

struct TokenUnpermuteContext {
    AivJobContext job;
    uint32_t tokenNum;
    uint32_t tokenHiddenDim;
    uint32_t topK;
    uint32_t sharedExpertNum;
    uint32_t gmm2NTilesPerGroup;
    uint32_t quantTokenSizeBytes;
    uint32_t fullTokenChunkJobCount;
    MegaMoeUnpermuteBufferConfig fullTokenChunkConfig;
    MegaMoeUnpermuteBufferConfig tailTokenChunkConfig;
};

struct TokenUnpermuteArgs {
    GM_ADDR topkWeightsPtr;
    GM_ADDR expandedXPtr;
    GM_ADDR outputPtr;
    GM_ADDR sharedExpertResultPtr;
    GM_ADDR sharedExpertGmm2TileCounterPtr;
};

struct TokenUnpermuteScratch {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<float> fp32ScaleTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
};

// Builds the adaptive UB views for one Unpermute job.
template <uint8_t CombineMode, typename TopkWeightsType>
__aicore__ inline MegaMoeUnpermuteBufferConfig InitTokenUnpermuteBuffers(
    const TokenUnpermuteContext &context, TokenUnpermuteScratch &scratch)
{
    MegaMoeUnpermuteBufferConfig bufferConfig =
        context.job.jobIndex < context.fullTokenChunkJobCount ?
            context.fullTokenChunkConfig :
            context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(context.tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) *
                static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(float)) *
                static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor = LocalTensor<bfloat16_t>(
        TPosition::VECCALC, dataResAddr, dataResBufBytes / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor = LocalTensor<float>(
        TPosition::VECCALC, dataResFp32Addr, dataResFp32BufBytes / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor = LocalTensor<float>(
        TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr,
            bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        scratch.bf16ScaleTensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        scratch.fp32ScaleTensor = LocalTensor<float>(
            TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
    }
    return bufferConfig;
}

// Loads and converts one batch of top-k weights.
template <typename TopkWeightsType>
__aicore__ inline void LoadTokenUnpermuteWeights(const TokenUnpermuteContext &context,
                                                 const TokenUnpermuteArgs &args,
                                                 TokenUnpermuteScratch &scratch, int32_t jobOffset,
                                                 int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.topkWeightsPtr));
        DataCopyPad(scratch.topKWeightsTensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * context.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * context.topK * sizeof(float)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.topkWeightsPtr));
        DataCopyPad(scratch.topKWeightsBf16Tensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * context.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * context.topK * sizeof(bfloat16_t)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(scratch.topKWeightsTensor, scratch.topKWeightsBf16Tensor, RoundMode::CAST_NONE,
             batchTokenCount * context.topK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}

// Accumulates all routed-expert inputs for one token.
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateRoutedExpertsForToken(
    const TokenUnpermuteContext &context, TokenUnpermuteScratch &scratch, int32_t tokenIdx,
    int32_t localIdx, const GlobalTensor<bfloat16_t> &expandedX,
    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(context.topK); ++expertIdx) {
        int32_t accumulationItemIdx =
            localIdx * static_cast<int32_t>(context.topK + context.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        WaitFlag<HardEvent::V_MTE2>(event);
        uint64_t routeIndex = static_cast<uint64_t>(tokenIdx) * context.topK + expertIdx;
        if constexpr (CombineMode == COMBINE_NO_QUANT) {
            DataCopy(dataInBf16, expandedX[routeIndex * context.tokenHiddenDim], context.tokenHiddenDim);
            SetFlag<HardEvent::MTE2_V>(event);
            WaitFlag<HardEvent::MTE2_V>(event);
            Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, context.tokenHiddenDim);
        } else {
            uint32_t quantTokenElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
            DataCopy(dataInBf16, expandedX[routeIndex * quantTokenElementCount], quantTokenElementCount);
            SetFlag<HardEvent::MTE2_V>(event);
            WaitFlag<HardEvent::MTE2_V>(event);
            uint32_t nScale = Ops::Base::CeilDiv(context.tokenHiddenDim,
                                                 static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
            using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT,
                                                      fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(
                dataInBf16, dataInFp32, scratch.bf16ScaleTensor, scratch.fp32ScaleTensor, nScale,
                context.tokenHiddenDim);
        }
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if constexpr (TopkWeightsPrefetch) {
            if (expertIdx == 0) {
                DataCopy(scratch.dataResFp32Tensor, dataInFp32, context.tokenHiddenDim);
            } else {
                Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32,
                    context.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
        } else {
            float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * context.topK + expertIdx);
            if (expertIdx == 0) {
                Muls(scratch.dataResFp32Tensor, dataInFp32, expertScale, context.tokenHiddenDim);
            } else {
                Muls(dataInFp32, dataInFp32, expertScale, context.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
                Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32,
                    context.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_MTE2>(event);
    }
}

// Accumulates one already-ready shared-expert input for one token.
__aicore__ inline void AccumulateSharedExpertForToken(
    const TokenUnpermuteContext &context, const TokenUnpermuteArgs &args, TokenUnpermuteScratch &scratch,
    int32_t tokenIdx, int32_t localIdx, uint32_t sharedExpertIdx,
    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.sharedExpertResultPtr));
    int32_t accumulationItemIdx =
        localIdx * static_cast<int32_t>(context.topK + context.sharedExpertNum) +
        static_cast<int32_t>(context.topK + sharedExpertIdx);
    int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    LocalTensor<bfloat16_t> dataInBf16 =
        scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> dataInFp32 =
        scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(dataInBf16,
             sharedResult[(static_cast<uint64_t>(sharedExpertIdx) * context.tokenNum + tokenIdx) *
                          context.tokenHiddenDim],
             context.tokenHiddenDim);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
    Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, context.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, context.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_UNPERMUTE_H
