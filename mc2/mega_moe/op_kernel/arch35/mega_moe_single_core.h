/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SINGLE_CORE_H
#define MEGA_MOE_SINGLE_CORE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

// AIV logical thread information supplied by the caller. threadIdx must be a
// zero-based, contiguous index in [0, threadNum), independent of physical AIV IDs.
struct AivThreadContext {
    uint32_t threadIdx;
    uint32_t threadNum;
};

struct MoeTopologyContext {
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
};

struct TokenShapeContext {
    int32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
};

// Stable information shared by the dispatch preparation stage.
struct DispatchPrepareContext {
    AivThreadContext thread;
    MoeTopologyContext topology;
    TokenShapeContext tokenShape;
};

__aicore__ inline void TilingByThreadContext(int32_t totalLen, int32_t &threadLen, int32_t &threadOffset,
    uint32_t threadIdx, uint32_t threadNum, int32_t align = ALIGN_32)
{
    if (threadNum == 0U || threadIdx >= threadNum) {
        threadLen = 0;
        threadOffset = 0;
        return;
    }
    int32_t lenPerThread = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), threadNum);
    int32_t lenPerThreadAlign =
        Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerThread), static_cast<uint32_t>(align));
    threadLen = lenPerThreadAlign;
    threadOffset = static_cast<int32_t>(threadIdx) * lenPerThreadAlign;
    if (threadOffset + threadLen >= totalLen) {
        threadLen = totalLen - threadOffset;
    }
    if (threadOffset >= totalLen) {
        threadLen = 0;
    }
}

struct ResetWorkspaceArgs {
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    int32_t flagNum;
    int32_t gmm2CombineSyncCounterNum;
    bool resetCombineCounters;
};

__aicore__ inline void ResetGmm2CombineSyncCountersOneCore(const AivThreadContext &thread,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> resetTensor)
{
    if constexpr (g_coreType == AIV) {
        if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
            return;
        }
        int32_t threadLen, threadOffset;
        TilingByThreadContext(args.gmm2CombineSyncCounterNum, threadLen, threadOffset,
            thread.threadIdx, thread.threadNum);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)args.gmm2CombineSyncCounterPtr);
        if (threadLen > 0) {
            Duplicate(resetTensor, 0, threadLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(gmm2CombineSyncCounterGm[threadOffset], resetTensor, threadLen);
        }
    }
}

__aicore__ inline void ResetFlagListOneCore(const DispatchPrepareContext &context,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
        return;
    }
    GlobalTensor<int32_t> swigluToGmm2FlagGm;
    swigluToGmm2FlagGm.SetGlobalBuffer((__gm__ int32_t*)args.flagSwiGluToGmm2Ptr);
    int32_t threadLen, threadOffset;
    TilingByThreadContext(args.flagNum, threadLen, threadOffset, thread.threadIdx, thread.threadNum, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(threadLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (threadLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm[threadOffset], resetTensor, rankSyncCopyParams);
    }
    if (args.resetCombineCounters) {
        ResetGmm2CombineSyncCountersOneCore(thread, args, resetTensor);
    }
}

struct SendMaskArgs {
    GM_ADDR expertIdxGmAddr;
    GM_ADDR *winRankAddr;
    int32_t compareCount;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;
    uint64_t maskWinOffset;
};

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    LocalTensor<uint8_t> sendMaskTensor[DOUBLE_BUFFER];
    LocalTensor<int32_t> sendGatherOutTensor;
};

__aicore__ inline LocalTensor<uint8_t> GetSendMaskTensor(SendMaskScratch &scratch, int32_t iter)
{
    return scratch.sendMaskTensor[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
}

__aicore__ inline void SendMaskCalOneExpert(const DispatchPrepareContext &context,
    const SendMaskArgs &args, SendMaskScratch &scratch, uint64_t sendTotalNum,
    int32_t curExpertId, int32_t iter)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    uint32_t countWordIdx = static_cast<uint32_t>(args.maskAlignSize) / sizeof(int32_t);
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(args.maskSlotSize), 0U, 0U, 0U};
    GlobalTensor<uint8_t> dstGlobalTensor;

    const MoeTopologyContext &topology = context.topology;
    int32_t dstRank = curExpertId / static_cast<int32_t>(topology.expertPerRank);
    int32_t expertNumPer = curExpertId % static_cast<int32_t>(topology.expertPerRank);
    TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
    LocalTensor<uint8_t> maskBuf = GetSendMaskTensor(scratch, iter);
    LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();

    WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
    CompareScalar(maskBuf, scratch.topkIdsTensor, curExpertId, AscendC::CMPMODE::EQ, args.compareCount);
    uint64_t sendCnt = 0;
    GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
        static_cast<uint32_t>(sendTotalNum), {1, 1, 0, 0}, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

    uint64_t dstOffset = args.maskWinOffset +
        static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(topology.worldSize) +
        static_cast<int32_t>(topology.rankId)) * static_cast<uint64_t>(args.maskSlotSize);
    dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(args.winRankAddr[dstRank] + dstOffset));
    DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
}

__aicore__ inline void SendMaskCalOneCore(const DispatchPrepareContext &context, const SendMaskArgs &args,
    SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
        return;
    }

    uint64_t sendTotalNum =
        static_cast<uint64_t>(context.tokenShape.tokenNum) * static_cast<uint64_t>(context.tokenShape.topK);
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)args.expertIdxGmAddr);
    Duplicate<int32_t>(scratch.topkIdsTensor, 0, args.compareCount);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts =
        static_cast<int32_t>(context.topology.worldSize * context.topology.expertPerRank);
    int32_t iter = 0;
    for (int32_t curExpertId = static_cast<int32_t>(thread.threadIdx); curExpertId < totalExperts;
        curExpertId += static_cast<int32_t>(thread.threadNum), ++iter) {
        SendMaskCalOneExpert(context, args, scratch, sendTotalNum, curExpertId, iter);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

struct QuantProcessArgs {
    GM_ADDR xGmAddr;
    GM_ADDR quantTokenScalePtr;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleNumAlignPerToken;
};

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

// QuantProcessInRankOneCore：量化当前逻辑线程负责的本卡 token。
// 依赖项由 DispatchPrepareContext/QuantProcessArgs/QuantProcessScratch 显式传入，
// 不读取物理核 ID 或 MegaMoe 成员。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantProcessInRankOneCore(const DispatchPrepareContext &context,
    const QuantProcessArgs &args, QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum) {
        return;
    }

    int32_t currentNum;
    int32_t currentOffset;
    TilingByThreadContext(
        context.tokenShape.tokenNum, currentNum, currentOffset, thread.threadIdx, thread.threadNum, 1);
    uint32_t hiddenDim = context.tokenShape.tokenHiddenDim;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(args.quantTokenAlignBytes + args.quantScaleAlignBytes), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)(args.xGmAddr +
            (currentOffset + index) * hiddenDim * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(args.quantTokenScalePtr +
            (currentOffset + index) * args.quantTokenScaleAlignBytes));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)scratch.mxTempTensor.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)scratch.mxTempTensor[Ops::Base::CeilAlign(
            args.quantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[args.quantTokenAlignBytes].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
            args.quantScaleNumAlignPerToken);
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace MegaMoeImpl

#endif
