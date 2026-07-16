/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_DISPATCH_PREPARE_H
#define MEGA_MOE_DISPATCH_PREPARE_H

#include "../single_core_context.h"

namespace MegaMoeImpl {

struct ResetWorkspaceArgs {
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    int32_t flagNum;
    int32_t gmm2CombineSyncCounterNum;
    bool resetCombineCounters;
};

__aicore__ inline void ResetGmm2CombineSyncCountersOneCore(const AivThreadContext &thread,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
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
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
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

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DISPATCH_PREPARE_H
