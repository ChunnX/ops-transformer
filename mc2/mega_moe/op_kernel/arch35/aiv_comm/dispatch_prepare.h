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
    int32_t resetBatchElementCount;
    bool resetCombineCounters;
};

// Prototype: MegaMoe::ResetGmm2CombineSyncCounters. Clears one logical AIV thread's combine counters in batches.
__aicore__ inline void ResetGmm2CombineSyncCountersOneCore(const AivThreadContext &thread,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIV) {
        if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum ||
            args.resetBatchElementCount <= 0) {
            return;
        }
        int32_t threadLen, threadOffset;
        TilingByThreadContext(args.gmm2CombineSyncCounterNum, threadLen, threadOffset,
            thread.threadIdx, thread.threadNum);
        GlobalTensor<int32_t> counterGm;
        counterGm.SetGlobalBuffer((__gm__ int32_t *)args.gmm2CombineSyncCounterPtr);
        if (threadLen > 0) {
            Duplicate(resetTensor, 0, args.resetBatchElementCount);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            for (int32_t offset = 0; offset < threadLen; offset += args.resetBatchElementCount) {
                int32_t batchCount = threadLen - offset < args.resetBatchElementCount ?
                    threadLen - offset : args.resetBatchElementCount;
                DataCopy(counterGm[threadOffset + offset], resetTensor, batchCount);
            }
        }
    }
}

// Prototype: MegaMoe::ResetFlagList. Clears one logical AIV thread's dispatch flags in bounded UB batches.
__aicore__ inline void ResetFlagListOneCore(const DispatchPrepareContext &context,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum || args.resetBatchElementCount <= 0) {
        return;
    }
    GlobalTensor<int32_t> flagGm;
    flagGm.SetGlobalBuffer((__gm__ int32_t *)args.flagSwiGluToGmm2Ptr);
    int32_t threadLen, threadOffset;
    TilingByThreadContext(args.flagNum, threadLen, threadOffset, thread.threadIdx, thread.threadNum, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    for (int32_t offset = 0; offset < threadLen; offset += args.resetBatchElementCount) {
        int32_t batchCount = threadLen - offset < args.resetBatchElementCount ?
            threadLen - offset : args.resetBatchElementCount;
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(flagGm[threadOffset + offset], resetTensor, copyParams);
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
    int32_t routeItemsPerBatch;
    int32_t routeBatchCount;
};

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    LocalTensor<uint8_t> sendMaskTensor[DOUBLE_BUFFER];
    LocalTensor<int32_t> sendGatherOutTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

// Prototype: MegaMoe::SendMaskCal. Builds and sends per-expert masks for one logical AIV thread using route batches.
__aicore__ inline void SendMaskCalOneCore(const DispatchPrepareContext &context, const SendMaskArgs &args,
    SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.threadNum == 0U || thread.threadIdx >= thread.threadNum ||
        args.routeItemsPerBatch <= 0 || args.routeBatchCount <= 0) {
        return;
    }

    const MoeTopologyContext &topology = context.topology;
    uint64_t sendTotalNum = static_cast<uint64_t>(context.tokenShape.tokenNum) * context.tokenShape.topK;
    int32_t totalExperts = static_cast<int32_t>(topology.worldSize * topology.expertPerRank);
    int32_t threadIdx = static_cast<int32_t>(thread.threadIdx);
    int32_t threadNum = static_cast<int32_t>(thread.threadNum);
    int32_t ownedExpertNum = threadIdx < totalExperts ?
        Ops::Base::CeilDiv(totalExperts - threadIdx, threadNum) : 0;
    if (ownedExpertNum <= 0) {
        return;
    }

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer((__gm__ int32_t *)args.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstMaskGm;
    int32_t fullSliceBytes = args.routeItemsPerBatch / 8;
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};

    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t bufIdx = 0; bufIdx < static_cast<int32_t>(DOUBLE_BUFFER); ++bufIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[bufIdx]);
    }

    int32_t iter = 0;
    for (int32_t batchIdx = 0; batchIdx < args.routeBatchCount; ++batchIdx) {
        int32_t batchStart = batchIdx * args.routeItemsPerBatch;
        bool isLastBatch = batchIdx == args.routeBatchCount - 1;
        int32_t validLen = args.routeItemsPerBatch;
        int32_t sliceBytes = fullSliceBytes;
        int32_t pushBytes = sliceBytes;
        if (isLastBatch) {
            validLen = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchStart));
            if (batchStart / 8 + sliceBytes > static_cast<int32_t>(args.maskAlignSize)) {
                sliceBytes = static_cast<int32_t>(args.maskAlignSize) - batchStart / 8;
            }
            pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
        }

        SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++iter) {
            int32_t globalExpertId = threadIdx + ownedIdx * threadNum;
            int32_t dstRank = globalExpertId / static_cast<int32_t>(topology.expertPerRank);
            int32_t localExpertId = globalExpertId % static_cast<int32_t>(topology.expertPerRank);
            TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint8_t> maskBuf = scratch.sendMaskTensor[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

            WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
            CompareScalar(maskBuf, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ,
                args.routeItemsPerBatch);
            uint64_t batchMatchedRouteCount = 0;
            GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

            int32_t expertMatchedRouteCount = scratch.sendCntAccTensor.GetValue(ownedIdx) +
                static_cast<int32_t>(batchMatchedRouteCount);
            scratch.sendCntAccTensor.SetValue(ownedIdx, expertMatchedRouteCount);
            if (isLastBatch) {
                maskBuf.template ReinterpretCast<int32_t>().SetValue(
                    sliceBytes / sizeof(int32_t), expertMatchedRouteCount);
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

            uint64_t dstOffset = args.maskWinOffset +
                static_cast<uint64_t>(localExpertId * static_cast<int32_t>(topology.worldSize) +
                    static_cast<int32_t>(topology.rankId)) * args.maskSlotSize +
                static_cast<uint64_t>(batchStart / 8);
            dstMaskGm.SetGlobalBuffer((__gm__ uint8_t *)(args.winRankAddr[dstRank] + dstOffset));
            DataCopyPad(dstMaskGm, maskBuf, {1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
    }

    for (int32_t bufIdx = 0; bufIdx < static_cast<int32_t>(DOUBLE_BUFFER); ++bufIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[bufIdx]);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DISPATCH_PREPARE_H
