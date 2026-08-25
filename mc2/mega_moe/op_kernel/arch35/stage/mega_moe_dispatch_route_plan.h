/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_DISPATCH_ROUTE_PLAN_H
#define MEGA_MOE_DISPATCH_ROUTE_PLAN_H

#include "mega_moe_token_dispatch.h"

namespace MegaMoeImpl {

using namespace AscendC;

/**
 * 将一个 (expert, source-rank) 的 sendMask 压紧为 topkIndex，并按 cumsum
 * 对应的全局 row 区间写入共享 Route Table。
 *
 * Route Table 不改变 Dispatch 的逻辑 row 语义：每个 rank 的 compact 序列仍
 * 按 mask 中 topkIndex 的升序排列。它只把原来每个 wave、每个 core 都会重复
 * 执行的 DataCopy/CreateVecIndex/GatherMask 前移为每个 (expert, rank) 一次。
 */
template <typename ActivationType>
__aicore__ inline void BuildDispatchExpertRankRouteIndex(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    uint32_t remoteRankIdx, uint32_t *gatherMaskCallCount)
{
    const uint32_t pairIndex = localExpertId * context.common.worldSize + remoteRankIdx;
    int32_t globalRowBegin = pairIndex == 0U ? 0 :
        scratch.cumsumInfoTensor.GetValue(pairIndex - 1U);
    int32_t globalRowEnd = scratch.cumsumInfoTensor.GetValue(pairIndex);
    if (globalRowBegin < 0) {
        globalRowBegin = 0;
    }
    if (globalRowEnd < globalRowBegin) {
        globalRowEnd = globalRowBegin;
    }
    if (static_cast<uint64_t>(globalRowBegin) >= context.maxOutputSize) {
        return;
    }
    if (static_cast<uint64_t>(globalRowEnd) > context.maxOutputSize) {
        globalRowEnd = static_cast<int32_t>(context.maxOutputSize);
    }
    const int32_t expectedRouteCount = globalRowEnd - globalRowBegin;
    if (expectedRouteCount <= 0) {
        return;
    }

    const uint64_t sendTotalNum =
        static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    int32_t routeCapacity = context.bufferConfig.routeItemsPerBatch /
        static_cast<int32_t>(ALIGN_256) * static_cast<int32_t>(ALIGN_256);
    if (routeCapacity <= 0) {
        return;
    }

    const uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
    GlobalTensor<int32_t> routeIndexGlobal;
    routeIndexGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.dispatchRouteIndexPtr));

    int32_t compactRouteCount = 0;
    for (int32_t routeBegin = 0;
         static_cast<uint64_t>(routeBegin) < sendTotalNum &&
         compactRouteCount < expectedRouteCount;
         routeBegin += routeCapacity) {
        int32_t validRouteCount = static_cast<int32_t>(
            sendTotalNum - static_cast<uint64_t>(routeBegin));
        if (validRouteCount > routeCapacity) {
            validRouteCount = routeCapacity;
        }
        int32_t maskSliceBytes = Ops::Base::CeilAlign(
            validRouteCount, static_cast<int32_t>(ALIGN_256)) / 8;
        const int32_t maskByteBegin = routeBegin / 8;
        if (maskByteBegin + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
            maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - maskByteBegin;
        }
        if (maskSliceBytes <= 0) {
            break;
        }

        remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
            params.peermemInfo.maskRecvPtr +
            static_cast<uint64_t>(pairIndex) * maskSlotSize +
            static_cast<uint64_t>(maskByteBegin)));
        DataCopy(scratch.maskBatchTensor, remoteRankMaskGlobal,
                 static_cast<uint32_t>(maskSliceBytes));
        CreateVecIndex(scratch.topkIndexTensor, routeBegin, routeCapacity);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

        uint64_t batchMatchedRouteCount = 0U;
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (gatherMaskCallCount != nullptr) {
            ++(*gatherMaskCallCount);
        }
#endif
        GatherMask(scratch.validTopkIndexTensor, scratch.topkIndexTensor,
                   scratch.maskBatchU32Tensor, true,
                   static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0},
                   batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

        int32_t writeCount = static_cast<int32_t>(batchMatchedRouteCount);
        const int32_t remainingRouteCount = expectedRouteCount - compactRouteCount;
        if (writeCount > remainingRouteCount) {
            writeCount = remainingRouteCount;
        }
        if (writeCount <= 0) {
            continue;
        }

        // GatherMask 的 vector 输出先对 MTE3 可见；写出完成后再允许下一批
        // GatherMask 复用 validTopkIndexTensor。
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopyPad(
            routeIndexGlobal[globalRowBegin + compactRouteCount],
            scratch.validTopkIndexTensor,
            {1U, static_cast<uint32_t>(writeCount * sizeof(int32_t)), 0U, 0U, 0U});
        SyncFuncStatic<AscendC::HardEvent::MTE3_V, SYNC_EVENT_ID3>();
        compactRouteCount += writeCount;
    }
}

/**
 * 所有 AIV1 按 (expert, rank) pair 均分 RoutePlan 工作。每核写完自己的 GM
 * 区间后发布独立 done；core0 汇总所有 done 后发布 Route Table ready。该同步
 * 与 Dispatch->GMM1、GMM2->Combine 的业务 flag 完全隔离。
 */
template <typename ActivationType>
__aicore__ inline void BuildDispatchRouteIndexTable(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch,
    uint32_t *gatherMaskCallCount = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || context.blockJob.totalJobs == 0U ||
        context.blockJob.jobIndex >= context.blockJob.totalJobs) {
        return;
    }

    const uint32_t pairCount =
        context.common.moeExpertPerRank * context.common.worldSize;
    for (uint32_t pairIndex = context.blockJob.jobIndex;
         pairIndex < pairCount; pairIndex += context.blockJob.totalJobs) {
        BuildDispatchExpertRankRouteIndex(
            context, params, scratch,
            pairIndex / context.common.worldSize,
            pairIndex % context.common.worldSize,
            gatherMaskCallCount);
    }

    // 确保本核 Route Table 的所有 MTE3 写入完成后才发布 producer done。
    PipeBarrier<PIPE_ALL>();
    __gm__ int32_t *routePlanFlags = reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.dispatchRoutePlanFlagPtr);
    WriteGmByPassDCache(
        routePlanFlags + static_cast<uint64_t>(context.blockJob.jobIndex) * INT_CACHELINE,
        static_cast<int32_t>(1));

    __gm__ int32_t *routeTableReady = routePlanFlags +
        static_cast<uint64_t>(context.blockJob.totalJobs) * INT_CACHELINE;
    if (context.blockJob.jobIndex == 0U) {
        for (uint32_t producerIdx = 0U;
             producerIdx < context.blockJob.totalJobs; ++producerIdx) {
            GmSignalWaitBarrier(
                routePlanFlags + static_cast<uint64_t>(producerIdx) * INT_CACHELINE,
                static_cast<int32_t>(1));
        }
        WriteGmByPassDCache(routeTableReady, static_cast<int32_t>(1));
    }
    GmSignalWaitBarrier(routeTableReady, static_cast<int32_t>(1));
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DISPATCH_ROUTE_PLAN_H
