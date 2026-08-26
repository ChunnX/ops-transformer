/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SEND_MASK_H
#define MEGA_MOE_SEND_MASK_H

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct SendMaskConfig {
    uint32_t maskAlignSize;
    uint64_t maskWinOffset;
    MegaMoeSendMaskBufferConfig bufferConfig;
    // A8W8 Wave compact 路由表：每个 (localExpert, sourceRank) 槽位的数据区大小和窗口偏移。
    uint32_t routeIndexAlignSize;
    uint64_t routeIndexWinOffset;
};

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    // A8W8 Wave compact 路径中保存当前 batch 的全局 topkIndex；每个 batch 只生成一次。
    LocalTensor<int32_t> topkIndexTensor;
    // Contiguous runtime-sized ring. Slot i starts at i * bufferConfig.bufferBytes.
    LocalTensor<uint8_t> sendMaskTensor;
    LocalTensor<int32_t> sendGatherOutTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

__aicore__ inline void GatherAndSendExpertMaskBatch(
    const DispatchPrepareConfig &context, GM_ADDR *winRankAddr, const SendMaskConfig &config,
    SendMaskScratch &scratch,
    GlobalTensor<int32_t> &topkIdsGm, GlobalTensor<uint8_t> &dstMaskGm, int32_t ownedExpertNum,
    int32_t batchIdx, int32_t &ringIteration)
{
    const AivJobContext &job = context.job;
    const MoeStageCommonConfig &common = context.common;
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    uint32_t maskSlotSize = config.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
    bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
    int32_t validLen = bufferConfig.routeItemsPerBatch;
    int32_t sliceBytes = bufferConfig.routeItemsPerBatch / 8;
    int32_t pushBytes = sliceBytes;
    if (isLastBatch) {
        uint64_t sendTotalNum = static_cast<uint64_t>(common.tokenNum) * common.topK;
        validLen = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchStart));
        if (batchStart / 8 + sliceBytes > static_cast<int32_t>(config.maskAlignSize)) {
            sliceBytes = static_cast<int32_t>(config.maskAlignSize) - batchStart / 8;
        }
        pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++ringIteration) {
        int32_t globalExpertId = jobIndex + ownedIdx * totalJobs;
        int32_t dstRank = globalExpertId / static_cast<int32_t>(common.moeExpertPerRank);
        int32_t localExpertId = globalExpertId % static_cast<int32_t>(common.moeExpertPerRank);
        int32_t bufferIdx = ringIteration % bufferConfig.bufferCount;
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        LocalTensor<uint8_t> maskBuf = scratch.sendMaskTensor[bufferIdx * bufferConfig.bufferBytes];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuf, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ,
                      bufferConfig.routeItemsPerBatch);
        uint64_t batchMatchedRouteCount = 0;
        GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
                   static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        int32_t expertMatchedRouteCount =
            scratch.sendCntAccTensor.GetValue(ownedIdx) + static_cast<int32_t>(batchMatchedRouteCount);
        scratch.sendCntAccTensor.SetValue(ownedIdx, expertMatchedRouteCount);
        if (isLastBatch) {
            maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t),
                                                                 expertMatchedRouteCount);
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        uint64_t dstOffset = config.maskWinOffset +
                             static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) +
                                                   static_cast<int32_t>(common.rankId)) *
                                 maskSlotSize +
                             static_cast<uint64_t>(batchStart / 8);
        dstMaskGm.SetGlobalBuffer((__gm__ uint8_t *)(winRankAddr[dstRank] + dstOffset));
        DataCopyPad(dstMaskGm, maskBuf, {1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
}

// Prototype: MegaMoe::SendMaskCal. Builds and sends per-expert masks for one logical AIV job using route batches.
__aicore__ inline void GatherAndSendExpertMasks(const DispatchPrepareConfig &context, const Params &params,
                                                GM_ADDR *winRankAddr, const SendMaskConfig &config,
                                                SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    const MoeStageCommonConfig &common = context.common;
    int32_t totalExperts = static_cast<int32_t>(common.worldSize * common.moeExpertPerRank);
    int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    int32_t ownedExpertNum = jobIndex < totalExperts ? Ops::Base::CeilDiv(totalExperts - jobIndex, totalJobs) : 0;
    if (ownedExpertNum <= 0) {
        return;
    }

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer((__gm__ int32_t *)params.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstMaskGm;
    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufIdx = 0; bufIdx < bufferConfig.bufferCount; ++bufIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufIdx));
    }

    int32_t ringIteration = 0;
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        GatherAndSendExpertMaskBatch(
            context, winRankAddr, config, scratch, topkIdsGm, dstMaskGm, ownedExpertNum, batchIdx, ringIteration);
    }

    for (int32_t bufIdx = 0; bufIdx < bufferConfig.bufferCount; ++bufIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufIdx));
    }
}

/**
 * A8W8 Wave compact 路径：源 rank 直接发送按原始 topkIndex 升序排列的有效索引。
 *
 * 一个远端槽位固定容纳 tokenNum 个 int32 index，末尾 32B 的第一个 int32 保存 count。
 * 正常 TopK 输入中同一 token 不会重复选择同一专家，因此单槽 count 不超过 tokenNum；写入前仍做
 * 上界裁剪，保证异常输入不会产生越界 MTE3。
 */
__aicore__ inline void GatherAndSendExpertCompactRouteBatch(
    const DispatchPrepareConfig &context, GM_ADDR *winRankAddr, const SendMaskConfig &config,
    SendMaskScratch &scratch, GlobalTensor<int32_t> &topkIdsGm, GlobalTensor<int32_t> &dstRouteIndexGm,
    int32_t ownedExpertNum, int32_t batchIdx, int32_t &ringIteration)
{
    const AivJobContext &job = context.job;
    const MoeStageCommonConfig &common = context.common;
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    const uint32_t maskBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    const uint32_t compactBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    const uint32_t routeSlotSize = config.routeIndexAlignSize + static_cast<uint32_t>(ALIGN_32);
    const int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
    const bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
    int32_t validLen = bufferConfig.routeItemsPerBatch;
    if (isLastBatch) {
        const uint64_t sendTotalNum = static_cast<uint64_t>(common.tokenNum) * common.topK;
        validLen = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchStart));
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    // 尾 batch 只生成真实索引，避免 batchStart + padding 跨越 int32 topkIndex 上界。
    CreateVecIndex(scratch.topkIndexTensor, batchStart, validLen);

    const int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    const int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++ringIteration) {
        const int32_t globalExpertId = jobIndex + ownedIdx * totalJobs;
        const int32_t dstRank = globalExpertId / static_cast<int32_t>(common.moeExpertPerRank);
        const int32_t localExpertId = globalExpertId % static_cast<int32_t>(common.moeExpertPerRank);
        const int32_t bufferIdx = ringIteration % bufferConfig.bufferCount;
        const TEventID eventId = static_cast<TEventID>(bufferIdx);
        LocalTensor<uint8_t> slot = scratch.sendMaskTensor[bufferIdx * bufferConfig.bufferBytes];
        LocalTensor<uint8_t> maskBuf = slot;
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> compactBuf = slot[maskBytes].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> countBuf = slot[maskBytes + compactBytes].template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuf, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ,
                      bufferConfig.routeItemsPerBatch);
        uint64_t batchMatchedRouteCount = 0U;
        GatherMask(compactBuf, scratch.topkIndexTensor, maskBufU32, true, static_cast<uint32_t>(validLen),
                   {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        const int32_t previousCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        const int32_t remainingCapacity = static_cast<int32_t>(common.tokenNum) - previousCount;
        int32_t copiedCount = static_cast<int32_t>(batchMatchedRouteCount);
        if (copiedCount > remainingCapacity) {
            copiedCount = remainingCapacity > 0 ? remainingCapacity : 0;
        }
        const int32_t expertMatchedRouteCount = previousCount + copiedCount;
        scratch.sendCntAccTensor.SetValue(ownedIdx, expertMatchedRouteCount);
        if (isLastBatch) {
            countBuf.SetValue(0, expertMatchedRouteCount);
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        const uint64_t slotOffset =
            config.routeIndexWinOffset +
            static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) +
                                  static_cast<int32_t>(common.rankId)) *
                routeSlotSize;
        if (copiedCount > 0) {
            dstRouteIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                winRankAddr[dstRank] + slotOffset + static_cast<uint64_t>(previousCount) * sizeof(int32_t)));
            DataCopyPad(dstRouteIndexGm, compactBuf,
                        {1U, static_cast<uint32_t>(copiedCount * sizeof(int32_t)), 0U, 0U, 0U});
        }
        if (isLastBatch) {
            dstRouteIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                winRankAddr[dstRank] + slotOffset + config.routeIndexAlignSize));
            DataCopyPad(dstRouteIndexGm, countBuf,
                        {1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
}

// A8W8 Wave 专用入口。旧 MTE/A8W4/URMA 路径继续调用 GatherAndSendExpertMasks。
__aicore__ inline void GatherAndSendExpertCompactRoutes(
    const DispatchPrepareConfig &context, const Params &params, GM_ADDR *winRankAddr,
    const SendMaskConfig &config, SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    const MoeStageCommonConfig &common = context.common;
    const int32_t totalExperts = static_cast<int32_t>(common.worldSize * common.moeExpertPerRank);
    const int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    const int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    const int32_t ownedExpertNum =
        jobIndex < totalExperts ? Ops::Base::CeilDiv(totalExperts - jobIndex, totalJobs) : 0;
    if (ownedExpertNum <= 0) {
        return;
    }

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertIdxGmAddr));
    GlobalTensor<int32_t> dstRouteIndexGm;
    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }

    int32_t ringIteration = 0;
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        GatherAndSendExpertCompactRouteBatch(context, winRankAddr, config, scratch, topkIdsGm,
                                             dstRouteIndexGm, ownedExpertNum, batchIdx, ringIteration);
    }

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SEND_MASK_H
