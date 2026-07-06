/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file topk_gather.h
 * \brief Single-core TopK gather module: SendMaskCal, SendCntCal, ResetFlagList.
 *        All multi-core primitives (GetBlockIdx, GetBlockNum, GetSubBlockIdx,
 *        blockIdx_, aivCoreIdx_, subBlockIdx_, blockAivNum_, blockNum_)
 *        are replaced by explicit TileInfo parameter.
 */

#ifndef MEGA_MOE_SINGLE_CORE_TOPK_GATHER_H
#define MEGA_MOE_SINGLE_CORE_TOPK_GATHER_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

namespace TopKGather {

__aicore__ inline void ResetGmm2CombineSyncCounters(
    MegaMoeModules::TileInfo tile,
    uint32_t expertPerRank,
    LocalTensor<int32_t> &resetTensor,
    const Params &params)
{
    if constexpr(g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(
            static_cast<int64_t>(expertPerRank) * tile.aivTotalTiles * INT_CACHELINE);
        int32_t tileLen, tileOffset;
        MegaMoeModules::TilingByTile(totalCounters, tileLen, tileOffset, tile.aivTileIdx, tile.aivTotalTiles);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.gmm2CombineSyncCounterPtr);
        if (tileLen > 0) {
            Duplicate(resetTensor, 0, tileLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(gmm2CombineSyncCounterGm[tileOffset], resetTensor, tileLen);
        }
    }
}

__aicore__ inline void ResetFlagList(
    MegaMoeModules::TileInfo tile,
    uint32_t expertPerRank, int32_t dispatchFlagSlotsPerExpert, uint32_t aicNum,
    LocalTensor<int32_t> &resetTensor,
    GlobalTensor<int32_t> &swigluToGmm2FlagGm,
    bool combineQuantEnabled,
    const Params &params)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    swigluToGmm2FlagGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(expertPerRank) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert +
        static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum));
    int32_t coreLen, coreOffset;
    MegaMoeModules::TilingByTile(flagNum, coreLen, coreOffset, tile.aivTileIdx, tile.aivTotalTiles, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm[coreOffset], resetTensor, rankSyncCopyParams);
    }
    if (combineQuantEnabled) {
        ResetGmm2CombineSyncCounters(tile, expertPerRank, resetTensor, params);
    }
}

__aicore__ inline void SendMaskCal(
    MegaMoeModules::TileInfo tile,
    uint32_t rankId, uint32_t worldSize, uint32_t expertPerRank,
    uint64_t sendTotalNum, int32_t compareCount,
    uint32_t maskAlignSize, uint32_t maskSlotSize, uint64_t maskWinOffset,
    LocalTensor<int32_t> &topkIdsTensor,
    LocalTensor<uint8_t> (&sendMaskTensor)[DOUBLE_BUFFER],
    LocalTensor<int32_t> &sendGatherOutTensor,
    const Params &params)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params.expertIdxGmAddr);
    Duplicate<int32_t>(topkIdsTensor, 0, compareCount);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize * expertPerRank);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize) / sizeof(int32_t);
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(maskSlotSize), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (int32_t curExpertId = tile.aivTileIdx; curExpertId < totalExperts;
         curExpertId += tile.aivTotalTiles, ++iter) {
        int32_t dstRank = curExpertId / static_cast<int32_t>(expertPerRank);
        int32_t expertNumPer = curExpertId % static_cast<int32_t>(expertPerRank);
        TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint8_t> maskBuf = sendMaskTensor[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> maskBufI32 = maskBuf.ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuf, topkIdsTensor, curExpertId, AscendC::CMPMODE::EQ, compareCount);
        uint64_t sendCnt = 0;
        GatherMask(sendGatherOutTensor, topkIdsTensor, maskBufU32, true,
            static_cast<uint32_t>(sendTotalNum), {1, 1, 0, 0}, sendCnt);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        uint64_t dstOffset = maskWinOffset +
            static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(worldSize) + static_cast<int32_t>(rankId)) *
            static_cast<uint64_t>(maskSlotSize);
        dstGlobalTensor.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
        DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

__aicore__ inline void SendCntCal(
    MegaMoeModules::TileInfo tile,
    int32_t localExpertId, uint64_t &sendCnt, uint64_t &cumsumRevCntInRank,
    uint32_t worldSize, uint32_t aicNum,
    uint32_t maskSlotSize, uint32_t maskAlignSize,
    LocalTensor<uint8_t> &gatherMaskTensor,
    LocalTensor<uint32_t> &gatherMaskInt32Tensor,
    LocalTensor<int32_t> &expertTokenCntTensor,
    LocalTensor<int32_t> &cumsumInfoTensor,
    GlobalTensor<int32_t> &expertRevNumsGlobalTensor,
    GlobalTensor<int32_t> &cumsumInfoGlobalTensor,
    bool enableA8W4,
    const Params &params)
{
    sendCnt = 0;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize) / sizeof(uint32_t);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize) / sizeof(uint32_t);

    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize * maskSlotSize));
    DataCopy(gatherMaskTensor, maskSrcGlobal, worldSize * maskSlotSize);

    if (enableA8W4) {
        if (localExpertId != 0) {
            DataCopyPad(cumsumInfoTensor, cumsumInfoGlobalTensor,
                        {1U, static_cast<uint32_t>(worldSize * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                        {true, 0U, 0U, 0U});
        }
    }

    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize); ++calRankId) {
        int32_t perRankCnt = gatherMaskInt32Tensor.GetValue(calRankId * slotWordStride + countWordIdx);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        cumsumRevCntInRank += static_cast<uint64_t>(perRankCnt);
        cumsumInfoTensor.SetValue(localExpertId * worldSize + calRankId,
            static_cast<int32_t>(cumsumRevCntInRank));
    }

    expertTokenCntTensor.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor[localExpertId * INT32_PER_256B * aicNum +
        INT32_PER_256B * tile.tileIdx], expertTokenCntTensor, INT32_PER_256B);
    if (enableA8W4) {
        DataCopyPad(cumsumInfoGlobalTensor, cumsumInfoTensor,
                    {1U, static_cast<uint32_t>(worldSize * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)params.workspaceInfo.flagSendCntCalToUpdParamsPtr +
        static_cast<uint64_t>(localExpertId) * aicNum * INT_CACHELINE +
        static_cast<uint64_t>(tile.tileIdx) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

} // namespace TopKGather

#endif
