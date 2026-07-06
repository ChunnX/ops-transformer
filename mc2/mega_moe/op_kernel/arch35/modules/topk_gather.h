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
 * \brief Single-core TopKGather module: source card computes mask+count for each global expert
 *        and pushes to target cards; target card reads counts for local experts.
 */

#ifndef MEGA_MOE_TOPK_GATHER_H
#define MEGA_MOE_TOPK_GATHER_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

namespace MegaMoeModules {
namespace TopKGather {

struct TopKGatherState {
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];
    LocalTensor<int32_t> sendGatherOutTensor_;
    LocalTensor<int32_t> resetTensor_;

    uint32_t sendTotalNum;
    int32_t compareCount;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;
    uint64_t maskWinOffset;
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
    int32_t dispatchFlagSlotsPerExpert;
    uint32_t aicNum;
    Params params;
};

__aicore__ inline void BuffInit(TopKGatherState &state, const TileInfo &tile,
    const Params &params, uint32_t m, uint32_t topK, uint32_t expertPerRank,
    uint32_t worldSize, uint32_t aicNum, uint32_t rankId, uint32_t aivTotalTiles,
    int32_t dispatchFlagSlotsPerExpert)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    state.params = params;
    state.sendTotalNum = m * topK;
    state.worldSize = worldSize;
    state.expertPerRank = expertPerRank;
    state.rankId = rankId;
    state.aicNum = aicNum;
    state.dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert;

    state.compareCount = Ops::Base::CeilAlign(static_cast<int64_t>(state.sendTotalNum * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    state.maskAlignSize = Ops::Base::CeilAlign(static_cast<int64_t>(state.compareCount) / 8,
        static_cast<int64_t>(ALIGN_32));
    state.maskSlotSize = state.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    state.maskWinOffset = static_cast<uint64_t>(params.peermemInfo.maskRecvPtr -
        params.peermemInfo.rankSyncInWorldPtr);

    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(state.sendTotalNum * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    state.topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr,
        topkIdsTensorSize / sizeof(int32_t));

    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert) +
        static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum));
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(aivTotalTiles));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    state.resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr,
        resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(state.resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));

    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        state.sendMaskTensor_[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * state.maskSlotSize, state.maskSlotSize);
    }

    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * state.maskSlotSize;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(static_cast<int64_t>(state.compareCount * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    state.sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
}

__aicore__ inline void SendMaskCal(TopKGatherState &state, const TileInfo &tile)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)state.params.expertIdxGmAddr);
    Duplicate<int32_t>(state.topkIdsTensor_, 0, state.compareCount);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(state.sendTotalNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(state.topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts = static_cast<int32_t>(state.worldSize * state.expertPerRank);
    uint32_t countWordIdx = static_cast<uint32_t>(state.maskAlignSize) / sizeof(int32_t);
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(state.maskSlotSize), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (int32_t curExpertId = tile.aivTileIdx; curExpertId < totalExperts;
         curExpertId += tile.aivTotalTiles, ++iter) {
        int32_t dstRank = curExpertId / static_cast<int32_t>(state.expertPerRank);
        int32_t expertNumPer = curExpertId % static_cast<int32_t>(state.expertPerRank);
        TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint8_t> maskBuf = state.sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuf, state.topkIdsTensor_, curExpertId, AscendC::CMPMODE::EQ, state.compareCount);
        uint64_t sendCnt = 0;
        GatherMask(state.sendGatherOutTensor_, state.topkIdsTensor_, maskBufU32, true,
            static_cast<uint32_t>(state.sendTotalNum), {1, 1, 0, 0}, sendCnt);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        uint64_t dstOffset = state.maskWinOffset +
            static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(state.worldSize) + static_cast<int32_t>(state.rankId)) *
            static_cast<uint64_t>(state.maskSlotSize);
        dstGlobalTensor.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
        DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

__aicore__ inline void ResetFlagList(TopKGatherState &state, const TileInfo &tile,
    GlobalTensor<int32_t> &swigluToGmm2FlagGm)
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    swigluToGmm2FlagGm.SetGlobalBuffer((__gm__ int32_t*)state.params.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(state.expertPerRank) *
        (static_cast<int32_t>(INT_CACHELINE) + state.dispatchFlagSlotsPerExpert +
        static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(state.aicNum));
    int32_t coreLen, coreOffset;
    TilingByTile(flagNum, coreLen, coreOffset, tile.aivTileIdx, tile.aivTotalTiles, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm[coreOffset], state.resetTensor_, rankSyncCopyParams);
    }
}

__aicore__ inline void SendCntCal(TopKGatherState &state, const TileInfo &tile,
    int32_t localExpertId, uint64_t &sendCnt,
    LocalTensor<int32_t> &expertTokenCntTensor,
    LocalTensor<uint32_t> &gatherMaskInt32Tensor,
    LocalTensor<uint8_t> &gatherMaskTensor,
    LocalTensor<int32_t> &cumsumInfoTensor,
    GlobalTensor<int32_t> &expertRevNumsGlobalTensor,
    GlobalTensor<int32_t> &cumsumInfoGlobalTensor,
    uint64_t &cumsumRevCntInRank)
{
    sendCnt = 0;
    uint32_t slotWordStride = static_cast<uint32_t>(state.maskSlotSize) / sizeof(uint32_t);
    uint32_t countWordIdx = static_cast<uint32_t>(state.maskAlignSize) / sizeof(uint32_t);

    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(state.params.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * state.worldSize * state.maskSlotSize));
    DataCopy(gatherMaskTensor, maskSrcGlobal, state.worldSize * state.maskSlotSize);

    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(state.worldSize); ++calRankId) {
        int32_t perRankCnt = gatherMaskInt32Tensor.GetValue(calRankId * slotWordStride + countWordIdx);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        cumsumRevCntInRank += static_cast<uint64_t>(perRankCnt);
        cumsumInfoTensor.SetValue(localExpertId * state.worldSize + calRankId,
            static_cast<int32_t>(cumsumRevCntInRank));
    }

    expertTokenCntTensor.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor[localExpertId * INT32_PER_256B * state.aicNum +
        INT32_PER_256B * tile.tileIdx], expertTokenCntTensor, INT32_PER_256B);
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)state.params.workspaceInfo.flagSendCntCalToUpdParamsPtr +
        static_cast<uint64_t>(localExpertId) * state.aicNum * INT_CACHELINE +
        static_cast<uint64_t>(tile.tileIdx) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

} // namespace TopKGather
} // namespace MegaMoeModules

#endif
