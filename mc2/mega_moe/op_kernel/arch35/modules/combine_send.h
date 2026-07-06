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
 * \file combine_send.h
 * \brief Single-core CombineSend module: send GMM2 output results back to source cards.
 */

#ifndef MEGA_MOE_COMBINE_SEND_MODULE_H
#define MEGA_MOE_COMBINE_SEND_MODULE_H

#include "kernel_operator.h"
#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "mega_moe_combine_send.h"

using namespace AscendC;

namespace MegaMoeModules {
namespace CombineSend {

struct CombineSendState {
    Params params;
};

template <typename ElementMMadOut2>
__aicore__ inline void CombineTokens(
    CombineSendState &state, const TileInfo &tile,
    uint32_t mLoc, uint32_t nLoc, uint32_t n,
    LocalTensor<int32_t> &tripleTensor,
    LocalTensor<ElementMMadOut2> &l0cOutUbGMM2,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &actualBlockShape)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = state.params.peermemInfo.combineSendPtr - state.params.peermemInfo.rankSyncInWorldPtr;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (tokenIdx * state.params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset],
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
}

template <typename ElementMMadOut2>
__aicore__ inline void CombineTokensLayered(
    CombineSendState &state, const TileInfo &tile,
    uint32_t mLoc, uint32_t nLoc, uint32_t n,
    LocalTensor<int32_t> &tripleTensor,
    LocalTensor<ElementMMadOut2> &l0cOutUbGMM2,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &actualBlockShape)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    uint32_t actualDataLength = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    uint32_t maxDataLengthPerBlock = Ops::Base::CeilAlign(static_cast<int64_t>(MegaMoeImpl::L1_TILE_N *
                                                      sizeof(ElementMMadOut2)), (int64_t)ALIGN_32);
    uint32_t maxDataSizePerToken = Ops::Base::CeilDiv(static_cast<int64_t>(state.params.tilingData->h),
        (int64_t)MegaMoeImpl::L1_TILE_N) * (ALIGN_32 + maxDataLengthPerBlock);
    LocalTensor<int32_t> tripleTempTensor = LocalTensor<int32_t>(TPosition::VECCALC,
                                                                 MegaMoeCombineImpl::COMBINE_SEND_ADDR,
                                                                 ALIGN_32 / sizeof(int32_t));
    AscendC::GlobalTensor<ElementMMadOut2> gmLocalD;
    AscendC::GlobalTensor<int32_t> gmLocalDTriple;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = actualDataLength;

    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        if (toRankId == state.params.combineCommParams.rankId) {
            uint64_t gmDstOffset = (tokenIdx * state.params.tilingData->topK + topkIdx) * n + nLoc;
            GM_ADDR localAddr = (GM_ADDR)(state.params.peermemInfo.combineSendPtr + gmDstOffset * sizeof(ElementMMadOut2));
            gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(localAddr));
            AscendC::DataCopyPad(gmLocalD,
                l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
            continue;
        }
        tripleTempTensor.SetValue(MegaMoeCombineImpl::rankIdIndex, toRankId);
        tripleTempTensor.SetValue(MegaMoeCombineImpl::tokenIdxIndex, tokenIdx);
        tripleTempTensor.SetValue(MegaMoeCombineImpl::topkIdxIndex, topkIdx);
        tripleTempTensor.SetValue(MegaMoeCombineImpl::blockLenIndex, nLoc);
        tripleTempTensor.SetValue(MegaMoeCombineImpl::tokenLenIndex, n);
        tripleTempTensor.SetValue(MegaMoeCombineImpl::tokenActualLenIndex, actualDataLength);
        tripleTempTensor.SetValue(MegaMoeCombineImpl::flagIndex, 1);
        __gm__ int32_t* notifyAddr = (__gm__ int32_t*)(state.params.workspaceInfo.combineCommNotifyPtr +
                                                        toRankId * sizeof(int32_t));
        GM_ADDR dataAddr = state.params.workspaceInfo.combineCommDataPtr +
                        static_cast<uint64_t>(toRankId) * maxDataSizePerToken * state.params.tilingData->bs *
                        state.params.tilingData->expertPerRank;
        int32_t cnt = AscendC::AtomicAdd(notifyAddr, 1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            dataAddr + cnt * (maxDataLengthPerBlock + ALIGN_32)));
        AscendC::DataCopyPad(gmLocalD,
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        gmLocalDTriple.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            dataAddr + cnt * (maxDataLengthPerBlock + ALIGN_32) + maxDataLengthPerBlock));
        AscendC::DataCopy(gmLocalDTriple, tripleTempTensor, ALIGN_32 / sizeof(int32_t));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }
}

template <typename ElementMMadOut2>
__aicore__ inline void CombineSendTokenToRemote(CombineSendState &state, const TileInfo &tile)
{
    MegaMoeCombineImpl::CombineSendTokenToRemote<ElementMMadOut2>(state.params);
}

} // namespace CombineSend
} // namespace MegaMoeModules

#endif
