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
 * \file mega_moe_combine_send.h
 * \brief
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape); // 当前token的数
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;//本卡combineSendPtr的偏移量
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2); // N_VALUE是当前tile块的n长度
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset))); // 设置目标rank的combineSendPtr
        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset],
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
}

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokensWithUrma( // 先这样写
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape); // 当前token的数
    uint32_t actualDataLength = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    uint32_t maxDataLength = L1_TILE_N * sizeof(ElementMMadOut2); // 32b对齐
    uint32_t max_data_size = (Ops::Base::CeilDiv(params.tilingData->h, L1_TILE_N) * 32 +  params.tilingData->h * sizeof(ElementMMadOut2)) * params.tilingData->bs * params.tilingData->expertPerRank;
    LocalTensor<int32_t> tripleTempTensor = LocalTensor<int32_t>(TPosition::VECCALC, 200 * 1024 + lenTile * 8, 32); // 三元组
    AscendC::GlobalTensor<ElementMMadOut2> gmLocalD;
    AscendC::GlobalTensor<int32_t> gmLocalDTriple;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;//每张卡combineSendPtr的偏移量
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2); // N_VALUE是当前tile块的n长度,即每一小块的长度
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        tripleTempTensor.SetValue(0, toRankId);
        tripleTempTensor.SetValue(1, tokenIdx);
        tripleTempTensor.SetValue(2, topkIdx);
        tripleTempTensor.SetValue(3, nLoc);
        tripleTempTensor.SetValue(4, n);
        tripleTempTensor.SetValue(5, actualDataLength);
        tripleTempTensor.SetValue(7, 1); // 标记为已处理
        GM_ADDR notifyAddr = params.combineNotifyPtr + params.tilingData->bs * params.tilingData->h * sizeof() + toRankId * sizeof(int32_t);
        int32_t cnt = AscendC::AtomicAdd(reinterpret_cast<int32_t*>(notifyAddr), 1);
        GM_ADDR dataAddr = params.workspaceInfo.combineDataPtr + toRankId * max_data_size;
        gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dataAddr + cnt * (maxDataLength + 32))); // 设置目标rank的combineSendPtr
        AscendC::DataCopyPad(gmLocalD,
                l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dataAddr + cnt * (maxDataLength + 32) + maxDataLength)); // 设置目标rank的combineSendPtr
        AscendC::DataCopy(gmLocalD, tripleTempTensor, 8);
    }
}



template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokensWithUrma(uint32_t gsendCnt, const Params& params)
{
    LocalTensor<int32_t> sendTensor = LocalTensor<int32_t>(TPosition::VECCALC, 0, 32); // 三元组
    uint32_t startRankId, endRankId, processRankNum;
    SplitToCore(aiv_num, epWorldSize_, startRankId, endRankId, processRankNum);
    if(startRankId >= endRankId || processRankNum == 0) {
        return ;
    }
    LocalTensor<int32_t> sendTokenCnt = LocalTensor<int32_t>(TPosition::VECCALC, 32, processRankNum * sizeof(uint32_t)); // 三元组
    LocalTensor<int32_t> totalTokenCnt = LocalTensor<int32_t>(TPosition::VECCALC, 32 + processRankNum * sizeof(uint32_t), processRankNum * sizeof(uint32_t)); // 三元组

    // maskRecvPtr布局: [localExpert][srcRank], 每个槽位 = maskAlignSize + 32B count
    int64_t sendTotalNum = static_cast<int64_t>(params.tilingData->bs) * params.tilingData->topK;
    int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t),
        (int64_t)ALIGN_256) / (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
    int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize / sizeof(int32_t));
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize / sizeof(int32_t));
    uint32_t expertPerRank = params.tilingData->expertPerRank;
    uint32_t worldSize = params.tilingData->epWorldSize;
    LocalTensor<uint32_t> slotReadTensor = LocalTensor<uint32_t>(TPosition::VECCALC,
        32 + processRankNum * sizeof(uint32_t) + processRankNum * sizeof(uint32_t), slotWordStride);
    AscendC::GlobalTensor<uint8_t> maskSrcGlobal;

    for (uint32_t index = startRankId; index < endRankId; index++) {
        uint32_t cnt = 0;
        for (uint32_t expertIdx = 0; expertIdx < expertPerRank; expertIdx++) {
            maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params.peermemInfo.maskRecvPtr +
                static_cast<uint64_t>(expertIdx) * worldSize * maskSlotSize +
                static_cast<uint64_t>(index) * maskSlotSize));
            AscendC::DataCopy(slotReadTensor, maskSrcGlobal, slotWordStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            cnt += slotReadTensor.GetValue(countWordIdx);
        }
        totalTokenCnt.SetValue(index - startRankId, cnt);
    }
    uint32_t rankIndex = 0;
    uint32_t processRank = 0;
    while (true) {
        while (sendTokenCnt(rankIndex) == totalTokenCnt(rankIndex)) {
            rankIndex = (rankIndex + 1) % processRankNum;
        }
        GM_ADDR notifyAddr = params.combineNotifyPtr + params.tilingData->bs * params.tilingData->h * sizeof() + (rankIndex + startRankId) * sizeof(int32_t);
        uint32_t curCnt  = ReadGmByPassDCache(notifyAddr);
        if (curCnt > sendTokenCnt(rankIndex)) {
            uint32_t needCnt = curCnt - sendTokenCnt(rankIndex);
            for (uint32_t i = 0; i < needCnt; i++) {
                WriteNbi()
            }
            sendTokenCnt.SetValue(rankIndex, curCnt);
            if (sendTokenCnt(rankIndex) == totalTokenCnt(rankIndex)) {
                processRank++;
                if (processRank == processRankNum) {
                    break;
                }
            }
        }
        rankIndex = (rankIndex + 1) % processRankNum;
    }
}


}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H