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
 * \brief MegaMoe Combine 发送逻辑: peermem 版(CombineTokens) 与 urma 版(CombineTokensWithUrma)
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {

// ==================================================================================================
// CombineTokens (peermem 版): GMM2 计算完成后直接通过 peermem 写远程卡的 combineSendPtr
// ==================================================================================================
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape); // 当前token的数
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
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
        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset],
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
}

// ==================================================================================================
// CombineTokensWithUrma (GMM2 内部调用版): 写本地 GM workspace + AtomicAdd notify
// --------------------------------------------------------------------------------------------------
//   将 GMM2 结果写到本地 workspace 中转区(combineDataPtr), 并写三元组元数据;
//   AtomicAdd 递增目标 rank 的 notify 计数, 供接收端轮询;
//   注意: 此函数仅写本地 GM, 实际 urma 发送由 CombineTokensWithUrmaSend 完成。
// ==================================================================================================
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokensWithUrma(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    uint32_t actualDataLength = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    uint32_t maxDataLength = L1_TILE_N * sizeof(ElementMMadOut2); // 32b 对齐
    // 每个 rank 的 workspace 大小: bs * expertPerRank 个 token, 每个 token 含数据+三元组(32B)
    uint32_t maxDataSize = (Ops::Base::CeilDiv(params.tilingData->h, L1_TILE_N) * 32 +
        params.tilingData->h * sizeof(ElementMMadOut2)) * params.tilingData->bs * params.tilingData->expertPerRank;
    LocalTensor<int32_t> tripleTempTensor = LocalTensor<int32_t>(TPosition::VECCALC,
        200 * 1024 + lenTile * 8, 32);
    AscendC::GlobalTensor<ElementMMadOut2> gmLocalD;
    AscendC::GlobalTensor<int32_t> gmLocalDTriple;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);
        // 组装三元组元数据 (toRankId, tokenIdx, topkIdx, nLoc, n, dataLen, _, flag)
        tripleTempTensor.SetValue(0, toRankId);
        tripleTempTensor.SetValue(1, tokenIdx);
        tripleTempTensor.SetValue(2, topkIdx);
        tripleTempTensor.SetValue(3, nLoc);
        tripleTempTensor.SetValue(4, n);
        tripleTempTensor.SetValue(5, actualDataLength);
        tripleTempTensor.SetValue(7, 1); // 标记为已处理
        // notify 地址: workspace combineNotifyPtr + 目标 rank 偏移
        // 修正: params.workspaceInfo.combineNotifyPtr (原代码误用 params.combineNotifyPtr, sizeof() 空参)
        GM_ADDR notifyAddr = params.workspaceInfo.combineNotifyPtr + toRankId * sizeof(int32_t);
        int32_t cnt = AscendC::AtomicAdd(reinterpret_cast<int32_t*>(notifyAddr), 1);
        // 数据区: 每个 rank 独立空间, 每个槽位 = maxDataLength(数据) + 32B(三元组)
        GM_ADDR dataAddr = params.workspaceInfo.combineDataPtr + toRankId * maxDataSize;
        gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            dataAddr + cnt * (maxDataLength + 32)));
        AscendC::DataCopyPad(gmLocalD,
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
        // 三元组区: 紧跟数据之后
        gmLocalDTriple.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            dataAddr + cnt * (maxDataLength + 32) + maxDataLength));
        AscendC::DataCopy(gmLocalDTriple, tripleTempTensor, 8);
    }
}

// ==================================================================================================
// CombineTokensWithUrmaSend (接收端轮询 + urma 发送): 在 GMM2 全部完成后调用
// --------------------------------------------------------------------------------------------------
//   Phase 1: 分核, 按 aivNum 均分 worldSize 个目标 rank;
//   Phase 2: 逐 rank 读 mask count 累加得到该 rank 应收的 token 总数;
//   Phase 3: 轮询 combineNotifyPtr, 收到新数据后读三元组, 通过 urma WriteNbi 发送到目标卡 combineSend 区;
//   Phase 4: 全部发送完成后 drain 所有 urma channel;
// ==================================================================================================
template <typename ElementMMadOut2>
__aicore__ inline void CombineTokensWithUrmaSend(
    const Params& params, __gm__ Mc2MoeContext* mc2Context,
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP>& hcomm, bool hcommInited, uint32_t rankId)
{
    uint32_t maxDataLength = L1_TILE_N * sizeof(ElementMMadOut2);
    uint32_t maxDataSize = (Ops::Base::CeilDiv(params.tilingData->h, L1_TILE_N) * 32 +
        params.tilingData->h * sizeof(ElementMMadOut2)) * params.tilingData->bs * params.tilingData->expertPerRank;
    LocalTensor<int32_t> sendTensor = LocalTensor<int32_t>(TPosition::VECCALC, 0, 32);

    // 分核: 按 aiv 核数均分 worldSize 个 rank (替代不存在的 SplitToCore)
    uint32_t aivIdx = GetBlockIdx();
    uint32_t aivNum = params.combineSendInfo.aivNum;
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t rankPerCore = worldSize / aivNum;
    uint32_t remainder = worldSize % aivNum;
    uint32_t startRankId = rankPerCore * aivIdx + (aivIdx < remainder ? aivIdx : remainder);
    uint32_t processRankNum = rankPerCore + (aivIdx < remainder ? 1 : 0);
    uint32_t endRankId = startRankId + processRankNum;
    if (startRankId >= endRankId || processRankNum == 0) {
        return;
    }

    LocalTensor<int32_t> sendTokenCnt = LocalTensor<int32_t>(TPosition::VECCALC,
        TRIPLE_SIZE, processRankNum * sizeof(uint32_t));
    LocalTensor<int32_t> totalTokenCnt = LocalTensor<int32_t>(TPosition::VECCALC,
        TRIPLE_SIZE + processRankNum * sizeof(uint32_t), processRankNum * sizeof(uint32_t));

    // maskRecvPtr 布局: [localExpert][srcRank], 每个槽位 = maskAlignSize + 32B count
    int64_t sendTotalNum = static_cast<int64_t>(params.tilingData->bs) * params.tilingData->topK;
    int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t),
        (int64_t)ALIGN_256) / (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
    int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize / sizeof(int32_t));
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize / sizeof(int32_t));
    uint32_t expertPerRank = params.tilingData->expertPerRank;
    LocalTensor<uint32_t> slotReadTensor = LocalTensor<uint32_t>(TPosition::VECCALC,
        TRIPLE_SIZE + processRankNum * sizeof(uint32_t) * 2, slotWordStride);
    AscendC::GlobalTensor<uint8_t> maskSrcGlobal;

    // Phase 2: 逐 rank 统计应收 token 总数 (从 mask count 累加)
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
        // 修正: CeilAlign 返回类型与 L1_TILE_N 一致, 用 uint32_t
        uint32_t totalDataBlockCnt = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(cnt) * params.tilingData->h,
            static_cast<int64_t>(L1_TILE_N)));
        totalTokenCnt.SetValue(index - startRankId, totalDataBlockCnt);
    }

    // Phase 3: 轮询 notify, 收到后 urma 发送
    uint32_t rankIndex = 0;
    uint32_t processRank = 0;
    AscendC::GlobalTensor<int32_t> gmLocalDTriple;
    AscendC::GlobalTensor<ElementMMadOut2> gmLocalDataRead;
    // combineSendPtr 相对 win 基址的偏移 (远程目标地址用)
    uint64_t combineWinOffset = static_cast<uint64_t>(params.peermemInfo.combineSendPtr -
        params.peermemInfo.rankSyncInWorldPtr);

    while (true) {
        // 跳过已完成的 rank
        // 修正: 用 GetValue 访问 LocalTensor (原代码误用 sendTokenCnt(rankIndex))
        while (sendTokenCnt.GetValue(rankIndex) == totalTokenCnt.GetValue(rankIndex)) {
            rankIndex = (rankIndex + 1) % processRankNum;
        }
        // 读 notify count
        GM_ADDR notifyAddr = params.workspaceInfo.combineNotifyPtr + (rankIndex + startRankId) * sizeof(int32_t);
        uint32_t curCnt = ReadGmByPassDCache(notifyAddr);
        if (curCnt > sendTokenCnt.GetValue(rankIndex)) {
            uint32_t needCnt = curCnt - sendTokenCnt.GetValue(rankIndex);
            uint32_t index = 0;
            // 轮询所有已到达的槽位
            while (needCnt > 0) {
                GM_ADDR dataAddr = params.workspaceInfo.combineDataPtr +
                    (rankIndex + startRankId) * maxDataSize + index * (maxDataLength + 32);
                gmLocalDTriple.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dataAddr + maxDataLength));
                AscendC::Nop<20>();
                AscendC::DataCopy(sendTensor, gmLocalDTriple, 8);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                if (sendTensor.GetValue(7) == 1) {
                    // 读取三元组元数据
                    uint32_t toRankId = sendTensor.GetValue(0);
                    uint32_t tokenIdx = sendTensor.GetValue(1);
                    uint32_t topkIdx = sendTensor.GetValue(2);
                    uint32_t nLoc = sendTensor.GetValue(3);
                    uint32_t n = sendTensor.GetValue(4);
                    uint32_t dataLen = sendTensor.GetValue(5);
                    if (hcommInited && toRankId != rankId) {
                        // urma 发送: 本地 GM 数据 → 远程卡 combineSendPtr
                        uint64_t channelHandle = GetUrmaCommHandle(mc2Context, toRankId, rankId);
                        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;
                        GM_ADDR remoteAddr = GetUrmaAddrByRankId(mc2Context, toRankId,
                            combineWinOffset + gmDstOffset * sizeof(ElementMMadOut2));
                        hcomm.WriteNbi(channelHandle, remoteAddr, dataAddr, dataLen);
                    }
                    // 标记已处理, 避免重复发送
                    sendTensor.SetValue(7, 0);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    AscendC::DataCopy(gmLocalDTriple, sendTensor, 8);
                    needCnt--;
                }
                // 修正: 轮询所有槽位, 用 curCnt 作为模数 (原代码 needCnt 会变为0导致除零)
                index = (index + 1) % curCnt;
            }
            sendTokenCnt.SetValue(rankIndex, curCnt);
            if (curCnt == totalTokenCnt.GetValue(rankIndex)) {
                processRank++;
                if (processRank == processRankNum) {
                    break;
                }
            }
        }
        rankIndex = (rankIndex + 1) % processRankNum;
    }

    // Phase 4: drain 所有 urma 发送
    if (hcommInited) {
        for (uint32_t dr = 0; dr < worldSize; ++dr) {
            if (dr == rankId) {
                continue;
            }
            hcomm.Drain(GetUrmaCommHandle(mc2Context, dr, rankId));
        }
    }
}

}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
