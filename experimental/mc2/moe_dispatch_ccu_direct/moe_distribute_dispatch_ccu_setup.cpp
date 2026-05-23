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
 * \file moe_distribute_dispatch_ccu_setup.cpp
 * \brief A6 CCU MoeDistributeDispatchCcuSetup device 算法（v3）。
 *
 * 核心机制（A6_CCU_0.5RTT.docx 流程图，image1/2/4/5）：
 *   - 发送方先 RecordExpectCount(对端, 精确字节数N, tag) 把"本卡会发 N 字节"写入对端 wishCntXn；
 *   - 发送方 WriteNbWithArriveNotify 写数据 + atomic add 累加对端 wtCount；
 *   - 硬件比对 wishCnt == wtCnt 时该 peer 状态置 1，对端 totalCntXn += 1；
 *   - 接收方 WaitAllPeersArriveNotify(peerCount, tag) 等 totalCntXn 达到 peerCount。
 *
 * v3 关键修复（绕过 CCU Variable 算术清单约束）：
 *   - 不再 device 端做 expertId / moeExpertPerRankNum 除法。host 端预算 routingTable
 *     喂入，device 端 LoadVariable 直接查表拿到 rank。
 *   - 每对端槽位偏移用 peerOccBytes_[r] 累加器逐步推进，不做 Var*const 乘法。
 *   - 第 7 阶段 scratch 用 commCmdInfo 后段 [epWorldSize, 2*epWorldSize) 槽位，每对端 r
 *     独占 commCmdInfo[epWorldSize + r]，消除原 commCmdInfo[0] 的双 race。
 *
 * Algorithm 八段式：
 *   1. InitResource
 *   2. LoadArgs
 *   3. PreSync               交换 yAddr (CKE_IDX_0) 与 commCmdInfoAddr (CKE_IDX_1)
 *   4. BuildSlotBase         构造 mySlotBaseVar / remoteSlotBaseVar[r]（host const 偏移吸入 Variable）
 *   5. PreCount              按 routingTable 统计每对端字节数 / token 数
 *   6. RecordExpectCount     用精确字节数设置对端 wishCntXn
 *   7. DoMainLoop            按 routingTable 路由 WriteNbWithArriveNotify / LocalCopyNb
 *   8. PostNotify+Sync       把 sendTokensPerPeer 跨卡写到对端 commCmdInfo[本卡rankId]，
 *                            等本地完成
 */

#include "moe_distribute_dispatch_ccu_setup.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeDispatchCcuSetup::CcuKernelArgMoeDistributeDispatchCcuSetup(
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    std::vector<uint32_t> subCommRanks_,
    uint32_t expertShardType_, uint32_t sharedExpertNum_,
    uint32_t sharedExpertRankNum_, uint32_t quantMode_,
    uint32_t globalBs_, uint32_t commType_, uint32_t commAlg_)
    : epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      expertShardType(expertShardType_), sharedExpertNum(sharedExpertNum_),
      sharedExpertRankNum(sharedExpertRankNum_), quantMode(quantMode_),
      globalBs(globalBs_ == 0 ? bs_ * epWorldSize_ : globalBs_),
      commType(commType_), commAlg(commAlg_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      tokenMsgLen(static_cast<uint64_t>(hiddenSize_) * sizeof(uint16_t)),
      moeExpertPerRankNum(moeExpertNum_ / (epWorldSize_ - sharedExpertRankNum_)),
      subCommRanks(std::move(subCommRanks_))
{}

std::string CcuKernelArgMoeDistributeDispatchCcuSetup::GetKernelSignature()
{
    return GenerateCcuKernelSignature("MoeDistributeDispatchCcuSetup",
        epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, commType, commAlg,
        bs, topK, hiddenSize);
}

// --------------------------- CcuTaskArg ----------------------------

CcuTaskArgMoeDistributeDispatchCcuSetup::CcuTaskArgMoeDistributeDispatchCcuSetup(
    uint64_t xAddr_, uint64_t expertIdsAddr_, uint64_t routingTableAddr_,
    uint64_t yAddr_, uint64_t expandIdxAddr_, uint64_t commCmdInfoAddr_,
    uint64_t scalesAddr_, uint64_t xActiveMaskAddr_)
    : xAddr(xAddr_), expertIdsAddr(expertIdsAddr_),
      scalesAddr(scalesAddr_), xActiveMaskAddr(xActiveMaskAddr_),
      routingTableAddr(routingTableAddr_),
      yAddr(yAddr_), expandIdxAddr(expandIdxAddr_), commCmdInfoAddr(commCmdInfoAddr_)
{}

// --------------------------- CcuKernel ----------------------------

CcuKernelMoeDistributeDispatchCcuSetup::CcuKernelMoeDistributeDispatchCcuSetup(const CcuKernelArg &arg)
    : Mc2CcuKernelBase(arg)
{
    const auto &myArg = static_cast<const CcuKernelArgMoeDistributeDispatchCcuSetup &>(arg);
    epWorldSize_         = myArg.epWorldSize;
    epRankId_            = myArg.epRankId;
    moeExpertPerRankNum_ = myArg.moeExpertPerRankNum;
    bs_                  = myArg.bs;
    topK_                = myArg.topK;
    tokenMsgLen_         = myArg.tokenMsgLen;
}

std::vector<uint64_t> CcuKernelMoeDistributeDispatchCcuSetup::GeneArgs(const CcuTaskArg &arg)
{
    const auto &myArg = static_cast<const CcuTaskArgMoeDistributeDispatchCcuSetup &>(arg);
    // 顺序与 LoadArgs 严格一致；expertIdsAddr 端到端契约保留但 device 端不使用，故不传入
    return {
        myArg.xAddr,
        myArg.yAddr,
        myArg.expandIdxAddr,
        myArg.commCmdInfoAddr,
        myArg.routingTableAddr,
        tokenMsgLen_,
    };
}

HcclResult CcuKernelMoeDistributeDispatchCcuSetup::Algorithm()
{
    // ============ 1. InitResource ============
    xAddrVar_           = CreateVariable();
    yAddrVar_           = CreateVariable();
    expandIdxAddrVar_   = CreateVariable();
    commCmdInfoAddrVar_ = CreateVariable();
    routingTableAddrVar_ = CreateVariable();
    tokenMsgLenVar_     = CreateVariable();
    localSrc_           = CreateLocalAddr();
    localDst_           = CreateLocalAddr();
    remoteDst_          = CreateRemoteAddr();
    event_              = CreateCompletedEvent();

    // 对端地址寄存器（PreSync 阶段填充）
    remoteYAddrs_.reserve(epWorldSize_);
    std::vector<CcuRep::Variable> remoteCommCmdInfoAddrs(epWorldSize_);
    std::vector<CcuRep::Variable> sendBytesPerPeer(epWorldSize_);
    std::vector<CcuRep::Variable> sendTokensPerPeer(epWorldSize_);
    peerOccurCnts_.reserve(epWorldSize_);
    peerOccBytes_.reserve(epWorldSize_);

    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) {
            remoteYAddrs_.push_back(CreateVariable());      // 占位，LoadArgs 后填本卡 yAddrVar_
            remoteCommCmdInfoAddrs[r] = CreateVariable();   // 占位
        } else {
            uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
            CcuRep::Variable vy, vc;
            CHK_RET(CreateVariable(channels_[channelIdx], CKE_IDX_0, &vy));
            CHK_RET(CreateVariable(channels_[channelIdx], CKE_IDX_1, &vc));
            remoteYAddrs_.push_back(vy);
            remoteCommCmdInfoAddrs[r] = vc;
        }
        CcuRep::Variable occ = CreateVariable(); occ = 0; peerOccurCnts_.push_back(occ);
        CcuRep::Variable ob  = CreateVariable(); ob  = 0; peerOccBytes_.push_back(ob);
        CcuRep::Variable sb  = CreateVariable(); sb  = 0; sendBytesPerPeer[r]  = sb;
        CcuRep::Variable st  = CreateVariable(); st  = 0; sendTokensPerPeer[r] = st;
    }

    // ============ 2. LoadArgs ============
    Load(xAddrVar_);
    Load(yAddrVar_);
    Load(expandIdxAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(routingTableAddrVar_);
    Load(tokenMsgLenVar_);
    remoteYAddrs_[epRankId_] = yAddrVar_;
    remoteCommCmdInfoAddrs[epRankId_] = commCmdInfoAddrVar_;

    // ============ 3. PreSync ============
    uint32_t selfBit = 1U << epRankId_;
    uint32_t allBits = ((1U << epWorldSize_) - 1U) & (~selfBit);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_0, CKE_IDX_0, yAddrVar_,           selfBit));
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_1, CKE_IDX_1, commCmdInfoAddrVar_, selfBit));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_IDX_0, allBits));
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_IDX_1, allBits));
    }

    // ============ 4. BuildSlotBase ============
    // 本卡发数据到对端 window 时，槽位起点 = 对端 yAddr + 本卡 epRankId * bs * tokenMsgLen。
    // 这个偏移在 device 端只能通过 Variable 表达；先放进 myBaseOffVar，再 Var+Var 吸入基址。
    CcuRep::Variable myBaseOffVar = CreateVariable();
    myBaseOffVar = static_cast<uint64_t>(epRankId_) * bs_ * tokenMsgLen_;   // Var = Imm，合法
    CcuRep::Variable mySlotBaseVar = CreateVariable();
    mySlotBaseVar = yAddrVar_ + myBaseOffVar;                               // Var = Var + Var，合法
    std::vector<CcuRep::Variable> remoteSlotBaseVar(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        CcuRep::Variable v = CreateVariable();
        v = remoteYAddrs_[r] + myBaseOffVar;                                // Var = Var + Var
        remoteSlotBaseVar[r] = v;
    }

    // ============ 5. PreCount ============
    // routingTable 已由 host 预算（routingTable[idx] = expertIds[idx] / moeExpertPerRankNum），
    // device 端直接 LoadVariable 查表，无需任何除法运算。
    CcuRep::Variable rankVarA = CreateVariable();
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            uint32_t idx = i * topK_ + j;
            LoadVariable(routingTableAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t), rankVarA);
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarA == r) {
                    sendBytesPerPeer[r]  = sendBytesPerPeer[r] + tokenMsgLenVar_;
                    sendTokensPerPeer[r] = sendTokensPerPeer[r] + 1U;
                }
            }
        }
    }

    // ============ 6. RecordExpectCount ============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(RecordExpectCount(channels_[channelIdx], sendBytesPerPeer[r], MOE_DISPATCH_CCU_TAG));
    }

    // ============ 7. DoMainLoop ============
    // 同样查 routingTable 拿到 rank；槽位偏移用累加器 peerOccBytes_[r]，无 Var*const。
    CcuRep::Variable rankVarB  = CreateVariable();
    CcuRep::Variable curOccVar = CreateVariable();
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            uint32_t idx = i * topK_ + j;
            LoadVariable(routingTableAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t), rankVarB);
            localSrc_.addr = xAddrVar_ + static_cast<uint64_t>(i) * tokenMsgLen_;   // Var + const
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarB == r) {
                    curOccVar = peerOccurCnts_[r];
                    StoreVariable(curOccVar,
                                  expandIdxAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t));
                    peerOccurCnts_[r] = curOccVar + 1U;
                    if (r == epRankId_) {
                        localDst_.addr = mySlotBaseVar + peerOccBytes_[r];          // Var + Var
                        CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenMsgLenVar_, event_));
                    } else {
                        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
                        remoteDst_.addr = remoteSlotBaseVar[r] + peerOccBytes_[r];  // Var + Var
                        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                         localSrc_, tokenMsgLenVar_,
                                                         MOE_DISPATCH_CCU_TAG));
                    }
                    peerOccBytes_[r] = peerOccBytes_[r] + tokenMsgLenVar_;          // Var += Var
                }
            }
        }
    }

    // ============ 8. PostNotify+Sync ============
    // 把 sendTokensPerPeer[r] 跨卡写到对端 commCmdInfo[本卡epRankId]，供 teardown 整理时
    // 知道每个 src rank 实际发了多少 token。
    //   - scratch 槽位：commCmdInfo[epWorldSize + r]（commCmdInfo 后段独占，避免与 src rank
    //     写本卡正式槽位 commCmdInfo[srcRank] 冲突）。
    //   - 用普通 WriteNb（不走 0.5RTT，避免污染 wtCount 计数）。
    CcuRep::LocalAddr localTokenBuf = CreateLocalAddr();
    CcuRep::Variable elemBytesVar = CreateVariable();
    elemBytesVar = static_cast<uint64_t>(sizeof(uint32_t));
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        uint64_t scratchOff = static_cast<uint64_t>(epWorldSize_ + r) * sizeof(uint32_t);
        StoreVariable(sendTokensPerPeer[r], commCmdInfoAddrVar_ + scratchOff);
        localTokenBuf.addr = commCmdInfoAddrVar_ + scratchOff;
        remoteDst_.addr    = remoteCommCmdInfoAddrs[r]
                             + static_cast<uint64_t>(epRankId_) * sizeof(uint32_t);
        CHK_RET(WriteNb(channels_[channelIdx], remoteDst_, localTokenBuf, elemBytesVar, event_));
    }

    event_.SetMask(0xFFFFFFFFU);
    CHK_RET(WaitEvent(event_));

    // 本卡发本卡的 token 数也写进本地 commCmdInfo[本卡rankId]，让 teardown 统一从 commCmdInfo 读
    StoreVariable(sendTokensPerPeer[epRankId_],
                  commCmdInfoAddrVar_ + static_cast<uint64_t>(epRankId_) * sizeof(uint32_t));

    return HCCL_SUCCESS;
}

} // namespace hcomm
