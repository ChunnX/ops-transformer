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
 * \file moe_distribute_dispatch_ccu_teardown.cpp
 * \brief A6 CCU MoeDistributeDispatchCcuTeardown device 算法（v3）。
 *
 * v3 修复要点：
 *   - segBytes 改为 host const `bs * tokenMsgLen`（每段固定预留 bs 槽位），消除原
 *     Var × Var 违法表达式。expand_x 整体布局：[r*bs : (r+1)*bs] 每段固定 bs 槽位，
 *     未使用槽位为 0。
 *   - dst 偏移直接用 host const `r * bs * tokenMsgLen`，不再用 device 端累加器。
 *   - expertTokenNums device 端不写（原 Var/const 均分违法）；host 端 sync 后 memset 归零。
 *   - assist_info_for_combine[r] 仍写精确累计 token 数（前缀和），用于 combine 阶段精确回写。
 *
 * Algorithm 五段式：
 *   1. InitResource
 *   2. LoadArgs
 *   3. WaitAllPeersArriveNotify(peerCount, tag)   ← target 是 peer 数
 *   4. 从 commCmdInfo[srcRank] 读 srcTokens[r]，按 host const 偏移整段拷贝 + 写 assist_info
 *   5. PostSync + ClearAllPeersArriveNotify
 */

#include "moe_distribute_dispatch_ccu_teardown.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeDispatchCcuTeardown::CcuKernelArgMoeDistributeDispatchCcuTeardown(
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    std::vector<uint32_t> subCommRanks_,
    uint32_t expertTokenNumsType_,
    uint32_t expertShardType_, uint32_t sharedExpertNum_,
    uint32_t sharedExpertRankNum_, uint32_t quantMode_,
    uint32_t globalBs_, uint32_t commType_, uint32_t commAlg_)
    : epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      expertShardType(expertShardType_), sharedExpertNum(sharedExpertNum_),
      sharedExpertRankNum(sharedExpertRankNum_), quantMode(quantMode_),
      globalBs(globalBs_ == 0 ? bs_ * epWorldSize_ : globalBs_),
      commType(commType_), commAlg(commAlg_),
      expertTokenNumsType(expertTokenNumsType_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      tokenMsgLen(static_cast<uint64_t>(hiddenSize_) * sizeof(uint16_t)),
      moeExpertPerRankNum(moeExpertNum_ / (epWorldSize_ - sharedExpertRankNum_)),
      subCommRanks(std::move(subCommRanks_))
{}

std::string CcuKernelArgMoeDistributeDispatchCcuTeardown::GetKernelSignature()
{
    return GenerateCcuKernelSignature("MoeDistributeDispatchCcuTeardown",
        epWorldSize, epRankId, moeExpertNum, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, commType, commAlg,
        expertTokenNumsType, bs, topK, hiddenSize);
}

// --------------------------- CcuTaskArg ----------------------------

CcuTaskArgMoeDistributeDispatchCcuTeardown::CcuTaskArgMoeDistributeDispatchCcuTeardown(
    uint64_t xAddr_, uint64_t yAddr_, uint64_t expertIdsAddr_, uint64_t commCmdInfoAddr_,
    uint64_t expandXAddr_, uint64_t assistInfoForCombineAddr_, uint64_t expertTokenNumsAddr_,
    uint32_t expectedPeerCount_, uint64_t dynamicScalesAddr_)
    : xAddr(xAddr_), yAddr(yAddr_), expertIdsAddr(expertIdsAddr_),
      commCmdInfoAddr(commCmdInfoAddr_),
      expandXAddr(expandXAddr_), dynamicScalesAddr(dynamicScalesAddr_),
      assistInfoForCombineAddr(assistInfoForCombineAddr_),
      expertTokenNumsAddr(expertTokenNumsAddr_),
      expectedPeerCount(expectedPeerCount_)
{}

// --------------------------- CcuKernel ----------------------------

CcuKernelMoeDistributeDispatchCcuTeardown::CcuKernelMoeDistributeDispatchCcuTeardown(const CcuKernelArg &arg)
    : Mc2CcuKernelBase(arg)
{
    const auto &myArg = static_cast<const CcuKernelArgMoeDistributeDispatchCcuTeardown &>(arg);
    epWorldSize_         = myArg.epWorldSize;
    epRankId_            = myArg.epRankId;
    moeExpertPerRankNum_ = myArg.moeExpertPerRankNum;
    bs_                  = myArg.bs;
    topK_                = myArg.topK;
    tokenMsgLen_         = myArg.tokenMsgLen;
}

std::vector<uint64_t> CcuKernelMoeDistributeDispatchCcuTeardown::GeneArgs(const CcuTaskArg &arg)
{
    const auto &myArg = static_cast<const CcuTaskArgMoeDistributeDispatchCcuTeardown &>(arg);
    // expertTokenNumsAddr 端到端契约保留（host 在 sync 后 memset 归零），device 端不传入
    return {
        myArg.yAddr,
        myArg.expandXAddr,
        myArg.assistInfoForCombineAddr,
        myArg.commCmdInfoAddr,
        static_cast<uint64_t>(myArg.expectedPeerCount),
        tokenMsgLen_,
    };
}

HcclResult CcuKernelMoeDistributeDispatchCcuTeardown::Algorithm()
{
    // ============ 1. InitResource ============
    yAddrVar_                = CreateVariable();
    expandXAddrVar_          = CreateVariable();
    assistInfoAddrVar_       = CreateVariable();
    commCmdInfoAddrVar_      = CreateVariable();
    expectedPeerCountVar_    = CreateVariable();
    tokenMsgLenVar_          = CreateVariable();
    srcAddr_                 = CreateLocalAddr();
    dstAddr_                 = CreateLocalAddr();
    event_                   = CreateCompletedEvent();

    // ============ 2. LoadArgs（与 GeneArgs 严格一致）============
    Load(yAddrVar_);
    Load(expandXAddrVar_);
    Load(assistInfoAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(expectedPeerCountVar_);
    Load(tokenMsgLenVar_);

    // ============ 3. WaitAllPeersArriveNotify ============
    // 0.5RTT 等齐：本端 totalCntXn 达到 expectedPeerCount 时解除阻塞；
    // target 是 peer 数（= rankSize - 1），不是字节数。
    CHK_RET(WaitAllPeersArriveNotify(expectedPeerCountVar_, MOE_DISPATCH_CCU_TAG));

    // ============ 4. 读 srcTokens + 整段拷贝 + 写 assist_info 前缀和 ============
    // 4.1 从本卡 commCmdInfo[srcRank] 读每个 src rank 发送的 token 数（setup 已写入）。
    std::vector<CcuRep::Variable> srcTokens(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        srcTokens[r] = CreateVariable();
        LoadVariable(commCmdInfoAddrVar_ + static_cast<uint64_t>(r) * sizeof(uint32_t), srcTokens[r]);
    }

    // 4.2 每段固定 bs * tokenMsgLen 字节整段拷贝（host const 长度，无 Var*Var）。
    //     window 区按"源 rank × bs"槽位预留：第 r 段在 yAddr + r*bs*tokenMsgLen，长度固定。
    //     expand_x 同布局：第 r 段在 expandX + r*bs*tokenMsgLen，未使用槽位为 0
    //     （setup 阶段 yDev 已被 host 端 memset 为 0，未写入的槽位天然是 0）。
    CcuRep::Variable segBytesVar = CreateVariable();
    segBytesVar = static_cast<uint64_t>(bs_) * tokenMsgLen_;

    // 4.3 assist_info 前缀和按精确 srcTokens[r] 累加。
    CcuRep::Variable accTokens = CreateVariable();
    accTokens = 0;
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        srcAddr_.addr = yAddrVar_       + static_cast<uint64_t>(r) * bs_ * tokenMsgLen_;
        dstAddr_.addr = expandXAddrVar_ + static_cast<uint64_t>(r) * bs_ * tokenMsgLen_;
        CHK_RET(LocalCopyNb(dstAddr_, srcAddr_, segBytesVar, event_));
        accTokens = accTokens + srcTokens[r];
        StoreVariable(accTokens, assistInfoAddrVar_ + static_cast<uint64_t>(r) * sizeof(uint32_t));
    }

    // ============ 5. PostSync + ClearAllPeersArriveNotify ============
    event_.SetMask(0xFFFFFFFFU);
    CHK_RET(WaitEvent(event_));
    CHK_RET(ClearAllPeersArriveNotify(MOE_DISPATCH_CCU_TAG));

    return HCCL_SUCCESS;
}

} // namespace hcomm
