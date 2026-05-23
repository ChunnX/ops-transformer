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
 * \brief A6 CCU MoeDistributeDispatchCcuTeardown device 算法（v4，端到端 IO 契约对齐库上 op_def）。
 *
 * Algorithm 五段式：
 *   1. InitResource              + 由 epWorldSize_-1 推导 expectedPeerCount（host const，Var=Imm 合法）
 *   2. LoadArgs                  (y, expandX, assistInfo, commCmdInfo, tokenMsgLen)
 *   3. WaitAllPeersArriveNotify  0.5RTT 等齐，target = expectedPeerCountVar_ = rankSize-1
 *   4. 读 srcTokens + 整段拷贝 + 写 assist_info 前缀和
 *   5. PostSync + ClearAllPeersArriveNotify
 */

#include "moe_distribute_dispatch_ccu_teardown.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeDispatchCcuTeardown::CcuKernelArgMoeDistributeDispatchCcuTeardown(
    const std::string &groupEp_,
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    uint32_t expertShardType_, uint32_t sharedExpertNum_,
    uint32_t sharedExpertRankNum_, uint32_t quantMode_,
    uint32_t globalBs_, uint32_t expertTokenNumsType_,
    uint32_t commType_, const std::string &commAlg_)
    : groupEp(groupEp_),
      epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      expertShardType(expertShardType_), sharedExpertNum(sharedExpertNum_),
      sharedExpertRankNum(sharedExpertRankNum_), quantMode(quantMode_),
      globalBs(globalBs_ == 0 ? bs_ * epWorldSize_ : globalBs_),
      expertTokenNumsType(expertTokenNumsType_),
      commType(commType_), commAlg(commAlg_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      // v6：tokenMsgSize（非量化）= Align256(H) * sizeof(YOutType)。与 setup 一致。
      tokenMsgLen(static_cast<uint64_t>(((hiddenSize_ + 255U) / 256U) * 256U) * sizeof(uint16_t)),
      moeExpertPerRankNum(moeExpertNum_ / (epWorldSize_ - sharedExpertRankNum_))
{}

// v5：用 CcuKernelSignature::Append<T> 手工拼。
CcuKernelSignature CcuKernelArgMoeDistributeDispatchCcuTeardown::GetKernelSignature() const
{
    CcuKernelSignature sig;
    sig.Append(std::string("MoeDistributeDispatchCcuTeardown"));
    sig.Append(groupEp);
    sig.Append(epWorldSize);
    sig.Append(epRankId);
    sig.Append(moeExpertNum);
    sig.Append(expertShardType);
    sig.Append(sharedExpertNum);
    sig.Append(sharedExpertRankNum);
    sig.Append(quantMode);
    sig.Append(globalBs);
    sig.Append(expertTokenNumsType);
    sig.Append(commType);
    sig.Append(commAlg);
    sig.Append(bs);
    sig.Append(topK);
    sig.Append(hiddenSize);
    return sig;
}

// --------------------------- CcuTaskArg ----------------------------

CcuTaskArgMoeDistributeDispatchCcuTeardown::CcuTaskArgMoeDistributeDispatchCcuTeardown(
    uint64_t xAddr_, uint64_t yAddr_, uint64_t expertIdsAddr_, uint64_t commCmdInfoAddr_,
    uint64_t expandXAddr_, uint64_t dynamicScalesAddr_,
    uint64_t assistInfoForCombineAddr_, uint64_t expertTokenNumsAddr_)
    : xAddr(xAddr_), yAddr(yAddr_), expertIdsAddr(expertIdsAddr_),
      commCmdInfoAddr(commCmdInfoAddr_),
      expandXAddr(expandXAddr_), dynamicScalesAddr(dynamicScalesAddr_),
      assistInfoForCombineAddr(assistInfoForCombineAddr_),
      expertTokenNumsAddr(expertTokenNumsAddr_)
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
    // 顺序与 LoadArgs 严格一致；
    //   xAddr / expertIdsAddr / dynamicScalesAddr / expertTokenNumsAddr 端到端 IO 契约保留
    //   但 device 端 v3 实现不使用（host driver sync 后归零），不传入。
    return {
        myArg.yAddr,
        myArg.expandXAddr,
        myArg.assistInfoForCombineAddr,
        myArg.commCmdInfoAddr,
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

    // v4：expectedPeerCount 从 KernelArg.epWorldSize 推导，不再走 TaskArg。
    // host const 求值，Var = Imm 合法。
    expectedPeerCountVar_ = static_cast<uint64_t>(epWorldSize_) - 1U;

    // ============ 2. LoadArgs ============
    Load(yAddrVar_);
    Load(expandXAddrVar_);
    Load(assistInfoAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(tokenMsgLenVar_);

    // ============ 3. WaitAllPeersArriveNotify ============
    CHK_RET(WaitAllPeersArriveNotify(expectedPeerCountVar_, MOE_DISPATCH_CCU_TAG));

    // ============ 4. 读 srcTokens + 整段拷贝 + 写 assist_info 前缀和 ============
    std::vector<CcuRep::Variable> srcTokens(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        srcTokens[r] = CreateVariable();
        LoadVariable(commCmdInfoAddrVar_ + static_cast<uint64_t>(r) * sizeof(uint32_t), srcTokens[r]);
    }

    CcuRep::Variable segBytesVar = CreateVariable();
    // v7：段大小 = bs * topK * tokenMsgLen 字节（与 setup 的每段 bs*topK 槽位一致）。
    //     与 v5/v6 的 bs * tokenMsgLen 相比扩 topK 倍，覆盖最坏路由分布。
    segBytesVar = static_cast<uint64_t>(bs_) * topK_ * tokenMsgLen_;

    CcuRep::Variable accTokens = CreateVariable();
    accTokens = 0;
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        // v7：每段起点 = r * bs * topK * tokenMsgLen（与 setup BuildSlotBase 一致）
        srcAddr_.addr = yAddrVar_       + static_cast<uint64_t>(r) * bs_ * topK_ * tokenMsgLen_;
        dstAddr_.addr = expandXAddrVar_ + static_cast<uint64_t>(r) * bs_ * topK_ * tokenMsgLen_;
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
