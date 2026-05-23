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
 * \brief A6 CCU MoeDistributeDispatchCcuSetup device 算法（v4，端到端 IO 契约对齐库上 op_def）。
 *
 * Algorithm 八段式（机制不变）：
 *   1. InitResource
 *   2. LoadArgs                顺序 (x, y, expandIdx, commCmdInfo, routingTable, tokenMsgLen)
 *   3. PreSync                 交换 yAddr (CKE_IDX_0) 与 commCmdInfoAddr (CKE_IDX_1)
 *   4. BuildSlotBase           mySlotBaseVar / remoteSlotBaseVar[r]
 *   5. PreCount                按 routingTable 统计每对端字节数 / token 数
 *   6. RecordExpectCount       0.5RTT 用精确字节数设置对端 wishCntXn
 *   7. DoMainLoop              0.5RTT WriteNbWithArriveNotify / 自发自 LocalCopyNb
 *   8. PostNotify+Sync         跨卡写 commCmdInfo + 等本地完成
 */

#include "moe_distribute_dispatch_ccu_setup.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeDispatchCcuSetup::CcuKernelArgMoeDistributeDispatchCcuSetup(
    const std::string &groupEp_,
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    uint32_t expertShardType_, uint32_t sharedExpertNum_,
    uint32_t sharedExpertRankNum_, uint32_t quantMode_,
    uint32_t globalBs_, uint32_t commType_,
    const std::string &commAlg_)
    : groupEp(groupEp_),
      epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      expertShardType(expertShardType_), sharedExpertNum(sharedExpertNum_),
      sharedExpertRankNum(sharedExpertRankNum_), quantMode(quantMode_),
      globalBs(globalBs_ == 0 ? bs_ * epWorldSize_ : globalBs_),
      commType(commType_), commAlg(commAlg_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      // v6：tokenMsgSize（非量化）= Align256(H) * sizeof(YOutType)；对 H=7168/4096 已 256 对齐时不变。
      // 来源：aclnnMoeDistributeDispatchSetup.md §约束说明 5（tokenMsgSize）。
      tokenMsgLen(static_cast<uint64_t>(((hiddenSize_ + 255U) / 256U) * 256U) * sizeof(uint16_t)),
      moeExpertPerRankNum(moeExpertNum_ / (epWorldSize_ - sharedExpertRankNum_))
{}

// v5：用 CcuKernelSignature::Append<T> 手工拼，attr 顺序严格对齐 op_def 注册顺序。
//   `GenerateCcuKernelSignature` 函数在 hcomm 仓不存在；改为手工 Append。
CcuKernelSignature CcuKernelArgMoeDistributeDispatchCcuSetup::GetKernelSignature() const
{
    CcuKernelSignature sig;
    sig.Append(std::string("MoeDistributeDispatchCcuSetup"));
    sig.Append(groupEp);
    sig.Append(epWorldSize);
    sig.Append(epRankId);
    sig.Append(moeExpertNum);
    sig.Append(expertShardType);
    sig.Append(sharedExpertNum);
    sig.Append(sharedExpertRankNum);
    sig.Append(quantMode);
    sig.Append(globalBs);
    sig.Append(commType);
    sig.Append(commAlg);
    sig.Append(bs);
    sig.Append(topK);
    sig.Append(hiddenSize);
    return sig;
}

// --------------------------- CcuTaskArg ----------------------------

CcuTaskArgMoeDistributeDispatchCcuSetup::CcuTaskArgMoeDistributeDispatchCcuSetup(
    uint64_t xAddr_, uint64_t expertIdsAddr_,
    uint64_t scalesAddr_, uint64_t xActiveMaskAddr_,
    uint64_t yAddr_, uint64_t expandIdxAddr_, uint64_t commCmdInfoAddr_,
    uint64_t routingTableAddr_)
    : xAddr(xAddr_), expertIdsAddr(expertIdsAddr_),
      scalesAddr(scalesAddr_), xActiveMaskAddr(xActiveMaskAddr_),
      yAddr(yAddr_), expandIdxAddr(expandIdxAddr_), commCmdInfoAddr(commCmdInfoAddr_),
      routingTableAddr(routingTableAddr_)
{}

// --------------------------- CcuKernel ----------------------------

CcuKernelMoeDistributeDispatchCcuSetup::CcuKernelMoeDistributeDispatchCcuSetup(const CcuKernelArg &arg)
    : Mc2CcuKernelBase(arg)
{
    const auto &myArg = static_cast<const CcuKernelArgMoeDistributeDispatchCcuSetup &>(arg);
    epWorldSize_         = myArg.epWorldSize;
    epRankId_            = myArg.epRankId;
    moeExpertNum_        = myArg.moeExpertNum;                // v8 新增缓存
    moeExpertPerRankNum_ = myArg.moeExpertPerRankNum;
    bs_                  = myArg.bs;
    topK_                = myArg.topK;
    tokenMsgLen_         = myArg.tokenMsgLen;
}

std::vector<uint64_t> CcuKernelMoeDistributeDispatchCcuSetup::GeneArgs(const CcuTaskArg &arg)
{
    const auto &myArg = static_cast<const CcuTaskArgMoeDistributeDispatchCcuSetup &>(arg);
    // v8：expertIdsAddr 重新加入 GeneArgs（device 端 DoMainLoop 阶段 A 需要读 expertId
    //     做 per-expert expand_idx 累计计数，对齐库上语义"给同一专家发送的 token 个数"）。
    return {
        myArg.xAddr,
        myArg.expertIdsAddr,
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
    xAddrVar_            = CreateVariable();
    expertIdsAddrVar_    = CreateVariable();                  // v8 ★
    yAddrVar_            = CreateVariable();
    expandIdxAddrVar_    = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    routingTableAddrVar_ = CreateVariable();
    tokenMsgLenVar_      = CreateVariable();
    localSrc_            = CreateLocalAddr();
    localDst_            = CreateLocalAddr();
    remoteDst_           = CreateRemoteAddr();
    event_               = CreateCompletedEvent();

    remoteYAddrs_.reserve(epWorldSize_);
    std::vector<CcuRep::Variable> remoteCommCmdInfoAddrs(epWorldSize_);
    std::vector<CcuRep::Variable> sendBytesPerPeer(epWorldSize_);
    std::vector<CcuRep::Variable> sendTokensPerPeer(epWorldSize_);
    peerOccBytes_.reserve(epWorldSize_);

    // v8 ★ per-expert 计数器（对齐库上 expand_idx "给同一专家发送的 token 个数"语义）
    expertOccCnts_.reserve(moeExpertNum_);
    for (uint32_t e = 0; e < moeExpertNum_; ++e) {
        CcuRep::Variable cnt = CreateVariable();
        cnt = 0;
        expertOccCnts_.push_back(cnt);
    }

    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) {
            remoteYAddrs_.push_back(CreateVariable());
            remoteCommCmdInfoAddrs[r] = CreateVariable();
        } else {
            uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
            CcuRep::Variable vy, vc;
            CHK_RET(CreateVariable(channels_[channelIdx], CKE_IDX_0, &vy));
            CHK_RET(CreateVariable(channels_[channelIdx], CKE_IDX_1, &vc));
            remoteYAddrs_.push_back(vy);
            remoteCommCmdInfoAddrs[r] = vc;
        }
        // v8：peerOccurCnts_ 已删除（v7 误用 per-rank 计数写 expand_idx，v8 改用 expertOccCnts_ per-expert）
        CcuRep::Variable ob  = CreateVariable(); ob  = 0; peerOccBytes_.push_back(ob);
        CcuRep::Variable sb  = CreateVariable(); sb  = 0; sendBytesPerPeer[r]  = sb;
        CcuRep::Variable st  = CreateVariable(); st  = 0; sendTokensPerPeer[r] = st;
    }

    // ============ 2. LoadArgs ============
    Load(xAddrVar_);
    Load(expertIdsAddrVar_);                                  // v8 ★ 顺序与 GeneArgs 一致
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
    // oneVar：替代 "Var = Var + 1U" 中的立即数（CCU 严格清单只支持 Var = Var + Var；
    // Var = Var + Imm 不被支持，需用 helper Variable 绕开）。
    CcuRep::Variable oneVar = CreateVariable();
    oneVar = 1U;
    // v7：每段大小 = bs * topK 槽位（严格 worst-case：本卡 bs*topK 副本可能全部路由到同一对端）。
    //     之前 v5/v6 用 "每段 bs 槽位" 是个隐含假设——只有当所有 topK 副本均匀分布到不同 rank
    //     时才不溢出。一旦 topK > rankSize 或路由极不均衡（router 把所有 topK 都路由到一个 rank），
    //     按 bs 段长度会越界覆盖下一段。
    //     正确语义：每个本卡产生 bs*topK 个副本，最坏情况单 rank 收 bs*topK 个，故每段 = bs*topK。
    CcuRep::Variable myBaseOffVar = CreateVariable();
    myBaseOffVar = static_cast<uint64_t>(epRankId_) * bs_ * topK_ * tokenMsgLen_;
    CcuRep::Variable mySlotBaseVar = CreateVariable();
    mySlotBaseVar = yAddrVar_ + myBaseOffVar;
    std::vector<CcuRep::Variable> remoteSlotBaseVar(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        CcuRep::Variable v = CreateVariable();
        v = remoteYAddrs_[r] + myBaseOffVar;
        remoteSlotBaseVar[r] = v;
    }

    // ============ 5. PreCount ============
    CcuRep::Variable rankVarA = CreateVariable();
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            uint32_t idx = i * topK_ + j;
            LoadVariable(routingTableAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t), rankVarA);
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarA == r) {
                    sendBytesPerPeer[r]  = sendBytesPerPeer[r] + tokenMsgLenVar_;
                    sendTokensPerPeer[r] = sendTokensPerPeer[r] + oneVar;
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
    // v8 改造：拆为两个独立阶段。
    //   阶段 A：按 expertId 维度写 expand_idx——per-expert 累计计数，对齐库上语义
    //           "给同一专家发送的 token 个数"（aclnnMoeDistributeDispatchSetup.md expandIdxOut 描述）。
    //   阶段 B：按 rank 维度做数据发送——per-rank peerOccBytes_ 字节累加器维护对端 window 槽位偏移。
    //   两阶段独立维护，避免 v7 把 per-rank 计数写到 expand_idx 的语义 bug。
    CcuRep::Variable expertIdVar = CreateVariable();
    CcuRep::Variable rankVarB    = CreateVariable();
    CcuRep::Variable curOccVar   = CreateVariable();
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            uint32_t idx = i * topK_ + j;
            // 读 expertId（用于阶段 A 的 per-expert 计数器索引）
            LoadVariable(expertIdsAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t), expertIdVar);
            // 读路由 rank（用于阶段 B 的发送地址定位）
            LoadVariable(routingTableAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t), rankVarB);
            localSrc_.addr = xAddrVar_ + static_cast<uint64_t>(i) * tokenMsgLen_;

            // ---- 阶段 A：按 expertId 维度写 expand_idx ----
            // host 端展开 moeExpertNum_ 个 CCU_IF，逐 expertId 匹配；
            // 命中的 expertId 对应的 expertOccCnts_[e] 是"已经发给 expert e 的副本数"。
            for (uint32_t e = 0; e < moeExpertNum_; ++e) {
                CCU_IF(expertIdVar == e) {
                    curOccVar = expertOccCnts_[e];
                    StoreVariable(curOccVar,
                                  expandIdxAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t));
                    expertOccCnts_[e] = curOccVar + oneVar;
                }
            }

            // ---- 阶段 B：按 rank 维度发送数据 ----
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarB == r) {
                    if (r == epRankId_) {
                        localDst_.addr = mySlotBaseVar + peerOccBytes_[r];
                        CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenMsgLenVar_, event_));
                    } else {
                        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
                        remoteDst_.addr = remoteSlotBaseVar[r] + peerOccBytes_[r];
                        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                         localSrc_, tokenMsgLenVar_,
                                                         MOE_DISPATCH_CCU_TAG));
                    }
                    peerOccBytes_[r] = peerOccBytes_[r] + tokenMsgLenVar_;
                }
            }
        }
    }

    // ============ 8. PostNotify+Sync ============
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

    StoreVariable(sendTokensPerPeer[epRankId_],
                  commCmdInfoAddrVar_ + static_cast<uint64_t>(epRankId_) * sizeof(uint32_t));

    return HCCL_SUCCESS;
}

} // namespace hcomm
