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
 * \file moe_distribute_combine_ccu_setup.cpp
 * \brief combine_setup CCU 算法（v20.2，与 dispatch_setup 完全对称，反向）。
 *
 * 三项机制 + 地址范式同 dispatch_setup.cpp（见其头注释）：notify 槽位 mask / token 双端设 /
 *   段 7 自发自逐条 SetMask(1)+WaitEvent 串行落地 + 段 9 WaitAllPeersArriveNotify(R-1) + 段 10 ready=1。
 *
 * 数据流（反向于 dispatch）：
 *   dispatch:  本卡 x[i]              → 对端 commWindow[本卡段, 槽 k]
 *   combine :  本卡 expand_x[紧密 idx] → 对端 srcRank 的 commWindow[dstByteOff]
 *
 * commWindow（combine 阶段）：单段 (bs*K, tokenWithMetaLen)，slot=srcSlot*K+dstK，
 *   host 预算 metadata.dstByteOff 直接作字节偏移（device 不乘法）。driver launch 前 memset 0。
 *
 * commCmdInfo 布局（v20 收口）：只剩 ready 区 [0,R)。combine teardown K-stride 不读 tokenCount。
 *
 * 0.5RTT tag：MOE_COMBINE_CCU_TAG ≠ dispatch，独立资源组互不干扰。
 */

#include "moe_distribute_combine_ccu_setup.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// ---- v19 ★ PreSync 信息槽（XN_ID），与 dispatch_setup 同款 ----
namespace {
constexpr uint32_t COMMWIN_ADDR_XN   = 0;
constexpr uint32_t COMMWIN_TOKEN_XN  = 1;
//   v20 收口后 commCmdInfo 纯本地（同 dispatch），无需 PreSync 交换对端 cmdInfo 地址/token。
constexpr uint32_t PRE_SLOT_BITS     = (1U << COMMWIN_ADDR_XN) | (1U << COMMWIN_TOKEN_XN); // = 0x3
constexpr uint32_t CKE_PRE  = 0;
constexpr uint32_t METADATA_BYTES = 32;
constexpr uint64_t META_SRCRANK_OFF = 0;     // metadata 内 srcRank 偏移（相对 token data 末尾）
constexpr uint64_t META_DSTBYTEOFF_OFF = 12; // metadata 内 dstByteOff 偏移
}

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeCombineCcuSetup::CcuKernelArgMoeDistributeCombineCcuSetup(
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    uint32_t commQuantMode_)
    : epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      commQuantMode(commQuantMode_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      tokenMsgLen(commQuantMode_ == 0
                  ? static_cast<uint64_t>(((hiddenSize_ + 255U) / 256U) * 256U) * sizeof(uint16_t)
                  : static_cast<uint64_t>(((hiddenSize_ + hiddenSize_ / 32U + 31U) / 32U) * 32U)),
      moeExpertPerRankNum(moeExpertNum_ / epWorldSize_)
{}

CcuKernelSignature CcuKernelArgMoeDistributeCombineCcuSetup::GetKernelSignature() const
{
    CcuKernelSignature sig;
    sig.Append(std::string("MoeDistributeCombineCcuSetup"));
    sig.Append(epWorldSize);
    sig.Append(epRankId);
    sig.Append(moeExpertNum);
    sig.Append(commQuantMode);
    sig.Append(bs);
    sig.Append(topK);
    sig.Append(hiddenSize);
    return sig;
}

// --------------------------- CcuTaskArg ----------------------------

CcuTaskArgMoeDistributeCombineCcuSetup::CcuTaskArgMoeDistributeCombineCcuSetup(
    uint64_t expandXAddr_, uint64_t commCmdInfoAddr_, uint64_t commWindowAddr_,
    uint64_t expandXToken_, uint64_t commWindowToken_, uint64_t commCmdInfoToken_,
    uint64_t oneBufAddr_, uint64_t oneBufToken_)
    : expandXAddr(expandXAddr_), commCmdInfoAddr(commCmdInfoAddr_),
      commWindowAddr(commWindowAddr_),
      expandXToken(expandXToken_), commWindowToken(commWindowToken_), commCmdInfoToken(commCmdInfoToken_),
      oneBufAddr(oneBufAddr_), oneBufToken(oneBufToken_)
{}

// --------------------------- CcuKernel ----------------------------

CcuKernelMoeDistributeCombineCcuSetup::CcuKernelMoeDistributeCombineCcuSetup(const CcuKernelArg &arg)
    : Mc2CcuKernelBase(arg)
{
    const auto &myArg = static_cast<const CcuKernelArgMoeDistributeCombineCcuSetup &>(arg);
    epWorldSize_         = myArg.epWorldSize;
    epRankId_            = myArg.epRankId;
    moeExpertPerRankNum_ = myArg.moeExpertPerRankNum;
    bs_                  = myArg.bs;
    topK_                = myArg.topK;
    tokenMsgLen_         = myArg.tokenMsgLen;
}

std::vector<uint64_t> CcuKernelMoeDistributeCombineCcuSetup::GeneArgs(const CcuTaskArg &arg)
{
    const auto &myArg = static_cast<const CcuTaskArgMoeDistributeCombineCcuSetup &>(arg);
    // GeneArgs 顺序 = device 端 LoadArgs 顺序。v19 ★ 末尾追加 3 个本地 token。
    return {
        myArg.expandXAddr,
        myArg.commCmdInfoAddr,
        myArg.commWindowAddr,
        tokenMsgLen_,
        myArg.expandXToken,          // v19 ★
        myArg.commWindowToken,       // v19 ★
        myArg.commCmdInfoToken,      // v19 ★
        myArg.oneBufAddr,            // 收口 ★ ready 写源（末尾追加，LoadArgs 同序）
        myArg.oneBufToken,           // 收口 ★
    };
}

HcclResult CcuKernelMoeDistributeCombineCcuSetup::Algorithm()
{
    // ============ 1. InitResource ============
    expandXAddrVar_      = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    commWindowAddrVar_   = CreateVariable();
    tokenMsgLenVar_      = CreateVariable();
    expandXTokenVar_     = CreateVariable();              // v19 ★
    commWindowTokenVar_  = CreateVariable();              // v19 ★
    commCmdInfoTokenVar_ = CreateVariable();              // v19 ★
    oneBufAddrVar_       = CreateVariable();              // 收口 ★ ready 写源 GM 地址
    oneBufTokenVar_      = CreateVariable();              // 收口 ★ oneBuf token
    localSrc_            = CreateLocalAddr();
    localDst_            = CreateLocalAddr();
    remoteDst_           = CreateRemoteAddr();
    event_               = CreateCompletedEvent();

    remoteCommWindowAddrs_.reserve(epWorldSize_);
    remoteCommWindowTokens_.reserve(epWorldSize_);
    std::vector<CcuRep::Variable> sendBytesPerPeer(epWorldSize_);

    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) {
            remoteCommWindowAddrs_.push_back(CreateVariable());
            remoteCommWindowTokens_.push_back(CreateVariable());
        } else {
            uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
            CcuRep::Variable vWinAddr, vWinTok;
            CHK_RET(CreateVariable(channels_[channelIdx], COMMWIN_ADDR_XN,  &vWinAddr));
            CHK_RET(CreateVariable(channels_[channelIdx], COMMWIN_TOKEN_XN, &vWinTok));
            remoteCommWindowAddrs_.push_back(vWinAddr);
            remoteCommWindowTokens_.push_back(vWinTok);
        }
        CcuRep::Variable sb = CreateVariable(); sb = 0; sendBytesPerPeer[r]  = sb;
    }

    // ============ 2. LoadArgs ============（顺序严格 = GeneArgs）
    Load(expandXAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(commWindowAddrVar_);
    Load(tokenMsgLenVar_);
    Load(expandXTokenVar_);            // v19 ★
    Load(commWindowTokenVar_);         // v19 ★
    Load(commCmdInfoTokenVar_);        // v19 ★
    Load(oneBufAddrVar_);              // 收口 ★（顺序严格 = GeneArgs 末尾）
    Load(oneBufTokenVar_);             // 收口 ★
    remoteCommWindowAddrs_[epRankId_]   = commWindowAddrVar_;
    remoteCommWindowTokens_[epRankId_]  = commWindowTokenVar_;

    // 常用立即数 Variable（收口后 combine 无 elem4Var/oneVar：段 5.5/段 8 删 + PreCount 不再数 token）
    CcuRep::Variable metaBytesVar = CreateVariable(); metaBytesVar = static_cast<uint64_t>(METADATA_BYTES);
    CcuRep::Variable tokenWithMetaLenVar = CreateVariable();
    tokenWithMetaLenVar = tokenMsgLenVar_ + metaBytesVar;
    // metadata 两个字段相对 token 行首的偏移 Variable（= tokenMsgLen + 常量）
    CcuRep::Variable metaSrcRankOffVar = CreateVariable();
    {
        CcuRep::Variable c = CreateVariable(); c = META_SRCRANK_OFF;
        metaSrcRankOffVar = tokenMsgLenVar_ + c;          // = tokenMsgLen + 0
    }
    CcuRep::Variable metaDstByteOffOffVar = CreateVariable();
    {
        CcuRep::Variable c = CreateVariable(); c = META_DSTBYTEOFF_OFF;
        metaDstByteOffOffVar = tokenMsgLenVar_ + c;       // = tokenMsgLen + 12
    }

    // ============ 2.5. Clear 上次 combine 的 0.5RTT 状态 ============
    CHK_RET(ClearAllPeersArriveNotify(MOE_COMBINE_CCU_TAG));

    // ============ 3. PreSync（需求1 ★ 槽位 mask）============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_PRE, COMMWIN_ADDR_XN,
                             commWindowAddrVar_,   1U << COMMWIN_ADDR_XN));
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_PRE, COMMWIN_TOKEN_XN,
                             commWindowTokenVar_,  1U << COMMWIN_TOKEN_XN));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_PRE, PRE_SLOT_BITS));
    }

    // ============ 4. BuildSlotBase（单段：base = commWindowAddr，无 epRankId 偏移）============
    CcuRep::Variable mySlotBaseVar = CreateVariable();
    mySlotBaseVar = commWindowAddrVar_;                   // Variable = Variable
    std::vector<CcuRep::Variable> remoteSlotBaseVar(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        CcuRep::Variable v = CreateVariable();
        v = remoteCommWindowAddrs_[r];
        remoteSlotBaseVar[r] = v;
    }

    // ============ 5. PreCount（地址递进范式 ★）============
    //   host unroll 上界 aBound = bs * rankSize * min(localExpertNum, K)（op_def 标准 A，覆盖紧密区 worst-case）。
    //   维护 srcRankReadAddr 递进读每 token 的 metadata.srcRank（行内偏移 = tokenMsgLen + 0）。
    const uint32_t aBound = bs_ * epWorldSize_
                          * ((moeExpertPerRankNum_ < topK_) ? moeExpertPerRankNum_ : topK_);
    CcuRep::Variable srcRankVarA = CreateVariable();
    // 行首递进地址（每行步进 tokenWithMetaLen）
    CcuRep::Variable rowAddrA = CreateVariable(); rowAddrA = expandXAddrVar_;
    for (uint32_t idx = 0; idx < aBound; ++idx) {
        // srcRank 读地址 = 行首 + (tokenMsgLen + 0)
        CcuRep::Variable srcRankAddr = CreateVariable();
        srcRankAddr = rowAddrA;
        srcRankAddr += metaSrcRankOffVar;
        LoadVariable(srcRankAddr, srcRankVarA);
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            CCU_IF(srcRankVarA == r) {
                sendBytesPerPeer[r] = sendBytesPerPeer[r] + tokenWithMetaLenVar;
            }
        }
        rowAddrA += tokenWithMetaLenVar;                  // 递进到下一行
    }

    // ============ 5.5.（已删）commCmdInfo 4 字节 overhead（同 dispatch 收口）============
    //   tokenCount 不再跨卡写（改 teardown 侧处理），sendBytes 回归纯数据字节，不再 +4。

    // ============ 6. RecordExpectCount ============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(RecordExpectCount(channels_[channelIdx], sendBytesPerPeer[r], MOE_COMBINE_CCU_TAG));
    }

    // ============ 7. DoMainLoop（token + 地址递进 + dstByteOff 直接索引 ★）============
    //   localSrc_.token = 本地 expandX token（数据源始终本地）；行首地址 rowAddrB 递进。
    //   每行：读 srcRank、读 dstByteOff（都用临时地址 Variable = 行首 + 偏移）；
    //   按 srcRank：自反自 LocalCopyNb(token=本地 commWindow)，跨卡 WriteNbWithArriveNotify(token=对端 commWindow)。
    CcuRep::Variable srcRankVarB   = CreateVariable();
    CcuRep::Variable dstByteOffVar = CreateVariable();
    CcuRep::Variable rowAddrB = CreateVariable(); rowAddrB = expandXAddrVar_;
    localSrc_.token = expandXTokenVar_;
    for (uint32_t idx = 0; idx < aBound; ++idx) {
        // 读 srcRank（行首 + tokenMsgLen + 0）
        CcuRep::Variable srcRankAddr = CreateVariable();
        srcRankAddr = rowAddrB;
        srcRankAddr += metaSrcRankOffVar;
        LoadVariable(srcRankAddr, srcRankVarB);
        // 读 dstByteOff（行首 + tokenMsgLen + 12）
        CcuRep::Variable dstByteOffAddr = CreateVariable();
        dstByteOffAddr = rowAddrB;
        dstByteOffAddr += metaDstByteOffOffVar;
        LoadVariable(dstByteOffAddr, dstByteOffVar);
        // localSrc_.addr = 行首（整行 data+meta）
        localSrc_.addr = rowAddrB;

        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            CCU_IF(srcRankVarB == r) {
                if (r == epRankId_) {
                    localDst_.addr  = mySlotBaseVar;
                    localDst_.addr += dstByteOffVar;       // 直接用 host 预算的字节偏移
                    localDst_.token = commWindowTokenVar_;
                    // ★ event 范式同 dispatch（对齐 all_gather_nhr1d:269-272 CCU_IF 内"发一条等一条"）：
                    //   自发自 per-token，SetMask(bit0)+LocalCopyNb+立即 WaitEvent 串行落地（bit0 复用）。
                    event_.SetMask(1U);
                    CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenWithMetaLenVar, event_));
                    CHK_RET(WaitEvent(event_));
                } else {
                    uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
                    remoteDst_.addr  = remoteSlotBaseVar[r];
                    remoteDst_.addr += dstByteOffVar;
                    remoteDst_.token = remoteCommWindowTokens_[r];
                    CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                    localSrc_, tokenWithMetaLenVar,
                                                    MOE_COMBINE_CCU_TAG));
                }
            }
        }
        rowAddrB += tokenWithMetaLenVar;                  // 递进到下一行
    }

    // ============ 8.（已删）写 tokenCount 跨卡 + 本卡（同 dispatch 收口）============
    //   原 R 处自造 StoreVariableAt（scratch 暂存 + 跨卡 + 自卡）全删；combine 的 tokenCount 也不再由
    //   发送方 CCU 产出。combine_teardown AIV 用 K-stride 算法（从 commWindow[i*K] 取副本），本就不读
    //   tokenCount，故删除后 teardown 仅需调整 ready 区偏移，无自数逻辑（详见 combine_teardown.cpp 注释）。

    // ============ 9. WaitAllPeersArriveNotify（需求3 ★）============
    //   段 7 自发自已 CCU_IF 内逐条 WaitEvent 落地，此处无需再等 event。
    //   ★ SE 依赖：WaitAllPeersArriveNotify target 语义同 dispatch（见 dispatch 段9 注释 / OPEN_QUESTIONS）。
    CcuRep::Variable arriveTargetVar = CreateVariable();
    arriveTargetVar = static_cast<uint64_t>(epWorldSize_ - 1U);
    CHK_RET(WaitAllPeersArriveNotify(arriveTargetVar, MOE_COMBINE_CCU_TAG));

    // ============ 10. 设 ready[0..R)=1（1 次 hccl 现役 LocalCopyNb，同 dispatch 收口）============
    //   把本卡 oneBuf（host 填 R 个 int32=1）整块拷到 commCmdInfo ready 区（off=0，最简布局）；
    //   LocalAddr 两端 .token 都设（本地 buffer 本地 token）；len = R*4。随后 WaitEvent 确保落地。
    CcuRep::Variable readyBaseOff = CreateVariable();
    readyBaseOff = static_cast<uint64_t>(0U);             // ready 区从偏移 0 起（删 tokenCount/scratch 后最简）
    CcuRep::Variable readyAddr = CreateVariable();
    readyAddr = commCmdInfoAddrVar_;                      // Address = Variable
    readyAddr += readyBaseOff;                            // Address += Variable（=0，保持地址范式一致）
    CcuRep::Variable rankSizeBytesVar = CreateVariable();
    rankSizeBytesVar = static_cast<uint64_t>(epWorldSize_) * sizeof(uint32_t);   // R*4 字节
    CcuRep::LocalAddr readyDst = CreateLocalAddr();
    readyDst.addr  = readyAddr;
    readyDst.token = commCmdInfoTokenVar_;
    CcuRep::LocalAddr readySrc = CreateLocalAddr();
    readySrc.addr  = oneBufAddrVar_;
    readySrc.token = oneBufTokenVar_;
    // ready 单条 LocalCopyNb 占 bit0（DMA 前 SetMask 指定位，对齐 event 位图范式）
    event_.SetMask(1U);
    CHK_RET(LocalCopyNb(readyDst, readySrc, rankSizeBytesVar, event_));
    event_.SetMask(1U);
    CHK_RET(WaitEvent(event_));

    return HCCL_SUCCESS;
}

} // namespace hcomm
