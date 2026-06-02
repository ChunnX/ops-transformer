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
 *   段 7 自发自逐条 SetMask(1)+WaitEvent 串行落地 + 段 9 WaitallpeersarriveNotify(R-1) + 段 10 ready=1。
 *
 * 数据流（v20.7 ★ 反向于 dispatch，全程干净 token，按 owner 分段回送）：
 *   dispatch:  本卡 x[i]              → 对端 commWindow[本卡段, 槽 k]（带 metadata）
 *   combine :  本卡 expandX[紧密 idx]（干净）+ assistInfo[idx]=dstByteOff（单字段）
 *              → owner 卡 commWindow[dstByteOff]（发干净 tokenMsgLen，无 metadata，Req1）
 *   ★ expandX/assistInfo 按 ownerRank 严格分段（卡0 段、卡1 段…），故 combine 外层 host 展开 R 段、
 *     owner=r 编译期已知，内层 CCU_WHILE 跑 cnt[r] 次——无 per-token rank 判断，self/cross 编译期分支。
 *
 * commWindow（combine 阶段）：单段 (bs*K, tokenMsgLen) 干净槽，slot=srcSlot*K+dstK，
 *   host 预算 dstByteOff=(i*K+j)*tokenMsgLen 直接作字节偏移（device 不乘法）。driver launch 前 memset 0。
 *
 * commCmdInfo 布局（v20.7 ★，与 dispatch_teardown/driver 三处一致）：
 *   ready[0..R) int32 │ 8 对齐 │ sendBytes[0..R) int64（=cnt[r]*tokenMsgLen）│ negCnt[0..R) int64（=-cnt[r]）
 *   段 5 读 sendBytes（段6 wishCnt）+ negCnt（段7 每段倒计数），不再扫描 expandX 自累加。
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
//   v20.7：删除 metadata 行内偏移常量——combine 不再从 expandX 行内读 metadata（expandX 已干净），
//   路由偏移 dstByteOff 改从独立 assistInfo（int64 单字段）LoadVariable 取；owner 由 combine 按段隐含。
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
    uint64_t expandXAddr_, uint64_t assistInfoAddr_, uint64_t commCmdInfoAddr_, uint64_t commWindowAddr_,
    uint64_t expandXToken_, uint64_t assistInfoToken_, uint64_t commWindowToken_, uint64_t commCmdInfoToken_,
    uint64_t oneBufAddr_, uint64_t oneBufToken_)
    : expandXAddr(expandXAddr_), assistInfoAddr(assistInfoAddr_), commCmdInfoAddr(commCmdInfoAddr_),
      commWindowAddr(commWindowAddr_),
      expandXToken(expandXToken_), assistInfoToken(assistInfoToken_),
      commWindowToken(commWindowToken_), commCmdInfoToken(commCmdInfoToken_),
      oneBufAddr(oneBufAddr_), oneBufToken(oneBufToken_)
{}

// --------------------------- CcuKernel ----------------------------

CcuKernelMoeDistributeCombineCcuSetup::CcuKernelMoeDistributeCombineCcuSetup(const CcuKernelArg &arg)
    : Mc2CcuKernelBase(arg)
{
    const auto &myArg = static_cast<const CcuKernelArgMoeDistributeCombineCcuSetup &>(arg);
    epWorldSize_         = myArg.epWorldSize;
    epRankId_            = myArg.epRankId;
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
        myArg.assistInfoAddr,        // v20.6 ★ 路由元信息
        myArg.commCmdInfoAddr,
        myArg.commWindowAddr,
        tokenMsgLen_,
        myArg.expandXToken,          // v19 ★
        myArg.assistInfoToken,       // v20.6 ★
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
    assistInfoAddrVar_   = CreateVariable();              // v20.6 ★
    commCmdInfoAddrVar_  = CreateVariable();
    commWindowAddrVar_   = CreateVariable();
    tokenMsgLenVar_      = CreateVariable();
    expandXTokenVar_     = CreateVariable();              // v19 ★
    assistInfoTokenVar_  = CreateVariable();              // v20.6 ★
    commWindowTokenVar_  = CreateVariable();              // v19 ★
    commCmdInfoTokenVar_ = CreateVariable();              // v19 ★
    oneBufAddrVar_       = CreateVariable();              // 收口 ★ ready 写源 GM 地址
    oneBufTokenVar_      = CreateVariable();              // 收口 ★ oneBuf token
    localSrc_            = CreateLocalAddr();
    localDst_            = CreateLocalAddr();
    remoteDst_           = CreateRemoteAddr();
    event_               = CreateCompletedEvent();
    // ★ A6 LocalCopyNb(dst,src,len,bufs,bufCount,event) 的 UB 暂存 buf 必须来自 moRes 式块分配
    //   （SE 对齐 Req4：不能用单个 CreateCcuBuf()，须 CreateBlockCcuBuf——moRes.ccuBuf 即此块分配的元素）。
    //   单槽串行复用：每条 LocalCopyNb 后紧跟 WaitEvent 落地，buf 用完即释放、无并发，故 1 槽即正确。
    CHK_RET(CreateBlockCcuBuf(1U, &localCopyBuf_));

    remoteCommWindowAddrs_.reserve(epWorldSize_);
    remoteCommWindowTokens_.reserve(epWorldSize_);
    // v20.7 ★ 两组 per-peer Variable：sendBytesPerPeer（段6 wishCnt）+ negCntPerPeer（段7 每段倒计数初值）
    std::vector<CcuRep::Variable> sendBytesPerPeer(epWorldSize_);
    std::vector<CcuRep::Variable> negCntPerPeer(epWorldSize_);

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
        // ★ vector<Variable>(n) 元素默认构造（空 context），赋值前先 CreateVariable() 绑真实 XN 槽（SE 对齐）
        sendBytesPerPeer[r] = CreateVariable(); sendBytesPerPeer[r] = 0;
        negCntPerPeer[r]    = CreateVariable(); negCntPerPeer[r]    = 0;
    }

    // ============ 2. LoadArgs ============（顺序严格 = GeneArgs）
    Load(expandXAddrVar_);
    Load(assistInfoAddrVar_);          // v20.6 ★
    Load(commCmdInfoAddrVar_);
    Load(commWindowAddrVar_);
    Load(tokenMsgLenVar_);
    Load(expandXTokenVar_);            // v19 ★
    Load(assistInfoTokenVar_);         // v20.6 ★
    Load(commWindowTokenVar_);         // v19 ★
    Load(commCmdInfoTokenVar_);        // v19 ★
    Load(oneBufAddrVar_);              // 收口 ★（顺序严格 = GeneArgs 末尾）
    Load(oneBufTokenVar_);             // 收口 ★
    remoteCommWindowAddrs_[epRankId_]   = commWindowAddrVar_;
    remoteCommWindowTokens_[epRankId_]  = commWindowTokenVar_;

    // v20.7 ★ 立即数 Variable：
    //   elem8Var=8：commCmdInfo int64 读递进 + assistInfo 单字段 dstByteOff 行步进（每 token 1×int64=8）；
    //   oneVar=1：每段 CCU_WHILE 倒计数 idxVar 自增。combine 发【干净 token】，长度 = tokenMsgLenVar_（Req1）。
    CcuRep::Variable elem8Var = CreateVariable(); elem8Var = static_cast<uint64_t>(sizeof(int64_t));   // =8
    CcuRep::Variable oneVar   = CreateVariable(); oneVar   = static_cast<uint64_t>(1);

    // ============ 2.5. Clear 上次 combine 的 0.5RTT 状态 ============
    CHK_RET(clearallpeersarriveNotify(MOE_COMBINE_CCU_TAG));

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
        remoteSlotBaseVar[r] = CreateVariable();   // ★ 默认构造元素先绑真实 XN 槽再赋值（SE 对齐）
        remoteSlotBaseVar[r] = v;
    }

    // ============ 5. 读 sendBytes[0..R) + negCnt[0..R)（v20.7 ★ Req2：来自 dispatch_teardown 写的 commCmdInfo）============
    //   不再扫描 expandX 自累加（dispatch_teardown 已按 ownerRank 分段算好 cnt[r]*tokenMsgLen 与 negCnt[r]=-cnt[r]）。
    //   commCmdInfo 布局（与 teardown/driver 三处严格一致）：
    //     ready[0..R) int32（byte [0,4R)）│ 8 字节对齐 │ sendBytes[0..R) int64 │ negCnt[0..R) int64
    //   cinfoByteOff = 8 对齐后的 sendBytes 起始字节偏移；sbAddr 递进 8 字节连续 LoadVariable（int64 全宽）。
    const uint64_t cinfoByteOff = ((static_cast<uint64_t>(epWorldSize_) * sizeof(int32_t) + 7U) / 8U) * 8U;
    CcuRep::Variable sbAddr = CreateVariable();
    sbAddr = commCmdInfoAddrVar_;                         // base
    { CcuRep::Variable off = CreateVariable(); off = cinfoByteOff; sbAddr += off; }   // → &sendBytes[0]
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        LoadVariable(sbAddr, sendBytesPerPeer[r]);        // sendBytes[r] = cnt[r]*tokenMsgLen（段6 wishCnt，干净字节）
        sbAddr += elem8Var;                               // +8 → 下一 int64
    }
    // sbAddr 现指向 negCnt[0]（紧跟 sendBytes[0..R)）；连续读 negCnt[0..R)（每段 CCU_WHILE 倒计数初值）
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        LoadVariable(sbAddr, negCntPerPeer[r]);           // negCnt[r] = (uint64_t)(0 - cnt[r])
        sbAddr += elem8Var;                               // +8 → 下一 int64
    }

    // ============ 6. RecordExpectCount（SE 对齐：新增 sem+mask 完成事件）============
    //   写对端 wishCnt 是异步 issue，经 event_ 的 bit0 标记本地完成；逐条 SetMask(1)+WaitEvent 串行落地
    //   （event_ 串行复用，bit0 用完即清，与段 7 自反自 LocalCopyNb 同范式）。
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        event_.SetMask(1U);
        CHK_RET(RecordExpectCount(channels_[channelIdx], sendBytesPerPeer[r], MOE_COMBINE_CCU_TAG, event_, 1U));
        CHK_RET(WaitEvent(event_));
    }

    // ============ 7. DoMainLoop（v20.7 ★ 按 owner 分段：消除 per-token ownerRank 读 + R 路 CCU_IF）============
    //   关键不变式（dispatch_teardown 保证）：expandX/assistInfo 按 ownerRank 严格分段连续堆放
    //   （卡0 的 cnt[0] 个、卡1 的 cnt[1] 个…依次紧密）。故 combine 按 owner 分段消费：
    //     · 外层 host 展开 R 段——owner = r 是【编译期常量】，self/cross 走【编译期分支】(r vs epRankId_)；
    //     · 内层 CCU_WHILE 跑 cnt[r] 次（negCnt[r] 自增到 0，CCU Variable 只有 +，倒计数已由 teardown AIV 算好）。
    //   相比扁平 CCU_WHILE(total)：省 total 次 ownerRank LoadVariable + total×R 次运行期 CCU_IF；assistInfo 因此
    //   只需单字段 dstByteOff。rowAddrB/assistAddrB 在所有段外初始化、跨段连续推进（与全局紧密排布对应）。
    //   ★ Variable 全部在 CCU_WHILE 外创建一次；每段循环体仅 trace 一遍、发射循环微码，体内只做赋值/Load/通信。
    CcuRep::Variable dstByteOffVar = CreateVariable();
    CcuRep::Variable idxVar        = CreateVariable();
    CcuRep::Variable rowAddrB      = CreateVariable(); rowAddrB    = expandXAddrVar_;     // expandX 干净行指针（跨段连续）
    CcuRep::Variable assistAddrB   = CreateVariable(); assistAddrB = assistInfoAddrVar_;  // assistInfo 行指针（跨段连续）
    localSrc_.token = expandXTokenVar_;
    for (uint32_t r = 0; r < epWorldSize_; ++r) {            // host 展开 R 段，owner=r 编译期已知
        const bool isSelf = (r == epRankId_);
        uint32_t channelIdx = isSelf ? 0U : ((r < epRankId_) ? r : r - 1);
        idxVar = negCntPerPeer[r];                          // 该段倒计数初值 = -cnt[r]（cnt[r]==0 → 0 → 该段 0 次）
        if (isSelf) {
            localDst_.token = commWindowTokenVar_;          // 自反自目的 token（整段不变，循环外设一次）
        } else {
            remoteDst_.token = remoteCommWindowTokens_[r];  // 对端 r 目的 token（整段不变）
        }
        CCU_WHILE(idxVar != 0) {
            LoadVariable(assistAddrB, dstByteOffVar);       // assistInfo[t] = dstByteOff（单字段，回送目的偏移）
            localSrc_.addr = rowAddrB;                      // 数据源 = expandX 干净行（仅 data，tokenMsgLen）
            if (isSelf) {                                   // ★ 编译期分支：self 走 LocalCopyNb
                localDst_.addr  = mySlotBaseVar;
                localDst_.addr += dstByteOffVar;            // = (i*K+j)*tokenMsgLen
                event_.SetMask(1U);                         // 自反自 per-token：SetMask+LocalCopyNb+立即 WaitEvent 串行
                CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenMsgLenVar_, &localCopyBuf_, 1U, event_));
                CHK_RET(WaitEvent(event_));
            } else {                                        // ★ 编译期分支：cross 走 WriteNbWithArriveNotify
                remoteDst_.addr  = remoteSlotBaseVar[r];
                remoteDst_.addr += dstByteOffVar;
                CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                localSrc_, tokenMsgLenVar_, MOE_COMBINE_CCU_TAG));
            }
            rowAddrB    += tokenMsgLenVar_;                 // expandX 干净行步进（跨段连续）
            assistAddrB += elem8Var;                        // assistInfo 单字段步进 = 8（跨段连续）
            idxVar      += oneVar;                          // 自增 → 回绕到 0 退出该段（共 cnt[r] 次）
        }
    }

    // ============ 8.（已删）写 tokenCount 跨卡 + 本卡（同 dispatch 收口）============
    //   原 R 处自造 StoreVariableAt（scratch 暂存 + 跨卡 + 自卡）全删；combine 的 tokenCount 也不再由
    //   发送方 CCU 产出。combine_teardown AIV 用 K-stride 算法（从 commWindow[i*K] 取副本），本就不读
    //   tokenCount，故删除后 teardown 仅需调整 ready 区偏移，无自数逻辑（详见 combine_teardown.cpp 注释）。

    // ============ 9. WaitallpeersarriveNotify（需求3 ★）============
    //   段 7 自发自已 CCU_IF 内逐条 WaitEvent 落地，此处无需再等 event。
    //   ★ SE 依赖：WaitallpeersarriveNotify target 语义同 dispatch（见 dispatch 段9 注释 / OPEN_QUESTIONS）。
    CcuRep::Variable arriveTargetVar = CreateVariable();
    arriveTargetVar = static_cast<uint64_t>(epWorldSize_ - 1U);
    CHK_RET(WaitallpeersarriveNotify(arriveTargetVar, MOE_COMBINE_CCU_TAG));

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
    //   ★ A6 新签名：同样传 UB 暂存 buf（&localCopyBuf_, 1），复用同一槽（前序 copy 已 WaitEvent 落地）。
    event_.SetMask(1U);
    CHK_RET(LocalCopyNb(readyDst, readySrc, rankSizeBytesVar, &localCopyBuf_, 1U, event_));
    event_.SetMask(1U);
    CHK_RET(WaitEvent(event_));

    return HCCL_SUCCESS;
}

} // namespace hcomm
