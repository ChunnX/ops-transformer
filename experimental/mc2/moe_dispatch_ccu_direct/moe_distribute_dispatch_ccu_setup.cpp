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
 * \brief A6 CCU MoeDistributeDispatchCcuSetup device 算法（v20.2，对齐 hccl 真实 CCU 范式 + 死代码清理：PreSync 2 槽、commCmdInfo 纯本地）。
 *
 * 三项核心机制（亲核实 all_to_all_v / all_reduce / reduce_scatter_v / all_gather mesh1d）：
 *   需求1 notify：PreSync 槽位 mask——NotifyRecord(ch, CKE_PRE, XN_ID, var, 1<<XN_ID) +
 *                 NotifyWait(ch, CKE_PRE, allSlotBits)。CKE=同步轮次，XN_ID=信息槽，mask=1<<XN（非 rank）。
 *   需求2 token：所有 LocalAddr/RemoteAddr 的 .token 必设——本地 buffer 用本地 token（Load 自 SQE），
 *                 对端 buffer 用对端 token（CreateVariable(ch,XN,&v) + PreSync 交换）。
 *   需求3 异步时序：段 7 自发自 LocalCopyNb 逐条 SetMask(1)+WaitEvent 串行落地（event mask 是 per-DMA
 *                 位图，对齐 all_gather_nhr1d:269-272）；段 9 WaitAllPeersArriveNotify(R-1) 等本卡收齐；
 *                 段 10 一次 LocalCopyNb 置 ready[0..R)=1。teardown AIV 轮询 ready 区全 1。
 *
 * 地址范式（hcomm 不支持 Variable+立即数，实测 operator+ error；Address=Variable+Variable 亦非法）：
 *   读 GM 维护地址 Variable，LoadVariable(addrVar,dst) 后 addrVar += strideVar 递进；
 *   建址 addr=baseVar; addr+=offVar；立即数先 offVar=imm 再 += offVar。
 *
 * commCmdInfo 布局（v20 收口）：只剩 ready 区 [0,R)（uint32 0/1 状态位）。
 *   tokenCount 改由 teardown AIV 自数（遍历 commWindow 按 metadata.srcRank==r），setup 不再产出。
 *
 * Algorithm 段式：
 *   1 InitResource → 2 LoadArgs（顺序严格=GeneArgs，含 4 token+oneBuf）→ 2.5 Clear →
 *   3 PreSync（2 XN 槽：commWin 地址+token；commCmdInfo 纯本地不交换）→ 4 BuildSlotBase →
 *   5 PreCount（算 sendBytes 供 wishCnt）→ 6 RecordExpectCount →
 *   7 DoMainLoop（A=写 expand_idx，B=按 rank 发 token+meta：自发自 LocalCopyNb / 跨卡 WriteNbWithArriveNotify）→
 *   9 WaitAllPeersArriveNotify(R-1) → 10 LocalCopyNb 置 ready=1。
 *   ★ SE 依赖：段 7A 用 StoreVariable(Variable 地址版) 写 expand_idx —— legacy CcuContext 已有、mc2 在用，待 SE 补进 pkg_inc CcuKernel。
 */

#include "moe_distribute_dispatch_ccu_setup.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// ---- v19 ★ PreSync 信息槽（XN_ID）：每块独立信息一个槽，mask = 1<<XN_ID ----
//   严格对齐 hccl mesh1d：all_reduce 用 INPUT_XN=0/OUTPUT_XN=1/TOKEN_XN=2；本算子按需 4 槽。
namespace {
constexpr uint32_t COMMWIN_ADDR_XN   = 0;   // 对端 commWindow 起址
constexpr uint32_t COMMWIN_TOKEN_XN  = 1;   // 对端 commWindow token
//   v20 收口后 commCmdInfo 是纯本地 buffer（段 10 ready 本地 LocalCopyNb 写 + teardown 本地轮询），
//   不再跨卡，故无需 PreSync 交换对端 cmdInfo 地址/token——只留 commWindow 2 个槽（省一半 PreSync）。
constexpr uint32_t PRE_SLOT_BITS     = (1U << COMMWIN_ADDR_XN) | (1U << COMMWIN_TOKEN_XN); // = 0x3
constexpr uint32_t CKE_PRE  = 0;   // 同步轮次：前同步
constexpr uint32_t METADATA_BYTES = 32;     // 每 token metadata 三元组+pad 对齐
}

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeDispatchCcuSetup::CcuKernelArgMoeDistributeDispatchCcuSetup(
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    uint32_t quantMode_)
    : epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      quantMode(quantMode_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      // tokenMsgLen 按 quantMode 切公式：
      //   quantMode == 0 (BF16)：Align256(H) * sizeof(BF16)
      //   quantMode == 1 (MXFP8)：Align32(H + H/32)（FP8 data H + E8M0 scale H/32）
      tokenMsgLen(quantMode_ == 0
                  ? static_cast<uint64_t>(((hiddenSize_ + 255U) / 256U) * 256U) * sizeof(uint16_t)
                  : static_cast<uint64_t>(((hiddenSize_ + hiddenSize_ / 32U + 31U) / 32U) * 32U))
{}

CcuKernelSignature CcuKernelArgMoeDistributeDispatchCcuSetup::GetKernelSignature() const
{
    CcuKernelSignature sig;
    sig.Append(std::string("MoeDistributeDispatchCcuSetup"));
    sig.Append(epWorldSize);
    sig.Append(epRankId);
    sig.Append(moeExpertNum);
    sig.Append(quantMode);
    sig.Append(bs);
    sig.Append(topK);
    sig.Append(hiddenSize);
    return sig;
}

// --------------------------- CcuTaskArg ----------------------------

CcuTaskArgMoeDistributeDispatchCcuSetup::CcuTaskArgMoeDistributeDispatchCcuSetup(
    uint64_t expertIdsAddr_, uint64_t expandIdxAddr_, uint64_t commCmdInfoAddr_,
    uint64_t commWindowAddr_, uint64_t routingTableAddr_,
    uint64_t xWithMetaAddr_,
    uint64_t xWithMetaToken_,
    uint64_t commWindowToken_, uint64_t commCmdInfoToken_,
    uint64_t oneBufAddr_, uint64_t oneBufToken_)
    : expertIdsAddr(expertIdsAddr_), expandIdxAddr(expandIdxAddr_), commCmdInfoAddr(commCmdInfoAddr_),
      commWindowAddr(commWindowAddr_), routingTableAddr(routingTableAddr_),
      xWithMetaAddr(xWithMetaAddr_),
      xWithMetaToken(xWithMetaToken_),
      commWindowToken(commWindowToken_), commCmdInfoToken(commCmdInfoToken_),
      oneBufAddr(oneBufAddr_), oneBufToken(oneBufToken_)
{}

// --------------------------- CcuKernel ----------------------------

CcuKernelMoeDistributeDispatchCcuSetup::CcuKernelMoeDistributeDispatchCcuSetup(const CcuKernelArg &arg)
    : Mc2CcuKernelBase(arg)
{
    const auto &myArg = static_cast<const CcuKernelArgMoeDistributeDispatchCcuSetup &>(arg);
    epWorldSize_         = myArg.epWorldSize;
    epRankId_            = myArg.epRankId;
    moeExpertNum_        = myArg.moeExpertNum;
    bs_                  = myArg.bs;
    topK_                = myArg.topK;
    tokenMsgLen_         = myArg.tokenMsgLen;
}

std::vector<uint64_t> CcuKernelMoeDistributeDispatchCcuSetup::GeneArgs(const CcuTaskArg &arg)
{
    const auto &myArg = static_cast<const CcuTaskArgMoeDistributeDispatchCcuSetup &>(arg);
    // GeneArgs 顺序 = device 端 LoadArgs 顺序（务必一致，否则 device 读错地址/token）。
    //   v19 ★ 末尾追加 3 个本地 token（host GetTokenInfo(addr,size) 算）。
    return {
        myArg.expertIdsAddr,
        myArg.commWindowAddr,
        myArg.expandIdxAddr,
        myArg.commCmdInfoAddr,
        myArg.routingTableAddr,
        myArg.xWithMetaAddr,
        tokenMsgLen_,
        myArg.xWithMetaToken,        // v19 ★
        myArg.commWindowToken,       // v19 ★
        myArg.commCmdInfoToken,      // v19 ★
        myArg.oneBufAddr,            // 收口 ★ ready 写源（末尾追加，LoadArgs 同序）
        myArg.oneBufToken,           // 收口 ★
    };
}

HcclResult CcuKernelMoeDistributeDispatchCcuSetup::Algorithm()
{
    // ============ 1. InitResource ============
    expertIdsAddrVar_    = CreateVariable();
    commWindowAddrVar_   = CreateVariable();
    expandIdxAddrVar_    = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    routingTableAddrVar_ = CreateVariable();
    xWithMetaAddrVar_    = CreateVariable();
    tokenMsgLenVar_      = CreateVariable();
    tokenWithMetaLenVar_ = CreateVariable();
    xWithMetaTokenVar_   = CreateVariable();              // v19 ★ 本地 token
    commWindowTokenVar_  = CreateVariable();              // v19 ★ 本地 token
    commCmdInfoTokenVar_ = CreateVariable();              // v19 ★ 本地 token
    oneBufAddrVar_       = CreateVariable();              // 收口 ★ ready 写源 GM（host 填 R 个 int32=1）地址
    oneBufTokenVar_      = CreateVariable();              // 收口 ★ oneBuf 的 token（本地 buffer 独立 GetTokenInfo）
    localSrc_            = CreateLocalAddr();
    localDst_            = CreateLocalAddr();
    remoteDst_           = CreateRemoteAddr();
    event_               = CreateCompletedEvent();

    remoteCommWindowAddrs_.reserve(epWorldSize_);
    remoteCommWindowTokens_.reserve(epWorldSize_);
    std::vector<CcuRep::Variable> sendBytesPerPeer(epWorldSize_);
    peerOccBytes_.reserve(epWorldSize_);

    // v8 ★ per-expert 计数器（对齐库上 expand_idx "给同一专家发送的 token 个数"语义）
    expertOccCnts_.reserve(moeExpertNum_);
    for (uint32_t e = 0; e < moeExpertNum_; ++e) {
        CcuRep::Variable cnt = CreateVariable();
        cnt = 0;
        expertOccCnts_.push_back(cnt);
    }

    // v19 ★ 对端资源用 channel 的 XN 寄存器（CreateVariable(ch, XN_ID, &v)）；本地用普通 Variable。
    //   严格对齐 hccl mesh1d：每对端 commWindow 地址+token 各占一个 XN 槽（commCmdInfo 纯本地不交换）。
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
        CcuRep::Variable ob = CreateVariable(); ob = 0; peerOccBytes_.push_back(ob);
        CcuRep::Variable sb = CreateVariable(); sb = 0; sendBytesPerPeer[r]  = sb;
    }

    // ============ 2. LoadArgs ============（顺序严格 = GeneArgs）
    Load(expertIdsAddrVar_);
    Load(commWindowAddrVar_);
    Load(expandIdxAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(routingTableAddrVar_);
    Load(xWithMetaAddrVar_);
    Load(tokenMsgLenVar_);
    Load(xWithMetaTokenVar_);          // v19 ★
    Load(commWindowTokenVar_);         // v19 ★
    Load(commCmdInfoTokenVar_);        // v19 ★
    Load(oneBufAddrVar_);              // 收口 ★（顺序严格 = GeneArgs 末尾）
    Load(oneBufTokenVar_);             // 收口 ★
    // 本卡自己的"对端槽"用本卡真实地址/token 填（自发自路径走本地 token）
    remoteCommWindowAddrs_[epRankId_]   = commWindowAddrVar_;
    remoteCommWindowTokens_[epRankId_]  = commWindowTokenVar_;

    // tokenWithMetaLenVar_ = tokenMsgLen + 32（Variable = Variable + Variable 合法）
    CcuRep::Variable metaBytesVar = CreateVariable();
    metaBytesVar = static_cast<uint64_t>(METADATA_BYTES);
    tokenWithMetaLenVar_ = tokenMsgLenVar_ + metaBytesVar;

    // 常用立即数 Variable（替代非法的 Variable+立即数）
    CcuRep::Variable oneVar = CreateVariable();   oneVar = 1U;
    CcuRep::Variable elem4Var = CreateVariable(); elem4Var = static_cast<uint64_t>(sizeof(uint32_t)); // =4

    // ============ 2.5. Clear 上次 dispatch 的 0.5RTT 状态（起头清零）============
    CHK_RET(ClearAllPeersArriveNotify(MOE_DISPATCH_CCU_TAG));

    // ============ 3. PreSync（需求1 ★ 槽位 mask）============
    //   每对端 channel：用 2 个 NotifyRecord 把本卡 commWindow 地址+token 写到对端对应 XN 槽
    //   （remoteVarIdx=XN_ID），mask=1<<XN_ID（槽位掩码，非 rank）。
    //   等待：NotifyWait(ch, CKE_PRE, PRE_SLOT_BITS=0x3)——本 channel 2 个槽全到齐（OR）。
    //   v20.2：commCmdInfo 纯本地（段 10 本地写 + teardown 本地轮询），不跨卡，故不交换其地址/token。
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

    // ============ 4. BuildSlotBase（地址递进范式）============
    //   每段大小 = bs*topK*tokenWithMetaLen（host 立即数）；本卡段偏移 = epRankId * 段大小。
    //   mySlotBase = commWindowAddr + myBaseOff；remoteSlotBase[r] = remoteCommWindowAddr[r] + myBaseOff。
    uint64_t tokenWithMetaLenImm = tokenMsgLen_ + METADATA_BYTES;     // host 立即数
    CcuRep::Variable myBaseOffVar = CreateVariable();
    myBaseOffVar = static_cast<uint64_t>(epRankId_) * bs_ * topK_ * tokenWithMetaLenImm;
    // mySlotBase（Variable）= commWindowAddr + myBaseOff（Variable = Variable + Variable 合法）
    CcuRep::Variable mySlotBaseVar = CreateVariable();
    mySlotBaseVar = commWindowAddrVar_ + myBaseOffVar;
    std::vector<CcuRep::Variable> remoteSlotBaseVar(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        CcuRep::Variable v = CreateVariable();
        v = remoteCommWindowAddrs_[r] + myBaseOffVar;
        remoteSlotBaseVar[r] = v;
    }

    // ============ 5. PreCount（地址递进范式 ★）============
    //   维护 routingAddrVar 递进读 routingTable[idx]（不用 Variable+立即数）。
    CcuRep::Variable rankVarA = CreateVariable();
    CcuRep::Variable routingAddrA = CreateVariable();
    routingAddrA = routingTableAddrVar_;                  // Variable = Variable
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            LoadVariable(routingAddrA, rankVarA);         // 读 routingTable[idx]
            routingAddrA += elem4Var;                     // 递进 4 字节
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarA == r) {
                    sendBytesPerPeer[r] = sendBytesPerPeer[r] + tokenWithMetaLenVar_;
                }
            }
        }
    }

    // ============ 5.5.（已删）commCmdInfo 4 字节 overhead ============
    //   收口（方案 A）：tokenCount 不再跨卡 WriteNbWithArriveNotify 写（改由 teardown AIV 自数），
    //   段 8 已无对端 cmdInfo 写，sendBytes 回归"纯数据字节"（段 5 累加的 tokenWithMetaLen×token 数），
    //   不再 +4 cmdInfo overhead。0.5RTT wishCnt 与 WriteNbWithArriveNotify 的数据字节精确对账。

    // ============ 6. RecordExpectCount ============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(RecordExpectCount(channels_[channelIdx], sendBytesPerPeer[r], MOE_DISPATCH_CCU_TAG));
    }

    // ============ 7. DoMainLoop（token + 地址递进 ★）============
    //   维护 4 个递进地址 Variable：expertIdAddr / routingAddrB / xMetaAddr / expandIdxAddr。
    //   localSrc_.addr = xMetaAddr（递进），localSrc_.token = 本地 xWithMeta token。
    //   阶段 A：按 expertId 写 expand_idx（StoreVariable Variable 地址版，写本卡 HBM 无 token）。
    //   阶段 B：按 rank 发——自发自 LocalCopyNb(localDst.token=本地 commWindow token)，
    //          跨卡 WriteNbWithArriveNotify(remoteDst.token=对端 commWindow token)。
    CcuRep::Variable expertIdVar = CreateVariable();
    CcuRep::Variable rankVarB    = CreateVariable();
    CcuRep::Variable curOccVar   = CreateVariable();
    CcuRep::Variable expertIdAddr   = CreateVariable(); expertIdAddr   = expertIdsAddrVar_;
    CcuRep::Variable routingAddrB   = CreateVariable(); routingAddrB   = routingTableAddrVar_;
    CcuRep::Variable xMetaAddr      = CreateVariable(); xMetaAddr      = xWithMetaAddrVar_;
    CcuRep::Variable expandIdxAddr  = CreateVariable(); expandIdxAddr  = expandIdxAddrVar_;
    // localSrc_.token 固定为本地 xWithMeta token（数据源始终本地）
    localSrc_.token = xWithMetaTokenVar_;
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            LoadVariable(expertIdAddr, expertIdVar);      // 读 expertId[idx]
            expertIdAddr += elem4Var;
            LoadVariable(routingAddrB, rankVarB);         // 读 routingTable[idx]
            routingAddrB += elem4Var;
            // localSrc_.addr = xMetaAddr（当前 token 行起址）
            localSrc_.addr = xMetaAddr;                   // Address = Variable（合法）

            // ---- 阶段 A：按 expertId 维度写 expand_idx ----
            for (uint32_t e = 0; e < moeExpertNum_; ++e) {
                CCU_IF(expertIdVar == e) {
                    curOccVar = expertOccCnts_[e];
                    // expand_idx 写本卡 HBM 输出 buffer：用 hcomm 标准 StoreVariable 的 Variable 地址版
                    //   （legacy CcuContext::StoreVariable(Variable,Variable) 已实现、mc2_context 在用，
                    //    写本卡 HBM 无需 token）。pkg_inc CcuKernel 暂漏暴露该重载，已在 api.h 占位，
                    //    待 SE 补进 CcuKernel 后即用基类版（见 OPEN_QUESTIONS SE-2）。
                    StoreVariable(curOccVar, expandIdxAddr);
                    expertOccCnts_[e] = curOccVar + oneVar;
                }
            }
            expandIdxAddr += elem4Var;                    // 递进 expand_idx 写指针

            // ---- 阶段 B：按 rank 维度发送 token+metadata 整体（一次 RDMA tokenWithMetaLen 字节）----
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarB == r) {
                    if (r == epRankId_) {
                        // 自发自：本卡 commWindow[mySlotBase + peerOccBytes]，token=本地 commWindow token
                        localDst_.addr  = mySlotBaseVar;          // Address = Variable
                        localDst_.addr += peerOccBytes_[r];       // Address += Variable
                        localDst_.token = commWindowTokenVar_;
                        // ★ event 范式（对齐 all_gather_nhr1d_mem2mem:269-272 CCU_IF 内"发一条等一条"）：
                        //   CompletedEvent.mask 是"在途 DMA 位图"，每条 DMA 占 1 位。自发自 per-token
                        //   最多 bs*topK 条（可能 >32），无法用单 event 位图攒批一把等，故 SetMask(bit0)
                        //   + LocalCopyNb + 立即 WaitEvent 串行落地（bit0 用完即清，下条复用）。
                        //   跨卡走 WriteNbWithArriveNotify（0.5RTT，不占 event），故 event 仅服务自发自。
                        event_.SetMask(1U);
                        CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenWithMetaLenVar_, event_));
                        CHK_RET(WaitEvent(event_));
                    } else {
                        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
                        // 跨卡：对端 commWindow[remoteSlotBase + peerOccBytes]，token=对端 commWindow token
                        remoteDst_.addr  = remoteSlotBaseVar[r];
                        remoteDst_.addr += peerOccBytes_[r];
                        remoteDst_.token = remoteCommWindowTokens_[r];
                        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                        localSrc_, tokenWithMetaLenVar_,
                                                        MOE_DISPATCH_CCU_TAG));
                    }
                    peerOccBytes_[r] = peerOccBytes_[r] + tokenWithMetaLenVar_;
                }
            }
            xMetaAddr += tokenWithMetaLenVar_;            // 递进数据源指针
        }
    }

    // ============ 8.（已删）写 tokenCount 跨卡 + 本卡 ============
    //   收口（方案 A）：原本段 8 用自造 StoreVariableAt 把 sendTokens 落到 scratch GM 再
    //   WriteNbWithArriveNotify 跨卡写对端 tokenCount，自卡再 StoreVariableAt 自写——共 R 处自造 API。
    //   现全部删除：tokenCount 不再由发送方 CCU 跨卡产出，改由【接收方】dispatch_teardown AIV
    //   自数（遍历本卡 commWindow 段 r 的 slot，按 metadata.srcRank==r 计有效 token 数）。
    //   scratch 区随之取消，commCmdInfo 简化为只剩 ready 区。

    // ============ 9. WaitAllPeersArriveNotify（需求3 ★）============
    //   段 7 自发自 LocalCopyNb 已在 CCU_IF 内逐条 WaitEvent 落地，此处无需再等 event。
    //   只需等本卡作为接收方收齐：totalCntXn 达 R-1（其余对端都 WriteNbWithArriveNotify 写到位）。
    //   ★ SE 依赖：WaitAllPeersArriveNotify 的 target 语义需 SE 确认。当前按"对端到达计数=R-1"；
    //     若某对端给本卡发 0 token（routing 不均），其 arrive-notify 是否仍计入需 SE 明确——
    //     若按字节 wishCnt 匹配（0 字节自动满足）则无碍；若按对端个数计数则需把 target 改为
    //     "实际会给本卡发≥1 token 的对端数"（host 按列预算）。见 OPEN_QUESTIONS。
    CcuRep::Variable arriveTargetVar = CreateVariable();
    arriveTargetVar = static_cast<uint64_t>(epWorldSize_ - 1U);
    CHK_RET(WaitAllPeersArriveNotify(arriveTargetVar, MOE_DISPATCH_CCU_TAG));

    // ============ 10. 设 ready[0..R)=1（本卡收齐后本地置位）============
    //   收口（方案 A）：原本 R 次自造 StoreVariableAt(oneVar, readyAddr) 逐槽写 1，
    //   改为【1 次 hccl 现役 LocalCopyNb】：把本卡 oneBuf（GM，host 端填 R 个 int32=1）整块
    //   拷到 commCmdInfo ready 区，一次落 R 个 1。LocalAddr 两端 .token 都设（本地 buffer 本地 token）：
    //     dst = commCmdInfo ready 区（off=0），token = commCmdInfo token；
    //     src = oneBuf，token = oneBuf token；len = R*4 字节。
    //   commCmdInfo 简化布局：ready 区直接从 [0, R) 起（删 tokenCount/scratch 后最简）。
    //   teardown AIV 轮询 ready[0..R) 全 1 → 本卡数据齐。
    CcuRep::Variable readyBaseOff = CreateVariable();
    readyBaseOff = static_cast<uint64_t>(0U);              // ready 区从偏移 0 起（最简布局）
    CcuRep::Variable readyAddr = CreateVariable();
    readyAddr = commCmdInfoAddrVar_;                       // Address = Variable
    readyAddr += readyBaseOff;                             // Address += Variable（=0，保持地址范式一致）
    CcuRep::Variable rankSizeBytesVar = CreateVariable();
    rankSizeBytesVar = static_cast<uint64_t>(epWorldSize_) * sizeof(uint32_t);   // R*4 字节
    // LocalCopyNb 的 dst/src（LocalAddr）两端 token 都设
    CcuRep::LocalAddr readyDst = CreateLocalAddr();
    readyDst.addr  = readyAddr;
    readyDst.token = commCmdInfoTokenVar_;
    CcuRep::LocalAddr readySrc = CreateLocalAddr();
    readySrc.addr  = oneBufAddrVar_;
    readySrc.token = oneBufTokenVar_;
    // ready 单条 LocalCopyNb 占 bit0（DMA 前 SetMask 指定位，对齐 event 位图范式）
    event_.SetMask(1U);
    CHK_RET(LocalCopyNb(readyDst, readySrc, rankSizeBytesVar, event_));
    // 等 ready 区落地（bit0），保证 teardown AIV 能读到稳定的 ready=1。
    event_.SetMask(1U);
    CHK_RET(WaitEvent(event_));

    return HCCL_SUCCESS;
}

} // namespace hcomm
