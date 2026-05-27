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
      // v17 ★ tokenMsgLen 按 quantMode 切公式：
      //   quantMode == 0 (非量化 BF16)：Align256(H) * sizeof(BF16)
      //     来源：aclnnMoeDistributeDispatchSetup.md §约束说明 5
      //   quantMode == 1 (MXFP8)：Align32(H + H/32)
      //     - data 段 H bytes (FP8 = 1 byte/elem)
      //     - scale 段 H/32 bytes (E8M0 per-32-elem 共享，每 block 1 byte)
      //     - 整体 Align32 满足 ascendc DataCopy 32B 对齐
      //   device 端字节透传：dispatch_setup CCU 算法不感知 quantMode，
      //     段 4 段长 / 段 7 单 token RDMA 长度 / metadata 偏移全部用 tokenMsgLen 变量，
      //     host 端按 quantMode 预拼好 xWithMeta 即可（data + scale 紧排到 token 内部）。
      tokenMsgLen(quantMode_ == 0
                  ? static_cast<uint64_t>(((hiddenSize_ + 255U) / 256U) * 256U) * sizeof(uint16_t)
                  : static_cast<uint64_t>(((hiddenSize_ + hiddenSize_ / 32U + 31U) / 32U) * 32U)),
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
    uint64_t commWindowAddr_, uint64_t routingTableAddr_,
    uint64_t xWithMetaAddr_)
    : xAddr(xAddr_), expertIdsAddr(expertIdsAddr_),
      scalesAddr(scalesAddr_), xActiveMaskAddr(xActiveMaskAddr_),
      yAddr(yAddr_), expandIdxAddr(expandIdxAddr_), commCmdInfoAddr(commCmdInfoAddr_),
      commWindowAddr(commWindowAddr_), routingTableAddr(routingTableAddr_),
      xWithMetaAddr(xWithMetaAddr_)
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
    // v13 ★ 关键变更：用 commWindowAddr 替代 yAddr（op_def yAddr 字段语义错位修复）。
    //   op_def yAddr 是 caller 占位 alloc（src 端发送缓冲，库上语义），device 端 v13 不读不写。
    //   对端接收 window 改用 v13 新增的 commWindowAddr 字段（工程内部 scratch）。
    // v15 ★ 新增 xWithMetaAddr：host 预拼装 (x[i] data + (selfRank,i,j) metadata) 后的 buffer，
    //   段 7 阶段 B 直接从这个 buffer 一次 RDMA 把 token + metadata 整体发送。
    //   xAddr 仍保留 op_def 占位但 device 端 v15 不再直接读取。
    return {
        myArg.xAddr,                  // v15 device 端不读但保留契约
        myArg.expertIdsAddr,
        myArg.commWindowAddr,
        myArg.expandIdxAddr,
        myArg.commCmdInfoAddr,
        myArg.routingTableAddr,
        myArg.xWithMetaAddr,          // v15 ★ device 端段 7 阶段 B 真正读取的 src buffer
        tokenMsgLen_,
    };
}

HcclResult CcuKernelMoeDistributeDispatchCcuSetup::Algorithm()
{
    // ============ 1. InitResource ============
    xAddrVar_            = CreateVariable();
    expertIdsAddrVar_    = CreateVariable();                  // v8 ★
    commWindowAddrVar_            = CreateVariable();
    expandIdxAddrVar_    = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    routingTableAddrVar_ = CreateVariable();
    xWithMetaAddrVar_    = CreateVariable();                  // v15 ★
    tokenMsgLenVar_      = CreateVariable();
    tokenWithMetaLenVar_ = CreateVariable();                  // v15 ★
    localSrc_            = CreateLocalAddr();
    localDst_            = CreateLocalAddr();
    remoteDst_           = CreateRemoteAddr();
    event_               = CreateCompletedEvent();

    remoteCommWindowAddrs_.reserve(epWorldSize_);
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
            remoteCommWindowAddrs_.push_back(CreateVariable());
            remoteCommCmdInfoAddrs[r] = CreateVariable();
        } else {
            uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
            CcuRep::Variable vy, vc;
            CHK_RET(CreateVariable(channels_[channelIdx], CKE_IDX_0, &vy));
            CHK_RET(CreateVariable(channels_[channelIdx], CKE_IDX_1, &vc));
            remoteCommWindowAddrs_.push_back(vy);
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
    Load(commWindowAddrVar_);
    Load(expandIdxAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(routingTableAddrVar_);
    Load(xWithMetaAddrVar_);                                  // v15 ★ 新增 Load 顺序对齐 GeneArgs
    Load(tokenMsgLenVar_);
    remoteCommWindowAddrs_[epRankId_] = commWindowAddrVar_;
    remoteCommCmdInfoAddrs[epRankId_] = commCmdInfoAddrVar_;

    // v15 ★ tokenWithMetaLenVar_ = tokenMsgLen + 32（CCU 严格清单：Var = Var + Var，所以用 helper）
    CcuRep::Variable metaBytesVar = CreateVariable();
    metaBytesVar = static_cast<uint64_t>(32);    // 三元组 12B + padding 到 32B 对齐
    tokenWithMetaLenVar_ = tokenMsgLenVar_ + metaBytesVar;

    // ============ 2.5. Clear 上次 dispatch 的 0.5RTT 状态（v11 ★ 起头清零）============
    // v11 架构纠正：setup 异步 + teardown AIV 内部等齐（轮询 commCmdInfo）。
    //   Clear 时机选择：
    //     - setup 末尾 → race（累加可能还在进行）
    //     - teardown → AIV 不能调 0.5RTT API
    //     - setup 起头 → stream 顺序保证此时上次 dispatch 已彻底完成 ✓
    //   时序证明：上次 teardown 等齐 commCmdInfo（≥ 上次 setup 段 8 RDMA 完成）
    //            → FIFO RDMA 保证 上次 setup 段 7 的 0.5RTT 累加也已完成
    //            → stream sync → 本次 setup 起头 Clear 时累加全部完成 ★ 安全
    //   第一次调用时状态本就是 0，Clear 是 no-op。
    CHK_RET(ClearAllPeersArriveNotify(MOE_DISPATCH_CCU_TAG));

    // ============ 3. PreSync ============
    uint32_t selfBit = 1U << epRankId_;
    uint32_t allBits = ((1U << epWorldSize_) - 1U) & (~selfBit);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_0, CKE_IDX_0, commWindowAddrVar_,           selfBit));
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_1, CKE_IDX_1, commCmdInfoAddrVar_, selfBit));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_IDX_0, allBits));
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_IDX_1, allBits));
    }

    // ============ 4. BuildSlotBase ============
    // oneVar：替代 "Var = Var + 1U" 中的立即数。
    CcuRep::Variable oneVar = CreateVariable();
    oneVar = 1U;
    // v15 ★ 每段大小公式从 tokenMsgLen 改为 tokenWithMetaLen（每槽含 token data + 32B metadata 三元组）：
    //   段内 slot 编号 = peerOccBytes[r] / tokenWithMetaLen
    //   段长 = bs * topK * tokenWithMetaLen 字节（host 端立即数预算）
    uint64_t tokenWithMetaLenImm = tokenMsgLen_ + 32U;     // host 端立即数
    CcuRep::Variable myBaseOffVar = CreateVariable();
    myBaseOffVar = static_cast<uint64_t>(epRankId_) * bs_ * topK_ * tokenWithMetaLenImm;
    CcuRep::Variable mySlotBaseVar = CreateVariable();
    mySlotBaseVar = commWindowAddrVar_ + myBaseOffVar;
    std::vector<CcuRep::Variable> remoteSlotBaseVar(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        CcuRep::Variable v = CreateVariable();
        v = remoteCommWindowAddrs_[r] + myBaseOffVar;
        remoteSlotBaseVar[r] = v;
    }

    // ============ 5. PreCount ============
    // v15 ★ 每 token 写量从 tokenMsgLen 改为 tokenWithMetaLen（含 32B metadata 三元组）。
    CcuRep::Variable rankVarA = CreateVariable();
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            uint32_t idx = i * topK_ + j;
            LoadVariable(routingTableAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t), rankVarA);
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarA == r) {
                    sendBytesPerPeer[r]  = sendBytesPerPeer[r] + tokenWithMetaLenVar_;
                    sendTokensPerPeer[r] = sendTokensPerPeer[r] + oneVar;
                }
            }
        }
    }

    // ============ 5.5. 加 commCmdInfo 跨卡写的 4 字节 overhead（v10.1 ★ 同步漏洞修复）============
    // v10 发现：段 8 commCmdInfo 跨卡写用普通 WriteNb 不累加 0.5RTT wtCount，
    //   导致 setup 末尾段 9 WaitAllPeersArriveNotify 等齐只覆盖段 7 数据写、不覆盖段 8 cmd 写。
    //   teardown AIV 读 commCmdInfo[r != self] 时对端 commCmdInfo 可能未到位 → 读到旧值。
    // v10.1 修复：段 8 改用 WriteNbWithArriveNotify（见下方段 8），同时这里把每对端预期字节数
    //   +4（commCmdInfo 4 字节），让 wishCntXn 与对端 wtCount（含段 7 数据 + 段 8 cmd）精确匹配。
    CcuRep::Variable cmdInfoBytesVar = CreateVariable();
    cmdInfoBytesVar = static_cast<uint64_t>(sizeof(uint32_t));   // = 4
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;   // 自发自不走 0.5RTT
        sendBytesPerPeer[r] = sendBytesPerPeer[r] + cmdInfoBytesVar;   // Var = Var + Var，合法
    }

    // ============ 6. RecordExpectCount ============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(RecordExpectCount(channels_[channelIdx], sendBytesPerPeer[r], MOE_DISPATCH_CCU_TAG));
    }

    // ============ 7. DoMainLoop ============
    // v8 拆双阶段：A=按 expertId 写 expand_idx, B=按 rank 发数据
    // v15 ★ 阶段 B 改造：
    //   - localSrc 从 xAddr[i] 改为 xWithMetaAddr[idx]（idx = i*K+j）：host 预拼装好的 token+metadata
    //   - 单次 WriteNb/LocalCopyNb 长度 tokenMsgLen → tokenWithMetaLen（一次 RDMA 把 data+meta 整体送过去）
    //   - peerOccBytes 累加单位 tokenMsgLen → tokenWithMetaLen（段内 slot 仍连续）
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
            // v15 ★ localSrc 从 host 预拼装的 xWithMeta[idx] 读，已含 (selfRank, i, j) metadata
            //   xWithMeta layout: (Bs*K, tokenWithMetaLen)，每行 idx = i*K+j 对应一个唯一副本
            localSrc_.addr = xWithMetaAddrVar_ + static_cast<uint64_t>(idx) * tokenWithMetaLenImm;

            // ---- 阶段 A：按 expertId 维度写 expand_idx ----
            for (uint32_t e = 0; e < moeExpertNum_; ++e) {
                CCU_IF(expertIdVar == e) {
                    curOccVar = expertOccCnts_[e];
                    StoreVariable(curOccVar,
                                  expandIdxAddrVar_ + static_cast<uint64_t>(idx) * sizeof(uint32_t));
                    expertOccCnts_[e] = curOccVar + oneVar;
                }
            }

            // ---- 阶段 B：按 rank 维度发送 token+metadata 整体（v15 一次 RDMA tokenWithMetaLen 字节）----
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarB == r) {
                    if (r == epRankId_) {
                        localDst_.addr = mySlotBaseVar + peerOccBytes_[r];
                        CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenWithMetaLenVar_, event_));
                    } else {
                        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
                        remoteDst_.addr = remoteSlotBaseVar[r] + peerOccBytes_[r];
                        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                         localSrc_, tokenWithMetaLenVar_,
                                                         MOE_DISPATCH_CCU_TAG));
                    }
                    peerOccBytes_[r] = peerOccBytes_[r] + tokenWithMetaLenVar_;
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
        // v10.1 ★ 同步漏洞修复：原 v10 这里用 WriteNb（普通），不累加对端 wtCount，
        //   导致段 9 WaitAllPeersArriveNotify 不能等齐 commCmdInfo 跨卡写。
        //   改为 WriteNbWithArriveNotify：4 字节也 atomic add 到对端 wtCount，
        //   配合段 5.5 的 sendBytesPerPeer[r] += 4，wishCntXn 与 wtCount 精确匹配。
        //   这样 teardown AIV launch 时 commCmdInfo[r != self] 必然已就位。
        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_, localTokenBuf,
                                         elemBytesVar, MOE_DISPATCH_CCU_TAG));
    }

    event_.SetMask(0xFFFFFFFFU);
    CHK_RET(WaitEvent(event_));

    StoreVariable(sendTokensPerPeer[epRankId_],
                  commCmdInfoAddrVar_ + static_cast<uint64_t>(epRankId_) * sizeof(uint32_t));

    // v11 ★ 删除 v10 的段 9（WaitAllPeersArriveNotify + ClearAllPeersArriveNotify）。
    //   v10 的同步阻塞设计违反了 0.5RTT 异步语义（库上 docs："通信指令发出后即刻退出"）。
    //   v11 恢复异步：
    //     - setup 不等齐，立即返回 → setup/teardown 间可以夹 matmul 等其他 kernel
    //     - 等齐工作移到 teardown AIV 内部（轮询本卡 commCmdInfo 等 sentinel 被覆盖）
    //     - Clear 移到本函数段 2.5（起头清上次残留）
    return HCCL_SUCCESS;
}

} // namespace hcomm
