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
 * \brief v11 combine_setup CCU 算法（双流并行架构，与 dispatch_setup 完全对称）。
 *
 * Algorithm 8 段式（同 dispatch_setup v11，方向反向）：
 *   1. InitResource
 *   2. LoadArgs                顺序 (expandX, commCmdInfo, yCombine, routingTable, tokenMsgLen)
 *   2.5 Clear                  起头清上次 0.5RTT 状态（v11 ★ 异步架构必需）
 *   3. PreSync                 交换 yCombine（CKE_IDX_0）+ commCmdInfo（CKE_IDX_1）地址
 *   4. BuildSlotBase           mySlotBaseVar / remoteSlotBaseVar[r]
 *   5. PreCount                按 routingTable 统计反向发给每个 dst rank 的字节数
 *   5.5 +commCmdInfo overhead  wishCntXn 加 4 字节（同 dispatch v10.1）
 *   6. RecordExpectCount       0.5RTT 用精确字节数设置对端 wishCntXn（tag = MOE_COMBINE_CCU_TAG）
 *   7. DoMainLoop              按 routingTable 反向发数据 + 写 expandIdx-like 索引
 *   8. PostNotify+Sync         跨卡写 commCmdInfo + 等本地完成
 *
 * 数据流（反向于 dispatch）：
 *   dispatch:  本卡 x[i]      → 对端 y[本卡段, 槽 k]
 *   combine :  本卡 expand_x[r 段, k 槽] → 对端 r 的 yCombine[本卡段, k 槽]
 *
 *   等价说法：dispatch 时本卡发给 rank r 的第 k 个 token，combine 时本卡把该 token 的
 *   expert 处理结果从 expand_x[r 段, k 槽] 取出，反向发回 rank r。
 *
 * 简化范围（v11）：
 *   - comm_quant_mode = 0：不做量化，quant_expand_x 输出 host 归零（device 端不写）
 *   - sharedExpertNum / sharedExpertRankNum = 0
 *   - 反向通信用与 dispatch 相同的 per-rank 槽位计数（peerOccBytes_ 字节累加器）
 *
 * 0.5RTT tag：MOE_COMBINE_CCU_TAG ≠ dispatch 的 tag，独立资源组互不干扰。
 */

#include "moe_distribute_combine_ccu_setup.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace hcomm {

// --------------------------- CcuKernelArg ----------------------------

CcuKernelArgMoeDistributeCombineCcuSetup::CcuKernelArgMoeDistributeCombineCcuSetup(
    uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
    uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
    uint32_t commQuantMode_)
    : epWorldSize(epWorldSize_), epRankId(epRankId_), moeExpertNum(moeExpertNum_),
      commQuantMode(commQuantMode_),
      bs(bs_), topK(topK_), hiddenSize(hiddenSize_),
      // tokenMsgLen 按 commQuantMode 切公式（与 dispatch_setup quantMode 分支严格一致）：
      //   commQuantMode==0 (BF16)：Align256(H)*sizeof(BF16)；==1 (MXFP8)：Align32(H+H/32)
      //   两端公式必须一致才能透传 RDMA 字节流（dispatch 发什么格式 combine 收同款）
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
    uint64_t expandXAddr_, uint64_t commCmdInfoAddr_, uint64_t commWindowAddr_)
    : expandXAddr(expandXAddr_), commCmdInfoAddr(commCmdInfoAddr_),
      commWindowAddr(commWindowAddr_)
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
    // GeneArgs 顺序 = device 端 LoadArgs 顺序（务必一致，否则 device 读错地址）。
    return {
        myArg.expandXAddr,
        myArg.commCmdInfoAddr,
        myArg.commWindowAddr,
        tokenMsgLen_,
    };
}

HcclResult CcuKernelMoeDistributeCombineCcuSetup::Algorithm()
{
    // ============ 1. InitResource ============
    // v16 ★ commWindowAddrVar_ 替代 yCombineAddrVar_；remoteCommWindowAddrs_ 替代 remoteYCombineAddrs_
    //      复用 dispatch 同一块 hccl commWindow buffer
    expandXAddrVar_      = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    commWindowAddrVar_   = CreateVariable();
    tokenMsgLenVar_      = CreateVariable();
    localSrc_            = CreateLocalAddr();
    localDst_            = CreateLocalAddr();
    remoteDst_           = CreateRemoteAddr();
    event_               = CreateCompletedEvent();

    remoteCommWindowAddrs_.reserve(epWorldSize_);
    std::vector<CcuRep::Variable> remoteCommCmdInfoAddrs(epWorldSize_);
    std::vector<CcuRep::Variable> sendBytesPerPeer(epWorldSize_);
    std::vector<CcuRep::Variable> sendTokensPerPeer(epWorldSize_);

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
        CcuRep::Variable sb = CreateVariable(); sb = 0; sendBytesPerPeer[r]  = sb;
        CcuRep::Variable st = CreateVariable(); st = 0; sendTokensPerPeer[r] = st;
    }

    // ============ 2. LoadArgs ============
    Load(expandXAddrVar_);
    Load(commCmdInfoAddrVar_);
    Load(commWindowAddrVar_);              // v16 ★ 替代 yCombineAddrVar_
    Load(tokenMsgLenVar_);
    remoteCommWindowAddrs_[epRankId_] = commWindowAddrVar_;
    remoteCommCmdInfoAddrs[epRankId_] = commCmdInfoAddrVar_;

    // ============ 2.5. Clear 上次 dispatch 的 0.5RTT 状态（v11 ★ 起头清零）============
    // 与 dispatch_setup v11 段 2.5 同款机制——但用 MOE_COMBINE_CCU_TAG（独立资源组）。
    CHK_RET(ClearAllPeersArriveNotify(MOE_COMBINE_CCU_TAG));

    // ============ 3. PreSync ============
    // v16 ★ 交换 commWindowAddr（替代 yCombineAddr）
    uint32_t selfBit = 1U << epRankId_;
    uint32_t allBits = ((1U << epWorldSize_) - 1U) & (~selfBit);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_0, CKE_IDX_0, commWindowAddrVar_,  selfBit));
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_IDX_1, CKE_IDX_1, commCmdInfoAddrVar_, selfBit));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_IDX_0, allBits));
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_IDX_1, allBits));
    }

    // ============ 4. BuildSlotBase ============
    // v16 ★ combine commWindow 改单段 layout：(bs*K, tokenWithMetaLen)，不分段，slot=srcSlot*K+dstK 直接索引
    //   - mySlotBaseVar = commWindowAddr（不加 epRankId 偏移，因为单段）
    //   - remoteSlotBaseVar[r] = remoteCommWindowAddrs[r]
    // 注意：物理上 commWindow 是 dispatch 共用的 (rankSize, bs*K, tokenWithMetaLen) buffer，
    //       combine 阶段仅用前 bs*K 个槽（= 1 段空间）—— host 端 combine_setup launch 前 memset 0xFF 清零
    CcuRep::Variable oneVar = CreateVariable();
    oneVar = 1U;
    uint64_t tokenWithMetaLenImm = tokenMsgLen_ + 32U;
    CcuRep::Variable mySlotBaseVar = commWindowAddrVar_;
    std::vector<CcuRep::Variable> remoteSlotBaseVar(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        remoteSlotBaseVar[r] = remoteCommWindowAddrs_[r];
    }

    // tokenWithMetaLenVar：段 5/7 用
    CcuRep::Variable metaBytesVar = CreateVariable();
    metaBytesVar = static_cast<uint64_t>(32);
    CcuRep::Variable tokenWithMetaLenVar = CreateVariable();
    tokenWithMetaLenVar = tokenMsgLenVar_ + metaBytesVar;

    // ============ 5. PreCount ============
    // v18 ★ FIX：host unroll 上界从 bs*K 改 A = bs * rankSize * min(localExpertNum, K)
    //   原因：本卡 expandX 紧密区 worst-case token 数 = A（op_def 标准），可能 > bs*K
    //   - 之前 bs*K 上界在 routing 极端不均衡时漏发 (A - bs*K) 个 token（demo 巧合掩盖）
    //   - 改 A 后覆盖所有可能 token；padding token srcRank=-1 自动 CCU_IF 跳过
    //   - A 是 host 编译期常量（KernelArg attr 派生）→ host unroll 合法
    //
    // wishCnt 累加：每 token tokenWithMetaLen 字节（含 metadata 一起发回）
    // sendTokensPerPeer 累加：用于段 8 写 commCmdInfo
    //
    // 注意：本卡 expandX 紧密区按 r 段（dispatch_teardown 顺序）顺序排，
    //   段 5 遍历顺序与段 7 完全一致（都是 idx 0..A-1），保证 wishCnt 与 wtCount 累加对账精确。
    const uint32_t aBound = bs_ * epWorldSize_
                          * ((moeExpertPerRankNum_ < topK_) ? moeExpertPerRankNum_ : topK_);
    CcuRep::Variable srcRankVarA = CreateVariable();
    for (uint32_t idx = 0; idx < aBound; ++idx) {
        // 读 expandX[idx] 末尾 metadata 的 srcRank（int32 在 metadata 块第 0 个 4B 位置）
        LoadVariable(expandXAddrVar_
                     + static_cast<uint64_t>(idx) * tokenWithMetaLenImm
                     + tokenMsgLen_,
                     srcRankVarA);
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            CCU_IF(srcRankVarA == r) {
                sendBytesPerPeer[r]  = sendBytesPerPeer[r] + tokenWithMetaLenVar;
                sendTokensPerPeer[r] = sendTokensPerPeer[r] + oneVar;
            }
        }
    }

    // ============ 5.5. commCmdInfo 4 字节 overhead（与 dispatch v10.1 同款修复）============
    // 段 5 已按 per-token tokenWithMetaLen 累加 sendBytesPerPeer，这里再加段 8 的 4B 跨卡 cmdInfo
    CcuRep::Variable cmdInfoBytesVar = CreateVariable();
    cmdInfoBytesVar = static_cast<uint64_t>(sizeof(uint32_t));
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        sendBytesPerPeer[r] = sendBytesPerPeer[r] + cmdInfoBytesVar;
    }

    // ============ 6. RecordExpectCount ============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(RecordExpectCount(channels_[channelIdx], sendBytesPerPeer[r], MOE_COMBINE_CCU_TAG));
    }

    // ============ 7. DoMainLoop ============
    // v16 ★ 完全重写：per-token + metadata.dstByteOff 直接定位（不再用 peerOccBytes 递增）
    //
    // metadata layout（32 字节）:
    //   [0..4):   srcRank (int32)   ← dst rank = combine 时反向目的卡（dispatch 时这个 token 的 src）
    //   [4..8):   srcSlot (int32)   ← srcRank 视角的 bs 索引 i (combine_setup 不直接用)
    //   [8..12):  dstK (int32)      ← srcRank 视角的 K 索引 j (combine_setup 不直接用)
    //   [12..16): dstByteOff (int32) ← host 预算 = (srcSlot*K + dstK) * tokenWithMetaLen
    //                                 直接作为对端 commWindow 的字节偏移（不需 device 乘法）
    //
    // 算法：遍历本卡 expandX 紧密区每个 token
    //   - 读 srcRank → 决定写到本卡还是对端
    //   - 读 dstByteOff → 直接作为对端 commWindow 的目标偏移
    //   - 一次 WriteNb(tokenWithMetaLen) 把 token+metadata 整体送过去
    //
    // commWindow 结果 layout（按用户描述）：
    //   slot=i*K+0..i*K+K-1 都是 token i 的 K 个副本（紧密 K 个相邻槽）
    //   slot=0..K-1: token 0 的副本；slot=K..2K-1: token 1 的副本；...
    //
    // padding token (srcRank=-1) 不命中任何 r ∈ [0, rankSize) → 自动跳过
    // v18 ★ FIX：host unroll 上界从 bs*K 改 aBound = A（与段 5 一致），
    //   覆盖 expandX 紧密区所有可能 token (worst-case A 个，可能 > bs*K)
    CcuRep::Variable srcRankVarB    = CreateVariable();
    CcuRep::Variable dstByteOffVar  = CreateVariable();
    for (uint32_t idx = 0; idx < aBound; ++idx) {
        // 读 metadata.srcRank（偏移 = idx*tokenWithMetaLen + tokenMsgLen + 0）
        LoadVariable(expandXAddrVar_
                     + static_cast<uint64_t>(idx) * tokenWithMetaLenImm
                     + tokenMsgLen_,
                     srcRankVarB);
        // 读 metadata.dstByteOff（偏移 = idx*tokenWithMetaLen + tokenMsgLen + 12）
        LoadVariable(expandXAddrVar_
                     + static_cast<uint64_t>(idx) * tokenWithMetaLenImm
                     + tokenMsgLen_ + 12U,
                     dstByteOffVar);
        // localSrc 指向整个 token（data + metadata）
        localSrc_.addr = expandXAddrVar_
                         + static_cast<uint64_t>(idx) * tokenWithMetaLenImm;

        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            CCU_IF(srcRankVarB == r) {
                if (r == epRankId_) {
                    // 自反自：本卡 commWindow[dstByteOff]（数据流向自己）
                    localDst_.addr = mySlotBaseVar + dstByteOffVar;
                    CHK_RET(LocalCopyNb(localDst_, localSrc_, tokenWithMetaLenVar, event_));
                } else {
                    // 跨卡反向：对端 r 的 commWindow[dstByteOff]
                    uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
                    remoteDst_.addr = remoteSlotBaseVar[r] + dstByteOffVar;
                    CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                                     localSrc_, tokenWithMetaLenVar,
                                                     MOE_COMBINE_CCU_TAG));
                }
            }
        }
    }

    // ============ 8. PostNotify+Sync ============
    // 跨卡写 commCmdInfo[本卡 epRankId] = 本卡反向发给该对端的字节数（用于 combine_teardown
    // 知道每个 src 发来多少字节）。沿用 dispatch v10.1：用 WriteNbWithArriveNotify 让 0.5RTT
    // 等齐机制覆盖此次通信（避免 v10 同步漏洞）。
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
        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_, localTokenBuf,
                                         elemBytesVar, MOE_COMBINE_CCU_TAG));
    }

    event_.SetMask(0xFFFFFFFFU);
    CHK_RET(WaitEvent(event_));

    StoreVariable(sendTokensPerPeer[epRankId_],
                  commCmdInfoAddrVar_ + static_cast<uint64_t>(epRankId_) * sizeof(uint32_t));

    // v11 异步：不调 WaitAllPeersArriveNotify；Clear 已经放在段 2.5 起头。
    return HCCL_SUCCESS;
}

} // namespace hcomm
