/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  combine_setup_ccu.cpp —— CCU combine_setup 设备侧算法（真实逐 token 送回，A6 视角，mxfp8）。
 *  与 dispatch_setup_ccu.cpp 完全对称，方向反转：源 = expandX(A 行)，按 host 预算 assistTable 送回
 *  owner 卡 combine 数据区（块 ge、位 expand_idx）。padding 行 ownerRank=-1(0xFFFF..FF) 哨兵（host BuildCombineAssist 写）→ CCU_IF(ownerVar==r) 对任意 r∈[0,R) 全 64 位 EQ 恒不等 → 全假、自动跳过。
 *  段式/地址范式/8字节铁律同 dispatch（见其头注释与 SETUP_DESIGN.md §3/§4）。
 */

#include "combine_setup_ccu.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace mc2 {

namespace {
constexpr uint32_t DATA_ADDR_XN   = 0;
constexpr uint32_t DATA_TOKEN_XN  = 1;
constexpr uint32_t SCALE_ADDR_XN  = 2;
constexpr uint32_t SCALE_TOKEN_XN = 3;
constexpr uint32_t PRE_SLOT_BITS  = (1U << DATA_ADDR_XN) | (1U << DATA_TOKEN_XN) |
                                    (1U << SCALE_ADDR_XN) | (1U << SCALE_TOKEN_XN);   // = 0xF
constexpr uint32_t CKE_PRE = 0;
constexpr uint64_t ROUTE_ROW_BYTES = 3U * sizeof(uint64_t);   // assistTable 每行 3×int64 = 24
}  // namespace

HcclResult CcuKernelCombineSetup::Algorithm()
{
#ifdef COMBINE_CCU_A6
    // ============ 1. InitResource ============
    expandXDataAddrVar_  = CreateVariable();
    expandXScaleAddrVar_ = CreateVariable();
    assistTableAddrVar_  = CreateVariable();
    combineDataAddrVar_  = CreateVariable();
    combineScaleAddrVar_ = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    oneBufAddrVar_       = CreateVariable();
    expandXDataTokenVar_ = CreateVariable();
    expandXScaleTokenVar_= CreateVariable();
    assistTableTokenVar_ = CreateVariable();
    combineDataTokenVar_ = CreateVariable();
    combineScaleTokenVar_= CreateVariable();
    commCmdInfoTokenVar_ = CreateVariable();
    oneBufTokenVar_      = CreateVariable();
    localSrcData_  = CreateLocalAddr();  localSrcScale_ = CreateLocalAddr();
    localDstData_  = CreateLocalAddr();  localDstScale_ = CreateLocalAddr();
    remoteDstData_ = CreateRemoteAddr(); remoteDstScale_= CreateRemoteAddr();
    event_ = CreateCompletedEvent();
    CHK_RET(CreateBlockCcuBuf(1U, &localCopyBuf_));

    remoteDataAddrs_.reserve(epWorldSize_);   remoteDataTokens_.reserve(epWorldSize_);
    remoteScaleAddrs_.reserve(epWorldSize_);  remoteScaleTokens_.reserve(epWorldSize_);
    sendBytesPerPeer_.reserve(epWorldSize_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) {
            remoteDataAddrs_.push_back(CreateVariable());   remoteDataTokens_.push_back(CreateVariable());
            remoteScaleAddrs_.push_back(CreateVariable());  remoteScaleTokens_.push_back(CreateVariable());
        } else {
            uint32_t ch = (r < epRankId_) ? r : r - 1;
            CcuRep::Variable vDA, vDT, vSA, vST;
            CHK_RET(CreateVariable(channels_[ch], DATA_ADDR_XN,   &vDA));
            CHK_RET(CreateVariable(channels_[ch], DATA_TOKEN_XN,  &vDT));
            CHK_RET(CreateVariable(channels_[ch], SCALE_ADDR_XN,  &vSA));
            CHK_RET(CreateVariable(channels_[ch], SCALE_TOKEN_XN, &vST));
            remoteDataAddrs_.push_back(vDA);   remoteDataTokens_.push_back(vDT);
            remoteScaleAddrs_.push_back(vSA);  remoteScaleTokens_.push_back(vST);
        }
        CcuRep::Variable sb = CreateVariable(); sb = static_cast<uint64_t>(0);
        sendBytesPerPeer_.push_back(sb);
    }

    // ============ 2. LoadArgs（顺序严格 = GeneArgs）============
    Load(expandXDataAddrVar_); Load(expandXScaleAddrVar_); Load(assistTableAddrVar_);
    Load(combineDataAddrVar_); Load(combineScaleAddrVar_); Load(commCmdInfoAddrVar_); Load(oneBufAddrVar_);
    Load(expandXDataTokenVar_);Load(expandXScaleTokenVar_);Load(assistTableTokenVar_);
    Load(combineDataTokenVar_);Load(combineScaleTokenVar_);Load(commCmdInfoTokenVar_);Load(oneBufTokenVar_);
    remoteDataAddrs_[epRankId_]  = combineDataAddrVar_;   remoteDataTokens_[epRankId_]  = combineDataTokenVar_;
    remoteScaleAddrs_[epRankId_] = combineScaleAddrVar_;  remoteScaleTokens_[epRankId_] = combineScaleTokenVar_;

    CcuRep::Variable elem8Var     = CreateVariable(); elem8Var     = static_cast<uint64_t>(sizeof(uint64_t));  // 8
    CcuRep::Variable rowStrideVar = CreateVariable(); rowStrideVar = static_cast<uint64_t>(ROUTE_ROW_BYTES);   // 24
    CcuRep::Variable hBytesVar    = CreateVariable(); hBytesVar    = hBytes_;
    CcuRep::Variable scaleBytesVar= CreateVariable(); scaleBytesVar= scaleBytes_;
    CcuRep::Variable tokBothVar   = CreateVariable(); tokBothVar   = hBytesVar + scaleBytesVar;

    // ============ 2.5. Clear ============
    CHK_RET(clearallpeersarriveNotify(COMBINE_SETUP_CCU_TAG));

    // ============ 3. PreSync：广播本卡 combine data/scale 接收区 addr+token（4 XN 槽）============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t ch = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, DATA_ADDR_XN,   combineDataAddrVar_,   1U << DATA_ADDR_XN));
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, DATA_TOKEN_XN,  combineDataTokenVar_,  1U << DATA_TOKEN_XN));
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, SCALE_ADDR_XN,  combineScaleAddrVar_,  1U << SCALE_ADDR_XN));
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, SCALE_TOKEN_XN, combineScaleTokenVar_, 1U << SCALE_TOKEN_XN));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t ch = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[ch], CKE_PRE, PRE_SLOT_BITS));
    }

    // ★★★★★ [R≥4 优化项 · 客观定档 defer 至 a6+R≥4] 段5/段7 内层 for r∈[0,R) CCU_IF(ownerVar==r) 是 R 宽运行时
    //   比较梯 → 静态 trace = A_·R = R²·bs·min(topk,E)，比 A6-proven 参考多一个 R 因子（参考只 R·bs·topk）。
    //   · R=2 目标零惩罚（比值 = R·min(topk,E)/topk = 2·4/8 = 1.0）；R≥4 才放大（R=4→2×, R=8→4×）。
    //   · 当前 row-major 结构【输出 docx-correct】：combineData 落点与已测 combine_teardown 字节级一致（self-check 9/9）。
    //   扩到 R≥4 前，照参考 moe_distribute_combine_ccu_setup.cpp 段7(266-296) 重构为 owner-major（Option B，不动已测 teardown）：
    //     (1) host BuildCombineAssist 改产【按目的卡 src 分段连续】的 assist（卡0 的行、卡1 的行…依次紧密），
    //         每行加一个 expandX 源行偏移字段（现 expert-major 源行交错，kernel 需按该偏移随机访问 expandX）；
    //         分段后 ownerRank 字段可省——owner=段号，编译期已知。
    //     (2) kernel 段5/段7 改：外层 for r∈[0,R)（owner=r 编译期常量、self/cross 走 host 编译期 if，非 CCU_IF），
    //         内层只遍历发往卡 r 的 cnt[r] 行、单个 CCU_IF(idxVar!=0) 倒计数守门 → trace 降回 R·bs·topk。
    //     (3) self-check 同步改 owner 分段模型后，必 a5 复跑 9/9 坐实 host 地址仍闭合。
    //   注：kernel 本体 a6-only、a5 编不了，故 defer（现在改=零收益 + 不可验的新集成代码，负 EV）。与 docx 契约层零冲突。
    // ============ 5. PreCount：读 assistTable ownerRank 累加 sendBytesPerPeer（padding 行 owner=-1 全假、不计）============
    // ★ 指令收敛（修 ret=7=HCCL_E_UNAVAIL，与 dispatch_setup 同治）：原 host for(m<A_) 把循环体静态展开成
    //   O(A_·R) 份：每个 CCU_IF 内部申请 1 XN(跳转 targetVar)、数据面原语按发射次数占 XN/CKE → register 期
    //   CheckResIfAvailable(ccu_kernel_mgr.cc:208) 统计 xnReq/ckeReq 超 die 容量 → UNAVAIL(7)（a6 实测 dispatch 侧
    //   xnReq[366]/ckeReq[33]，combine 同机制）。改 CCU_WHILE 运行时循环：body 只发一份 → XN/CKE 塌回 O(R)。语义等价。
    //   A_≥1 恒真；标准式 CCU_WHILE(=CcuWhileBegin 头判)对 0 迭代安全。
    //   注：R≥4 owner-major 重构（见上 defer 注释）是另一独立优化项，收 R² trace、此处不动。
    CcuRep::Variable ownerVarA  = CreateVariable();
    CcuRep::Variable routeAddrA = CreateVariable(); routeAddrA = assistTableAddrVar_;
    CcuRep::Variable oneVarA    = CreateVariable(); oneVarA = static_cast<uint64_t>(1);
    CcuRep::Variable cntA       = CreateVariable(); cntA    = static_cast<uint64_t>(0);
    CCU_WHILE(cntA != static_cast<uint64_t>(A_)) {
        LoadVariable(routeAddrA, ownerVarA);
        routeAddrA += rowStrideVar;
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            CCU_IF(ownerVarA == r) {
                sendBytesPerPeer_[r] = sendBytesPerPeer_[r] + tokBothVar;
            }
        }
        cntA += oneVarA;                          // 运行期计数：0→A_
    }

    // ============ 6. RecordExpectCount 已下移到段 7 之后（见下方，与 dispatch_setup 同治）============
    //   ★ 专业人士反馈 + 根因坐实：record_expect_count 必须放在【所有数据发完之后】
    //     （WriteNbWithArriveNotify + LocalCopyNb 全部发射完成）。原实现放在写之前 → 大批量时对端
    //     totalCntXn 判定时序错、cross 回传部分未交付。sendBytesPerPeer_ 段 5 已算好（Variable 持久）。

    // ============ 7. DoMainLoop：逐 expandX 行送回 owner combine 数据区（payload+scale）============
    CcuRep::Variable ownerVarB  = CreateVariable();
    CcuRep::Variable dataOffVar = CreateVariable();
    CcuRep::Variable scaleOffVar= CreateVariable();
    CcuRep::Variable routeAddrB = CreateVariable(); routeAddrB = assistTableAddrVar_;
    CcuRep::Variable xDataRow   = CreateVariable(); xDataRow   = expandXDataAddrVar_;
    CcuRep::Variable xScaleRow  = CreateVariable(); xScaleRow  = expandXScaleAddrVar_;
    localSrcData_.token  = expandXDataTokenVar_;
    localSrcScale_.token = expandXScaleTokenVar_;
    // ★ 指令收敛（修 ret=7）：原 host for(m<A_) 展开 O(A_·R) 条微码 → register 撑爆 CCU 资源。
    //   改成 CCU_WHILE 运行时循环：body 只发一份 → 塌回 O(R)。源行按 m 逐行步进（combine 每行一 token 送回）。
    //   for(r)CCU_IF 保留（channel 编译期常量、不可运行时索引）；padding 行 owner=-1 由 CCU_IF 全假自动跳过、语义不变。
    CcuRep::Variable oneVarB = CreateVariable(); oneVarB = static_cast<uint64_t>(1);
    CcuRep::Variable cntB    = CreateVariable(); cntB    = static_cast<uint64_t>(0);
    CCU_WHILE(cntB != static_cast<uint64_t>(A_)) {
        LoadVariable(routeAddrB, ownerVarB);  routeAddrB += elem8Var;
        LoadVariable(routeAddrB, dataOffVar); routeAddrB += elem8Var;
        LoadVariable(routeAddrB, scaleOffVar);routeAddrB += elem8Var;
        localSrcData_.addr  = xDataRow;
        localSrcScale_.addr = xScaleRow;
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            CCU_IF(ownerVarB == r) {          // padding 行 owner=-1(0xFFFF..)：全 64 位 EQ 对任意 r∈[0,R) 恒不等 → 全假、跳过
                if (r == epRankId_) {
                    localDstData_.addr = combineDataAddrVar_; localDstData_.addr += dataOffVar;
                    localDstData_.token = combineDataTokenVar_;
                    event_.SetMask(1U);
                    CHK_RET(LocalCopyNb(localDstData_, localSrcData_, hBytesVar, &localCopyBuf_, 1U, event_));
                    CHK_RET(WaitEvent(event_));
                    localDstScale_.addr = combineScaleAddrVar_; localDstScale_.addr += scaleOffVar;
                    localDstScale_.token = combineScaleTokenVar_;
                    event_.SetMask(1U);
                    CHK_RET(LocalCopyNb(localDstScale_, localSrcScale_, scaleBytesVar, &localCopyBuf_, 1U, event_));
                    CHK_RET(WaitEvent(event_));
                } else {
                    uint32_t ch = (r < epRankId_) ? r : r - 1;
                    remoteDstData_.addr = remoteDataAddrs_[r]; remoteDstData_.addr += dataOffVar;
                    remoteDstData_.token = remoteDataTokens_[r];
                    CHK_RET(WriteNbWithArriveNotify(channels_[ch], remoteDstData_, localSrcData_,
                                                    hBytesVar, COMBINE_SETUP_CCU_TAG));
                    remoteDstScale_.addr = remoteScaleAddrs_[r]; remoteDstScale_.addr += scaleOffVar;
                    remoteDstScale_.token = remoteScaleTokens_[r];
                    CHK_RET(WriteNbWithArriveNotify(channels_[ch], remoteDstScale_, localSrcScale_,
                                                    scaleBytesVar, COMBINE_SETUP_CCU_TAG));
                }
            }
        }
        xDataRow  += hBytesVar;
        xScaleRow += scaleBytesVar;
        cntB += oneVarB;                          // 运行期计数：0→A_
    }

    // ============ 6(下移). RecordExpectCount —— ★ 所有数据(WriteNb + LocalCopy)发完之后再声明预期字节 ============
    //   专业人士反馈：record_expect_count 放在所有 writenbwithnotify + localcopynb 之后（本次时序修复核心）。
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t ch = (r < epRankId_) ? r : r - 1;
        event_.SetMask(1U);
        CHK_RET(RecordExpectCount(channels_[ch], sendBytesPerPeer_[r], COMBINE_SETUP_CCU_TAG, event_, 1U));
        CHK_RET(WaitEvent(event_));
    }

    // ============ 9. WaitallpeersarriveNotify ============
    CcuRep::Variable arriveTargetVar = CreateVariable();
    arriveTargetVar = static_cast<uint64_t>(epWorldSize_ - 1U);
    CHK_RET(WaitallpeersarriveNotify(arriveTargetVar, COMBINE_SETUP_CCU_TAG));

    // ============ 10. ready[0..R)=1 ============
    CcuRep::Variable readyAddr = CreateVariable(); readyAddr = commCmdInfoAddrVar_;
    CcuRep::Variable readyBytesVar = CreateVariable();
    readyBytesVar = static_cast<uint64_t>(epWorldSize_) * sizeof(uint32_t);
    CcuRep::LocalAddr readyDst = CreateLocalAddr(); readyDst.addr = readyAddr; readyDst.token = commCmdInfoTokenVar_;
    CcuRep::LocalAddr readySrc = CreateLocalAddr(); readySrc.addr = oneBufAddrVar_; readySrc.token = oneBufTokenVar_;
    event_.SetMask(1U);
    CHK_RET(LocalCopyNb(readyDst, readySrc, readyBytesVar, &localCopyBuf_, 1U, event_));
    event_.SetMask(1U);
    CHK_RET(WaitEvent(event_));

    return HCCL_SUCCESS;
#else
    return HCCL_SUCCESS;  // a5 桩
#endif
}

}  // namespace mc2
