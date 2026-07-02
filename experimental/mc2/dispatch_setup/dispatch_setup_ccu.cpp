/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_setup_ccu.cpp —— CCU dispatch_setup 设备侧算法（真实逐 token 跨卡搬运，A6 视角，mxfp8）。
 *  逐段对照 A6-proven 参考 moe_distribute_dispatch_ccu_setup.cpp，按 SETUP_DESIGN.md 适配：
 *    · 落点 expert-major：全由 host 预算的 routeTable {dstRank, dataDstByteOff, scaleDstByteOff} 决定，
 *      device 只读 + 加，不算 ge/E、不乘法（照参考"host 预算 dstByteOff、device 不乘"范式）。
 *    · payload 与 scale 两个独立区：每 token 两次写（self=LocalCopyNb / cross=WriteNbWithArriveNotify）。
 *    · 无 metadata、无标量 GM 写：wishCnt/expand_idx 由 driver host 预算（CCU 无 StoreVariableAt）。
 *
 *  地址范式（hcomm 不支持 Variable+立即数）：立即数先落 Variable；Address=Variable / Address+=Variable /
 *    Variable=Variable+Variable 合法。8 字节铁律：routeTable 按 int64 排布、步进 8。
 */

#include "dispatch_setup_ccu.h"
#include "ccu_assist_pub.h"
#include "hcomm_primitives.h"

namespace mc2 {

namespace {
// PreSync 信息槽（XN_ID）：交换本卡【接收】数据区/scale 区的 起址+token（对端据此写本卡）。
constexpr uint32_t DATA_ADDR_XN   = 0;
constexpr uint32_t DATA_TOKEN_XN  = 1;
constexpr uint32_t SCALE_ADDR_XN  = 2;
constexpr uint32_t SCALE_TOKEN_XN = 3;
constexpr uint32_t PRE_SLOT_BITS  = (1U << DATA_ADDR_XN) | (1U << DATA_TOKEN_XN) |
                                    (1U << SCALE_ADDR_XN) | (1U << SCALE_TOKEN_XN);   // = 0xF
constexpr uint32_t CKE_PRE = 0;
constexpr uint64_t ROUTE_ROW_BYTES = 3U * sizeof(uint64_t);   // routeTable 每行 3×int64 = 24
}  // namespace

HcclResult CcuKernelDispatchSetup::Algorithm()
{
#ifdef DISPATCH_CCU_A6
    // ============ 1. InitResource ============
    xDataAddrVar_        = CreateVariable();
    xScaleAddrVar_       = CreateVariable();
    routeTableAddrVar_   = CreateVariable();
    dataRegionAddrVar_   = CreateVariable();
    scaleRegionAddrVar_  = CreateVariable();
    commCmdInfoAddrVar_  = CreateVariable();
    oneBufAddrVar_       = CreateVariable();
    xDataTokenVar_       = CreateVariable();
    xScaleTokenVar_      = CreateVariable();
    routeTableTokenVar_  = CreateVariable();
    dataRegionTokenVar_  = CreateVariable();
    scaleRegionTokenVar_ = CreateVariable();
    commCmdInfoTokenVar_ = CreateVariable();
    oneBufTokenVar_      = CreateVariable();
    localSrcData_  = CreateLocalAddr();
    localSrcScale_ = CreateLocalAddr();
    localDstData_  = CreateLocalAddr();
    localDstScale_ = CreateLocalAddr();
    remoteDstData_  = CreateRemoteAddr();
    remoteDstScale_ = CreateRemoteAddr();
    event_ = CreateCompletedEvent();
    CHK_RET(CreateBlockCcuBuf(1U, &localCopyBuf_));   // A6 LocalCopyNb 的 UB 暂存 buf（块分配）

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
    Load(xDataAddrVar_);       Load(xScaleAddrVar_);      Load(routeTableAddrVar_);
    Load(dataRegionAddrVar_);  Load(scaleRegionAddrVar_); Load(commCmdInfoAddrVar_); Load(oneBufAddrVar_);
    Load(xDataTokenVar_);      Load(xScaleTokenVar_);     Load(routeTableTokenVar_);
    Load(dataRegionTokenVar_); Load(scaleRegionTokenVar_);Load(commCmdInfoTokenVar_);Load(oneBufTokenVar_);
    // 本卡 self 槽用本卡接收区地址/token 占位（自发自走本地）
    remoteDataAddrs_[epRankId_]  = dataRegionAddrVar_;   remoteDataTokens_[epRankId_]  = dataRegionTokenVar_;
    remoteScaleAddrs_[epRankId_] = scaleRegionAddrVar_;  remoteScaleTokens_[epRankId_] = scaleRegionTokenVar_;

    // 立即数落 Variable（8 字节步进 / payload、scale 字节长度 / routeTable 行跨度）
    CcuRep::Variable elem8Var    = CreateVariable(); elem8Var    = static_cast<uint64_t>(sizeof(uint64_t));   // 8
    CcuRep::Variable rowStrideVar= CreateVariable(); rowStrideVar= static_cast<uint64_t>(ROUTE_ROW_BYTES);    // 24
    CcuRep::Variable hBytesVar   = CreateVariable(); hBytesVar   = hBytes_;                                   // h
    CcuRep::Variable scaleBytesVar= CreateVariable();scaleBytesVar= scaleBytes_;                              // h/32
    CcuRep::Variable tokBothVar  = CreateVariable(); tokBothVar  = hBytesVar + scaleBytesVar;                 // h + h/32

    // ============ 2.5. Clear 上轮 0.5RTT 状态 ============
    CHK_RET(clearallpeersarriveNotify(DISPATCH_SETUP_CCU_TAG));

    // ============ 3. PreSync：广播本卡 data/scale 接收区 addr+token 给对端（4 XN 槽）============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t ch = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, DATA_ADDR_XN,   dataRegionAddrVar_,   1U << DATA_ADDR_XN));
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, DATA_TOKEN_XN,  dataRegionTokenVar_,  1U << DATA_TOKEN_XN));
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, SCALE_ADDR_XN,  scaleRegionAddrVar_,  1U << SCALE_ADDR_XN));
        CHK_RET(NotifyRecord(channels_[ch], CKE_PRE, SCALE_TOKEN_XN, scaleRegionTokenVar_, 1U << SCALE_TOKEN_XN));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t ch = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[ch], CKE_PRE, PRE_SLOT_BITS));
    }

    // ============ 5. PreCount：读 routeTable dstRank 累加 sendBytesPerPeer（每 token = payload+scale 字节）============
    CcuRep::Variable rankVarA   = CreateVariable();
    CcuRep::Variable routeAddrA = CreateVariable(); routeAddrA = routeTableAddrVar_;
    for (uint32_t i = 0; i < bs_; ++i) {
        for (uint32_t j = 0; j < topK_; ++j) {
            LoadVariable(routeAddrA, rankVarA);       // 读 dstRank（行首字段）
            routeAddrA += rowStrideVar;               // 跳到下一行（3×int64）
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarA == r) {
                    sendBytesPerPeer_[r] = sendBytesPerPeer_[r] + tokBothVar;
                }
            }
        }
    }

    // ============ 6. RecordExpectCount（逐对端 SetMask(1)+WaitEvent 串行）============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t ch = (r < epRankId_) ? r : r - 1;
        event_.SetMask(1U);
        CHK_RET(RecordExpectCount(channels_[ch], sendBytesPerPeer_[r], DISPATCH_SETUP_CCU_TAG, event_, 1U));
        CHK_RET(WaitEvent(event_));
    }

    // ============ 7. DoMainLoop：逐 token 写 payload+scale 到对端 expert-major 数据区 ============
    CcuRep::Variable rankVarB   = CreateVariable();
    CcuRep::Variable dataOffVar = CreateVariable();
    CcuRep::Variable scaleOffVar= CreateVariable();
    CcuRep::Variable routeAddrB = CreateVariable(); routeAddrB = routeTableAddrVar_;
    CcuRep::Variable xDataRow   = CreateVariable(); xDataRow   = xDataAddrVar_;   // per-i payload 源行
    CcuRep::Variable xScaleRow  = CreateVariable(); xScaleRow  = xScaleAddrVar_;  // per-i scale 源行
    localSrcData_.token  = xDataTokenVar_;
    localSrcScale_.token = xScaleTokenVar_;
    for (uint32_t i = 0; i < bs_; ++i) {
        localSrcData_.addr  = xDataRow;               // token i 的 payload 源（topk 个 j 共用）
        localSrcScale_.addr = xScaleRow;
        for (uint32_t j = 0; j < topK_; ++j) {
            LoadVariable(routeAddrB, rankVarB);   routeAddrB += elem8Var;   // dstRank
            LoadVariable(routeAddrB, dataOffVar); routeAddrB += elem8Var;   // dataDstByteOff
            LoadVariable(routeAddrB, scaleOffVar);routeAddrB += elem8Var;   // scaleDstByteOff
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                CCU_IF(rankVarB == r) {
                    if (r == epRankId_) {
                        // ---- 自发自：本卡接收区（LocalCopyNb，串行落地）----
                        localDstData_.addr = dataRegionAddrVar_; localDstData_.addr += dataOffVar;
                        localDstData_.token = dataRegionTokenVar_;
                        event_.SetMask(1U);
                        CHK_RET(LocalCopyNb(localDstData_, localSrcData_, hBytesVar, &localCopyBuf_, 1U, event_));
                        CHK_RET(WaitEvent(event_));
                        localDstScale_.addr = scaleRegionAddrVar_; localDstScale_.addr += scaleOffVar;
                        localDstScale_.token = scaleRegionTokenVar_;
                        event_.SetMask(1U);
                        CHK_RET(LocalCopyNb(localDstScale_, localSrcScale_, scaleBytesVar, &localCopyBuf_, 1U, event_));
                        CHK_RET(WaitEvent(event_));
                    } else {
                        uint32_t ch = (r < epRankId_) ? r : r - 1;
                        // ---- 跨卡：对端接收区（WriteNbWithArriveNotify）----
                        remoteDstData_.addr = remoteDataAddrs_[r]; remoteDstData_.addr += dataOffVar;
                        remoteDstData_.token = remoteDataTokens_[r];
                        CHK_RET(WriteNbWithArriveNotify(channels_[ch], remoteDstData_, localSrcData_,
                                                        hBytesVar, DISPATCH_SETUP_CCU_TAG));
                        remoteDstScale_.addr = remoteScaleAddrs_[r]; remoteDstScale_.addr += scaleOffVar;
                        remoteDstScale_.token = remoteScaleTokens_[r];
                        CHK_RET(WriteNbWithArriveNotify(channels_[ch], remoteDstScale_, localSrcScale_,
                                                        scaleBytesVar, DISPATCH_SETUP_CCU_TAG));
                    }
                }
            }
        }
        xDataRow  += hBytesVar;      // 下一 token 源行
        xScaleRow += scaleBytesVar;
    }

    // ============ 9. WaitallpeersarriveNotify（等本卡收齐：totalCntXn 达 R-1）============
    CcuRep::Variable arriveTargetVar = CreateVariable();
    arriveTargetVar = static_cast<uint64_t>(epWorldSize_ - 1U);
    CHK_RET(WaitallpeersarriveNotify(arriveTargetVar, DISPATCH_SETUP_CCU_TAG));

    // ============ 10. ready[0..R)=1（本卡收齐后本地置位，teardown AIV 轮询）============
    CcuRep::Variable readyAddr = CreateVariable(); readyAddr = commCmdInfoAddrVar_;   // ready 区从偏移 0 起
    CcuRep::Variable readyBytesVar = CreateVariable();
    readyBytesVar = static_cast<uint64_t>(epWorldSize_) * sizeof(uint32_t);           // R*4
    CcuRep::LocalAddr readyDst = CreateLocalAddr(); readyDst.addr = readyAddr; readyDst.token = commCmdInfoTokenVar_;
    CcuRep::LocalAddr readySrc = CreateLocalAddr(); readySrc.addr = oneBufAddrVar_; readySrc.token = oneBufTokenVar_;
    event_.SetMask(1U);
    CHK_RET(LocalCopyNb(readyDst, readySrc, readyBytesVar, &localCopyBuf_, 1U, event_));
    event_.SetMask(1U);
    CHK_RET(WaitEvent(event_));

    return HCCL_SUCCESS;
#else
    // a5 桩：0.5RTT 数据面（clear/RecordExpectCount/WriteNbWithArriveNotify/Waitall）+ channel-XN PreSync
    //   为 a6-only（SE 链接期补全）。a5 只编 shell、不引用这些符号 → 链接通过。a6: -DDISPATCH_CCU_A6。
    return HCCL_SUCCESS;
#endif
}

}  // namespace mc2
