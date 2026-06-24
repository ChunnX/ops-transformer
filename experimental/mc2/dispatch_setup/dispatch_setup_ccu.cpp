/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_setup_ccu.cpp —— CCU dispatch_setup 设备侧算法（a6 0.5RTT 跨卡写状态区）。
 *  逐段对照 a6 已验证参考 moe_distribute_dispatch_ccu_setup.cpp，payload 塌缩成"一次定长写"：
 *  本卡把 oneBuf(slot0=1) 经 WriteNbWithArriveNotify 写进【对端】status；本卡 status 由对端写。
 */

#include "dispatch_setup_ccu.h"
#include "ccu_assist_pub.h"      // 与参考同源（PreSync/数据面 API 的上层依赖）
#include "hcomm_primitives.h"

namespace mc2 {

namespace {
// PreSync 信息槽（XN_ID）：每块独立信息一个槽，mask = 1<<XN_ID（对齐参考 mesh1d / hccl）。
constexpr uint32_t STATUS_ADDR_XN  = 0;     // 对端 status 起址
constexpr uint32_t STATUS_TOKEN_XN = 1;     // 对端 status token
constexpr uint32_t PRE_SLOT_BITS   = (1U << STATUS_ADDR_XN) | (1U << STATUS_TOKEN_XN);  // = 0x3
constexpr uint32_t CKE_PRE = 0;             // 同步轮次：前同步

// 跨卡写字节数 = 整块状态区（与驱动 STATUS_BYTES=32 / teardown 32B 读块一致；oneBuf=[1,0..] → 对端 status[0]=1）。
//   用整块写而非仅 4B：① 与 oneBuf/status 分配大小对齐、零越界；② 规避 CCU DMA 最小粒度未知；
//   ③ RecordExpectCount 与 WriteNbWithArriveNotify 用同一字节数，arrive-notify 对账才一致。
//   ★ 契约（审核 #1）：status 整 32B 块为 dispatch_setup【独占】的就绪标志区、无其它写者，故整块写、
//     连带把对端 slot1..7 写 0 是【预期语义】非越权。日后若把 status 扩成多字段，需同步收窄写范围。
constexpr uint64_t STATUS_BLOCK_BYTES = 32;
}  // namespace

HcclResult CcuKernelDispatchSetup::Algorithm()
{
#ifdef DISPATCH_CCU_A6
    // ============ 段 1. InitResource ============
    statusAddrVar_  = CreateVariable();
    statusTokenVar_ = CreateVariable();
    oneBufAddrVar_  = CreateVariable();
    oneBufTokenVar_ = CreateVariable();
    localSrc_  = CreateLocalAddr();
    remoteDst_ = CreateRemoteAddr();
    event_     = CreateCompletedEvent();

    remoteStatusAddrs_.reserve(epWorldSize_);
    remoteStatusTokens_.reserve(epWorldSize_);
    // 对端 status 地址/token 用 channel 的 XN 寄存器接收（PreSync 填）；本卡 self 槽用普通 Var 占位。
    //   （审核 #4）self 槽这两个占位 Var 段2 会被 statusAddrVar_/statusTokenVar_ 覆盖；保留仅为让
    //   remoteStatusAddrs_[r] 对全 r 索引对齐（self 不参与跨卡写）。微量资源、与参考同款。
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) {
            remoteStatusAddrs_.push_back(CreateVariable());
            remoteStatusTokens_.push_back(CreateVariable());
        } else {
            uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
            CcuRep::Variable vAddr, vTok;
            CHK_RET(CreateVariable(channels_[channelIdx], STATUS_ADDR_XN,  &vAddr));
            CHK_RET(CreateVariable(channels_[channelIdx], STATUS_TOKEN_XN, &vTok));
            remoteStatusAddrs_.push_back(vAddr);
            remoteStatusTokens_.push_back(vTok);
        }
    }

    // ============ 段 2. LoadArgs（顺序严格 = GeneArgs：status_addr, status_token, oneBuf_addr, oneBuf_token）============
    Load(statusAddrVar_);
    Load(statusTokenVar_);
    Load(oneBufAddrVar_);
    Load(oneBufTokenVar_);
    // self 槽用本卡 status 地址/token 占位（self 不发，仅占位、不参与跨卡写）。
    remoteStatusAddrs_[epRankId_]  = statusAddrVar_;
    remoteStatusTokens_[epRankId_] = statusTokenVar_;

    // 写长度落进 Variable（hcomm 不支持 Variable+立即数，先把立即数落 Variable 再用）。
    CcuRep::Variable writeBytesVar = CreateVariable();
    writeBytesVar = static_cast<uint64_t>(STATUS_BLOCK_BYTES);

    // ============ 段 2.5. Clear 上次 0.5RTT 状态（数据面 API ①）============
    CHK_RET(clearallpeersarriveNotify(DISPATCH_SETUP_CCU_TAG));

    // ============ 段 3. PreSync（NotifyRecord/Wait 交换对端 status 地址+token）============
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_PRE, STATUS_ADDR_XN,
                             statusAddrVar_,  1U << STATUS_ADDR_XN));
        CHK_RET(NotifyRecord(channels_[channelIdx], CKE_PRE, STATUS_TOKEN_XN,
                             statusTokenVar_, 1U << STATUS_TOKEN_XN));
    }
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        CHK_RET(NotifyWait(channels_[channelIdx], CKE_PRE, PRE_SLOT_BITS));
    }

    // ============ 段 6. RecordExpectCount（数据面 API ②：告诉对端我会发多少字节）============
    //   ★ 串行 record+wait（每对端 record 后【立即】WaitEvent）是关键：event_/mask 单槽复用在此模式下
    //     对任意 R 都安全（同参考 215-221 行）。勿改成"批量 record 再统一 wait"的流水化——单 event_ + 固定
    //     mask=1U 会让后一对端的完成事件覆盖前一个（审核 #3）。当前仅 R=2 上板验证。
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        event_.SetMask(1U);
        CHK_RET(RecordExpectCount(channels_[channelIdx], writeBytesVar, DISPATCH_SETUP_CCU_TAG, event_, 1U));
        CHK_RET(WaitEvent(event_));
    }

    // ============ 段 7. 跨卡写：把 oneBuf(slot0=1) 写进对端 status（数据面 API ③）============
    //   极简 R=2：本卡只发对端、不发自己（本卡 status 由对端写）；对端 status 偏移 0 = slot0。
    localSrc_.addr  = oneBufAddrVar_;
    localSrc_.token = oneBufTokenVar_;
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        if (r == epRankId_) continue;
        uint32_t channelIdx = (r < epRankId_) ? r : r - 1;
        remoteDst_.addr  = remoteStatusAddrs_[r];     // 对端 status 起址（偏移 0）
        remoteDst_.token = remoteStatusTokens_[r];
        CHK_RET(WriteNbWithArriveNotify(channels_[channelIdx], remoteDst_,
                                        localSrc_, writeBytesVar, DISPATCH_SETUP_CCU_TAG));
    }

    // ============ 段 9. WaitallpeersarriveNotify（数据面 API ④：等本卡 status 被对端写好）============
    //   段 7 全是跨卡 WriteNbWithArriveNotify、无自发自；此处等到达计数达 R-1（= 1 个对端写完本卡）。
    //   返回时本卡 status 已被对端写好 → 后续 teardown(AIV) 轮询 slot0 立即见 1。
    CcuRep::Variable arriveTargetVar = CreateVariable();
    arriveTargetVar = static_cast<uint64_t>(epWorldSize_ - 1U);
    CHK_RET(WaitallpeersarriveNotify(arriveTargetVar, DISPATCH_SETUP_CCU_TAG));

    return HCCL_SUCCESS;
#else
    // a5 桩：0.5RTT 数据面（clear/RecordExpectCount/WriteNbWithArriveNotify/Waitall）+ channel-XN PreSync
    //   的实现为 a6-only（SE 链接期补全）。a5 只编 shell、不引用这些符号 → 链接通过。a6: -DDISPATCH_CCU_A6。
    return HCCL_SUCCESS;
#endif
}

}  // namespace mc2
