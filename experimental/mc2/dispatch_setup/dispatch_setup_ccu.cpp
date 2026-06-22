/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_setup_ccu.cpp —— 最小 CCU dispatch_setup 的设备侧算法。
 *  逐段对照 a6 参考 moe_distribute_dispatch_ccu_setup.cpp 的“段 1 / 段 2 / 段 10”，
 *  只保留本卡写 1 的最小路径。
 */

#include "dispatch_setup_ccu.h"

namespace mc2 {

HcclResult CcuKernelDispatchSetup::Algorithm()
{
    // ============ 段 1. InitResource：创建本算法用到的 CCU 资源 ============
    //   每个 CreateXxx() 在 CCU 上绑定一个 XN 槽 / 资源句柄，必须先建后用。
    statusAddrVar_  = CreateVariable();
    statusTokenVar_ = CreateVariable();
    oneBufAddrVar_  = CreateVariable();
    oneBufTokenVar_ = CreateVariable();
    readyDst_       = CreateLocalAddr();
    readySrc_       = CreateLocalAddr();
    event_          = CreateCompletedEvent();
    // [T1] A6：LocalCopyNb(dst,src,len,bufs,bufCount,event) 的 UB 暂存 buf 必须块分配，
    //   不能用单个 CreateCcuBuf()。单槽串行复用：每条 LocalCopyNb 后紧跟 WaitEvent 落地。
    CHK_RET(CreateBlockCcuBuf(1U, &localCopyBuf_));

    // ============ 段 2. Load SQE 参数（顺序须与 GeneArgs 完全一致）============
    //   GeneArgs 返回 {statusAddr, statusToken, oneBufAddr, oneBufToken}，此处按同序 Load，
    //   否则 device 读到错位的地址/ token（host-device 二进制契约，错位即静默读垃圾）。
    Load(statusAddrVar_);
    Load(statusTokenVar_);
    Load(oneBufAddrVar_);
    Load(oneBufTokenVar_);

    // ============ 段 10. 本卡状态区置 1（把 oneBuf 整块 LocalCopyNb 到状态区）============
    //   dst = 状态区起址（偏移 0），token = 状态区 token；
    //   src = oneBuf（host 已填 R 个 int32=1），token = oneBuf token；
    //   len = R * sizeof(int32) 字节（单 die R=1 → 4 字节）。
    readyDst_.addr  = statusAddrVar_;     // 偏移 0，直接取状态区起址（Address = Variable）
    readyDst_.token = statusTokenVar_;
    readySrc_.addr  = oneBufAddrVar_;
    readySrc_.token = oneBufTokenVar_;

    CcuRep::Variable lenBytesVar = CreateVariable();
    // Variable = 立即数（合法）；注意 hcomm 不支持 Variable + 立即数，故长度先落进 Variable 再用。
    lenBytesVar = static_cast<uint64_t>(epWorldSize_) * DISPATCH_STATUS_DTYPE_BYTES;

#ifdef DISPATCH_CCU_A6
    // ===== a6-only 真实写：a5 hcomm 头无 6 参 LocalCopyNb / 1 参 WaitEvent / CompletedEvent::SetMask，====
    //   故 a5 编译时整体屏蔽；a6 上定义 -DDISPATCH_CCU_A6 即启用。[T1][T2]
    // DMA 前 SetMask 指定本次占用的 event 位（bit0）。
    event_.SetMask(1U);
    CHK_RET(LocalCopyNb(readyDst_, readySrc_, lenBytesVar, &localCopyBuf_, 1U, event_));
    // 等状态区落地（bit0），保证 teardown AIV 一定能读到稳定的 1。
    event_.SetMask(1U);
    CHK_RET(WaitEvent(event_));
#else
    // a5 占位：段 1/2 + 上面的 readyDst_/readySrc_/lenBytesVar 已建好（结构可编译）；
    //   真实 DMA 写状态区是 a6-only，见上 #ifdef 块。a6 移植：-DDISPATCH_CCU_A6。
    (void)lenBytesVar;
#endif

    return HCCL_SUCCESS;
}

}  // namespace mc2
