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
 * \file ccu_all_to_all_aic.h
 * \brief CCU AllToAll——AIC 侧同步原语（device 代码，__aicore__）
 *
 * 消费方接入（AIC kernel）：融合 kernel block 0 调 AllToAllProcess 发起
 * （帧首事件清扫 → prepare → commit）；计算核逐轮 WaitCcuCommDone
 * （或经 CommPolicy 逐 tile 等待，见融合层 CcuCommWaitPolicy）
 */

#pragma once

#include "../ccu_comm.h"
#include "ccu_all_to_all_tiling.h"

// ---- AIC 侧同步原语 ----

// 按轮号 + 槽位枚举写通信参数区（XN 寄存器 MMIO 直写，绕 DCache）
__aicore__ inline void WriteCcuMappedAddr(const CcuMappedAddrs &addrs, uint32_t r,
                                           CommSlot slot, uint64_t value)
{
    __gm__ uint64_t *addr = reinterpret_cast<__gm__ uint64_t *>(addrs.startVa[r][slot]);
    AscendC::WriteGmByPassDCache<uint64_t>(addr, value);
}

// AIC 侧同步第 1 步【prepare】：预写全部轮的通信参数（block 0 调用；
// 组 0 = scale 轮全量 A2A，组 1..dataCommTurn = data 轮 t = 组 t+1）
__aicore__ inline void PrepareAllToAllComm(const CcuCommTilingData &comm,
                                           __gm__ void *x1, __gm__ void *x1Scale,
                                           __gm__ void *recvBuf, __gm__ void *recvScaleBuf)
{
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_INPUT, reinterpret_cast<uint64_t>(x1Scale));
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_OUTPUT, reinterpret_cast<uint64_t>(recvScaleBuf));
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_TOKEN, comm.scaleToken);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_SRC_OFFSET, 0);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_DST_OFFSET, comm.scaleDstOffset);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_SLICE_SIZE, comm.scaleSliceSize);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_SRC_STRIDE, comm.scaleSrcStride);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_GO_ADDR_OFFSET, comm.scaleGoSize.addrOffset);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_GO_LOOP_ITER, comm.scaleGoSize.loopIterNum);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_GO_LOOP_EXTEND, comm.scaleGoSize.loopExtendNum);
    WriteCcuMappedAddr(comm.mappedAddrs, 0, SLOT_GO_TAIL_SIZE, comm.scaleGoSize.tailSize);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    for (uint32_t t = 0; t < comm.dataCommTurn; t++) {
        const uint32_t r = t + 1;
        const uint64_t srcOff = comm.srcOffsets[t];
        const uint64_t dstOff = comm.dstOffsets[t];
        const uint64_t sliceSize = comm.sliceSizes[t];

        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_INPUT, reinterpret_cast<uint64_t>(x1));
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_OUTPUT, reinterpret_cast<uint64_t>(recvBuf));
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_TOKEN, comm.token);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_SRC_OFFSET, srcOff);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_DST_OFFSET, dstOff);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_SLICE_SIZE, sliceSize);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_SRC_STRIDE, comm.srcStride);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_GO_ADDR_OFFSET, comm.goSize.addrOffset);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_GO_LOOP_ITER, comm.goSize.loopIterNum);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_GO_LOOP_EXTEND, comm.goSize.loopExtendNum);
        WriteCcuMappedAddr(comm.mappedAddrs, r, SLOT_GO_TAIL_SIZE, comm.goSize.tailSize);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    }
}

// AIC 侧同步第 2 步【commit】：写 wake event，通知 CCU 启动通信（block 0 调用）
__aicore__ inline void CommitCcuComm(const CcuMappedAddrs &mappedAddrs)
{
    __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
    AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}

// AIC 侧 AllToAll 通信处理（与融合侧 MatmulProcess 对应）：帧首 CKE 事件残值清扫 →
// prepare → commit（顺序约束：CCU 只唤醒一次，清零必须在 commit 之前，否则吞 round-done）。
// block 0 调用（守卫内置），每帧一次
__aicore__ inline void AllToAllProcess(const CcuCommTilingData &comm,
                                       __gm__ void *x1, __gm__ void *x1Scale,
                                       __gm__ void *recvBuf, __gm__ void *recvScaleBuf)
{
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) != 0) {
        return;
    }

    const uint32_t totalRounds = comm.dataCommTurn + 1;
    for (uint32_t i = 0; i <= totalRounds; i++) {
        __gm__ uint64_t *evAddr =
            reinterpret_cast<__gm__ uint64_t *>(comm.mappedAddrs.eventVa[i]);
        AscendC::WriteGmByPassDCache<uint64_t>(evAddr, 0);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    PrepareAllToAllComm(comm, x1, x1Scale, recvBuf, recvScaleBuf);
    CommitCcuComm(comm.mappedAddrs);
}

// AIC 侧同步第 3 步【wait】：等待 CCU 某轮通信完成（计算核逐轮调用）：
// 绕 DCache 直读 CKE 寄存器槽位（MMIO uncached 语义）
__aicore__ inline void WaitCcuCommDone(uint64_t eventVa)
{
    __gm__ uint64_t *doneAddr = reinterpret_cast<__gm__ uint64_t *>(eventVa);
    while (AscendC::ReadGmByPassDCache(doneAddr) != 1) {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        AscendC::Nop<800>();
    }
}
