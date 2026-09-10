/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file alltoall_aicore_kernel_impl.h
 * \brief AIC 桥 kernel 的 device 侧组件（阶段1：纯通信，无 Blaze/matmul）
 *
 * 由 ccu_all_gather_matmul 的 matmul_fusion_kernel.h 裁剪而来：
 *   - 删除 MatmulTilingData / MatmulProcess（Blaze MXFP8 matmul，阶段3 再恢复）
 *   - 保留 AIC 侧两步骨架：
 *     步骤 0（block 0）: 逐 round 预写共享内存参数区（"参数桥"，AIC/CCU 间传参的 GM 区域；
 *                          每轮 11 槽，槽位语义见 .asc 的 BridgeSlot 枚举）
 *                        + 写 wake（eventVa[0]=1）
 *     步骤 1（all blocks）: 循环 commTurn 次轮询 eventVa[r+1]==1（round r done）
 *
 * ─── Event 布局 ───
 *   eventVa[0]   = wake（AIC block0 → CCU，唯一唤醒信号）
 *   eventVa[r+1] = round r done（CCU → AIC），r = 0..commTurn-1
 *
 * ─── AllToAll 语义（地址由 host 侧烧定，本 kernel 只搬运不计算）───
 *   srcOffset[r] = r * sub                       （本 rank sendBuf 第 myRank 块内的第 r 子段）
 *   dstOffset[r] = myRank * blockSize + r * sub  （远端 recvBuf 第 myRank 块内的第 r 子段，
 *                                                  myRank 项由 host 预乘——CCU 侧无标量×Variable 运算）
 *   sliceSize[r] = sub,  srcStride = blockSize
 */

#pragma once

// AIC kernel 侧 API（WriteGmByPassDCache / ReadGmByPassDCache / DataSyncBarrier / Nop 等）
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

// ============================================================================
// WriteBridgeSlot — 参数桥槽位写入 helper（槽位语义单一出处：.asc 的 BridgeSlot 枚举）
// 将 value 写入 round r 第 slot 个 Variable 在 GM 中的地址，绕过 DCache 直写 GM
// ============================================================================
__aicore__ inline void WriteBridgeSlot(const CcuMappedAddrs &addrs, uint32_t r,
                                       BridgeSlot slot, uint64_t value)
{
    __gm__ uint64_t *slotAddr = reinterpret_cast<__gm__ uint64_t *>(addrs.startVa[r][slot]);
    AscendC::WriteGmByPassDCache<uint64_t>(slotAddr, value);
}

// ============================================================================
// A2aAicoreProcess — AIC 侧桥接主逻辑（阶段1 纯通信核心函数）
//
// 整体执行流程（对齐 AG 示例 AicDirectPollProcess，去 matmul）：
//   步骤 0 (block 0 only): 逐 round 预写 11 槽参数桥到 GM Variable 区，然后写 wake event
//   步骤 1 (all blocks):   循环 commTurn 次，轮询 eventVa[r+1]==1 等 round r 通信完成
//
// ─── 双流时序示意 ───
//   streamCcu (CCU):    [wake]→ A2a round0 → A2a round1 → ...
//                                ↓ev[1]      ↓ev[2]
//   streamAicore (AIC):  [writeWake]→ poll done0 → poll done1 → ...
//
// 阶段1 中 AIC 轮询 done 仅用于验证同步链路（阶段3 在轮询后接 matmul，即成通算流水）。
// ============================================================================
__aicore__ inline void A2aAicoreProcess(
    __gm__ void *sendBuf, __gm__ void *recvBuf,
    CcuMappedAddrs mappedAddrs,
    uint64_t token, uint64_t srcStride,
    A2aGoSize goSize,
    uint32_t rankSize, uint32_t commTurn,
    __gm__ const uint64_t *srcOffsets,
    __gm__ const uint64_t *dstOffsets,
    __gm__ const uint64_t *sliceSizes)
{
    (void)rankSize;  // 阶段1 AIC 侧不使用 rankSize（阶段3 matmul 切分时恢复使用）

    // ========================================================================
    // 步骤 0：block 0 一次性预写 commTurn 组参数桥 + 写 wake event
    //
    // 为什么由 AIC block 0 来写参数桥？
    //   CCU kernel 的 Variable 区是 GM 上的共享内存，AIC 和 CCU 都能访问。
    //   block 0 负责"装弹"——把 host 传入的地址/偏移/大小等参数写入 Variable 区，
    //   CCU kernel 运行时从这些槽位读取参数。只需 block 0 做一次。
    //
    // 为什么写完所有 round 的参数桥后才写 wake？
    //   CCU kernel 串行循环 commTurn 次，只需唤醒一次（eventVa[0]=1）。
    //   唤醒后 CCU 各 round 自动串行进入，因此必须先备好所有 round 的参数。
    // ========================================================================
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0) {
        for (uint32_t r = 0; r < commTurn; r++) {
            // 该 round 的 3 个 per-round 标量（srcOffset/dstOffset/sliceSize，从 device GM
            // 数组读取——host 预拷贝；其余槽位为全轮同值的标量参数，直接经 kernel 参数传入）
            const uint64_t srcOff = srcOffsets[r];
            const uint64_t dstOff = dstOffsets[r];
            const uint64_t sliceSize = sliceSizes[r];

            // 按 BridgeSlot 枚举写入 11 槽（槽位语义见枚举定义处，此处不重复）
            WriteBridgeSlot(mappedAddrs, r, SLOT_INPUT, reinterpret_cast<uint64_t>(sendBuf));
            WriteBridgeSlot(mappedAddrs, r, SLOT_OUTPUT, reinterpret_cast<uint64_t>(recvBuf));
            WriteBridgeSlot(mappedAddrs, r, SLOT_TOKEN, token);
            WriteBridgeSlot(mappedAddrs, r, SLOT_SRC_OFFSET, srcOff);
            WriteBridgeSlot(mappedAddrs, r, SLOT_DST_OFFSET, dstOff);
            WriteBridgeSlot(mappedAddrs, r, SLOT_SLICE_SIZE, sliceSize);
            WriteBridgeSlot(mappedAddrs, r, SLOT_SRC_STRIDE, srcStride);
            WriteBridgeSlot(mappedAddrs, r, SLOT_GO_ADDR_OFFSET, goSize.addrOffset);
            WriteBridgeSlot(mappedAddrs, r, SLOT_GO_LOOP_ITER, goSize.loopIterNum);
            WriteBridgeSlot(mappedAddrs, r, SLOT_GO_LOOP_EXTEND, goSize.loopExtendNum);
            WriteBridgeSlot(mappedAddrs, r, SLOT_GO_TAIL_SIZE, goSize.tailSize);
            // DataSyncBarrier: 确保上述写操作对 CCU 核可见后才进入下一 round 的写
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        }

        // ---- 写 wake event（eventVa[0]=1）----
        // CCU kernel round 0 通过 EventWait(events[0]) 阻塞等待此信号。
        __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
        AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
        // 确保 wake 写操作对所有 CCU 核可见
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    }

    // ========================================================================
    // 步骤 1：所有核循环 commTurn 次 — 轮询 round r 通信完成（eventVa[r+1]==1）
    //
    // CCU kernel 完成 round r 后执行 EventRecord(events[r+1])，将 eventVa[r+1] 置 1。
    // 轮询三件套（对齐 AG 已验证写法）：
    //   ReadGmByPassDCache — 绕过 DCache 直读 GM 最新值
    //   DataSyncBarrier    — 确保读操作完成
    //   Nop<800>           — 插入 800 个 nop 降低轮询频率，减少 GM 读带宽占用
    // ========================================================================
    for (uint32_t r = 0; r < commTurn; r++) {
        __gm__ uint64_t *doneAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[r + 1]);
        while (1) {
            uint64_t doneVal = AscendC::ReadGmByPassDCache(doneAddr);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
            AscendC::Nop<800>();
            if (doneVal == 1) {
                break;
            }
        }
        // 阶段1：轮询到 done 即结束本轮（阶段3 在此接 matmul tile r，与 CCU round r+1 通信 overlap）
    }
}
