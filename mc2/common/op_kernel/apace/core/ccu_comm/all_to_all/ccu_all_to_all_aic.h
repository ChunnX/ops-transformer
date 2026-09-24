/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ccu_all_to_all_aic.h
 * \brief CCU AllToAll——AIC 侧同步原语（device 代码，__aicore__）
 *
 * 消费方接入（AIC kernel）：融合 kernel block 0 调 AllToAllProcess 发起
 * （帧首事件清扫 → commit 唤醒）；计算核逐轮 WaitCcuCommDone
 * （或经 CommPolicy 逐 tile 等待，见融合层 CcuCommWaitPolicy）。
 * 通信参数不经 AIC：host 经 GM staging 直达 CCU（见 ccu_comm.h / _tiling.h）。
 */

#pragma once

#include "../ccu_comm.h"
#include "ccu_all_to_all_tiling.h"

// ---- AIC 侧同步原语 ----

// AIC 侧同步第 1 步【commit】：写 wake event，通知 CCU 启动通信（block 0 调用）
__aicore__ inline void CommitCcuComm(const CcuMappedAddrs &mappedAddrs)
{
    __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
    AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}

// AIC 侧 AllToAll 通信处理（与融合侧 MatmulProcess 对应）：帧首 CKE 事件残值清扫 →
// commit（顺序约束：CCU 只唤醒一次，清零必须在 commit 之前，否则吞 round-done）。
// block 0 调用（守卫内置），每帧一次
__aicore__ inline void AllToAllProcess(const CcuCommTilingData &comm)
{
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) != 0) {
        return;
    }

    // events: [0]=wake, [t+1]=data 轮 t done（轮 0 含其搭载的 scale 整块）
    const uint32_t totalEventNum = comm.dataCommTurn + 1;
    for (uint32_t i = 0; i < totalEventNum; i++) {
        __gm__ uint64_t *evAddr =
            reinterpret_cast<__gm__ uint64_t *>(comm.mappedAddrs.eventVa[i]);
        AscendC::WriteGmByPassDCache<uint64_t>(evAddr, 0);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

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
