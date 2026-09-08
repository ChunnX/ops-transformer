/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INPUT_PREPARE_SYNC_H
#define MEGA_MOE_INPUT_PREPARE_SYNC_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// 950 mode 4 为配对 AIC/AIV 的独立计数器；AIV1 对应 AIC 的 flag + 16。
// 避开 GMM/prologue 的 mode 4 0/1、4--7，以及 SyncAll 保留的 11--14。
constexpr uint16_t MTE_LOCAL_INPUT_READY_FLAG = 8;
constexpr uint16_t MTE_INPUT_READY_RELEASE_FLAG = 9;
constexpr uint16_t MTE_INPUT_UB_RELEASE_FLAG = 10;
constexpr uint16_t MTE_AIV1_FLAG_OFFSET = 16;
// mode 0 汇合全部 AIC；与 MoE 量化通知使用的 mode 0 flag 8 分离。
constexpr uint16_t MTE_ALL_INPUT_READY_FLAG = 9;

// 所有 AIV（含空 token）在状态清零及共享输入写回下发后调用，不等待其他 AIV。
__aicore__ inline void NotifyMteInputPrepared()
{
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(MTE_LOCAL_INPUT_READY_FLAG);
    }
}

// 拆开 SDK SyncAll<false> 的三段硬件握手：AIV 早通知，AIC 汇合，AIV 在本核路由完成后接收。
// 这里只保证输入和状态就绪，不保证 AIV 的路由/count 搬运完成或 UB 可被 AIC 覆盖。
__aicore__ inline void WaitForMteInputPrepared()
{
    if constexpr (g_coreType == AscendC::AIC) {
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(MTE_LOCAL_INPUT_READY_FLAG);
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(MTE_LOCAL_INPUT_READY_FLAG + MTE_AIV1_FLAG_OFFSET);
        // 与 SDK 一致用 FIX 发布 AIC 汇合；必须位于后面的 UB 释放等待之前。
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(MTE_ALL_INPUT_READY_FLAG);
        AscendC::CrossCoreWaitFlag<0, PIPE_S>(MTE_ALL_INPUT_READY_FLAG);
        AscendC::CrossCoreSetFlag<4, PIPE_S>(MTE_INPUT_READY_RELEASE_FLAG);
        AscendC::CrossCoreSetFlag<4, PIPE_S>(MTE_INPUT_READY_RELEASE_FLAG + MTE_AIV1_FLAG_OFFSET);
    } else {
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(MTE_INPUT_READY_RELEASE_FLAG);
    }
}

// AIV 在本核 PIPE_ALL 之后释放输入阶段 UB；AIC 仅在 FIX 流水上等待，允许提前装载和 MMAD。
// 每个 AIC/AIV 都配对一次，包括无共享专家、无计算 tile 的核，不能遗留硬件事件。
__aicore__ inline void ProtectMteInputUbReuse()
{
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(MTE_INPUT_UB_RELEASE_FLAG);
    } else {
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(MTE_INPUT_UB_RELEASE_FLAG);
        AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(MTE_INPUT_UB_RELEASE_FLAG + MTE_AIV1_FLAG_OFFSET);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_INPUT_PREPARE_SYNC_H
