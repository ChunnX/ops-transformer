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
 * \file comm_sync_utils.h
 * \brief AIV/AIC 跨核 GM 计数器同步工具（cube 通知 vector 的释放机制）
 *
 * 参考 mega_moe 的 GM atomicAdd 计数器 + AIV 轮询模式：
 * - AIC 侧通过 AtomicAdd 累加每 tile 的释放计数；
 * - AIV 侧通过 ReadGmByPassDCache 轮询直到计数达到期望值。
 */

#pragma once

#include "kernel_operator.h"

namespace Apace {

using namespace AscendC;

// 每个通信 tile 一个缓存行对齐的 int32 槽（16 个 int32 = 64B），避免多核 atomicAdd false-sharing。
constexpr uint32_t COMM_SYNC_SLOT_STRIDE = 16;
// AIV 清零计数器时使用的 UB 零块大小（int32 元素个数）。
constexpr uint32_t COMM_SYNC_ZERO_BLOCK = 256;

// 轮询 GM 中的 int32 计数直至不小于目标值，两次读取间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagAtLeast(__gm__ int32_t *flagAddr, int32_t targetValue)
{
    constexpr int64_t POLL_BACKOFF_CYCLES = 100;
    while (ReadGmByPassDCache(flagAddr) < targetValue) {
        int64_t startCycle = GetSystemCycle();
        while (GetSystemCycle() - startCycle < POLL_BACKOFF_CYCLES) {
        }
    }
}

} // namespace Apace
