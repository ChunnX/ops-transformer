/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"

namespace ChunkGatedDeltaRule {

// 核间同步描述谁完成了什么；枚举值为 slot 0 的 mode-2 flag ID。
// 一个 AIC 与两个 AIV 共享计数握手：每个 head 的 Notify/Wait 必须成对收尾。
// 它与下方核内 Mutex 各自管理不同的依赖，不能互相替代。
enum class CubeToVectorEvent : uint16_t {
    PrefixScoresReady = 0, // FIX: KKT/QKT 已从 L0C 搬到两个 AIV 的 UB；slot 1 使用 flag 10。
    StateDeltaReady = 2,   // FIX: delta=R.T()@Kg 已从 L0C 搬到 UB。
    StateBufferFree = 9,   // MTE1: 旧 L1 状态已搬入 L0B，Vector 可以更新状态副本。
    ChunkSlotFree = 11,    // MTE1: 当前 L1 chunk 槽已读完；slot 1 使用 flag 12。
    ResidualReady = 14,    // FIX: FP32 R 已从 L0C 搬到 UB，可转换为 Cube 所需的 BF16 副本。
};

enum class VectorToCubeEvent : uint16_t {
    PrefixScoresBufferFree = 1, // V: KKT/QKT 消费完毕，Cube 可以复用 UB；slot 1 使用 flag 13。
    StateDeltaBufferFree = 3,   // V: 状态增量消费完毕，Cube 可以复用其 UB。
    InverseInputsReady = 4,     // MTE3: 求逆/状态投影所需数据已到 L1；slot 1 使用 flag 5。
    CausalScoresReady = 6,      // MTE3: maskedQkt 已到 L1；slot 1 使用 flag 7。
    StateReady = 8,             // MTE3: BF16 状态副本已到 L1。
    ResidualReady = 15,         // MTE3: BF16 R 已从 UB 搬到 L1。
};

template <CubeToVectorEvent event>
__aicore__ inline constexpr uint16_t CrossCoreEventId(uint32_t slot = 0)
{
    if constexpr (event == CubeToVectorEvent::PrefixScoresReady) {
        return slot == 0 ? 0 : 10;
    } else if constexpr (event == CubeToVectorEvent::ChunkSlotFree) {
        return slot == 0 ? 11 : 12;
    } else {
        return static_cast<uint16_t>(event);
    }
}

template <VectorToCubeEvent event>
__aicore__ inline constexpr uint16_t CrossCoreEventId(uint32_t slot = 0)
{
    if constexpr (event == VectorToCubeEvent::PrefixScoresBufferFree) {
        return slot == 0 ? 1 : 13;
    } else if constexpr (event == VectorToCubeEvent::InverseInputsReady) {
        return slot == 0 ? 4 : 5;
    } else if constexpr (event == VectorToCubeEvent::CausalScoresReady) {
        return slot == 0 ? 6 : 7;
    } else {
        return static_cast<uint16_t>(event);
    }
}

// 事件类型在编译期检查通信方向；slot 为 chunk&1，单槽事件忽略此参数。
// Notify 的 PIPE 属于生产方，必须紧跟原来的 FIX/MTE1/MTE2/MTE3/V 工作完成点。
// Wait 沿用 CrossCoreWaitFlag 的默认 scalar wait，不额外插入全流水 barrier。
template <CubeToVectorEvent event>
__aicore__ inline void WaitForCube(uint32_t slot = 0)
{
    AscendC::CrossCoreWaitFlag(CrossCoreEventId<event>(slot));
}

template <VectorToCubeEvent event>
__aicore__ inline void WaitForVector(uint32_t slot = 0)
{
    AscendC::CrossCoreWaitFlag(CrossCoreEventId<event>(slot));
}

template <pipe_t producerPipe, CubeToVectorEvent event>
__aicore__ inline void NotifyVector(uint32_t slot = 0)
{
    AscendC::CrossCoreSetFlag<0x2, producerPipe>(CrossCoreEventId<event>(slot));
}

template <pipe_t producerPipe, VectorToCubeEvent event>
__aicore__ inline void NotifyCube(uint32_t slot = 0)
{
    AscendC::CrossCoreSetFlag<0x2, producerPipe>(CrossCoreEventId<event>(slot));
}

} // namespace ChunkGatedDeltaRule
