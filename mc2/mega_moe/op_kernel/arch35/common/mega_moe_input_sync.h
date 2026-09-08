/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INPUT_SYNC_H
#define MEGA_MOE_INPUT_SYNC_H

#include "kernel_operator.h"
#include "mega_moe_constants.h"

namespace MegaMoeImpl {

// MTE 窗口原有的逐 AIV 同步计数槽。每槽独占 64B，仅所属 AIV 标量写入；
// 与前面的远端 rank 信号、后面的通信数据、workspace MTE 清零区均不重叠。
constexpr uint64_t MTE_SYNC_COUNT_OFFSET = 48U * 1024U;
constexpr uint64_t MTE_SYNC_COUNT_STRIDE = INT_CACHELINE * sizeof(int32_t);

__aicore__ inline __gm__ int32_t *GetMteSyncCountAddress(GM_ADDR rankSyncBase, uint32_t aivIndex)
{
    return reinterpret_cast<__gm__ int32_t *>(rankSyncBase + MTE_SYNC_COUNT_OFFSET +
                                              static_cast<uint64_t>(aivIndex) * MTE_SYNC_COUNT_STRIDE);
}

__aicore__ inline uint32_t GetMteCountEpoch(int32_t inputSequence)
{
    return (static_cast<uint32_t>(inputSequence) & 0x7FU) | 0x80U;
}

// 每个 AIV 都调用，包括 token 数为 0、没有 route 专家任务的核。
// 量化完成后推进原入口同步的计数，但不发送/等待跨 rank 信号。出口仍推进一次，
// 因而每轮计数与原协议一致。无入口清零：旧的出口值与本轮输入值不同。
__aicore__ inline void PublishMteQuantReady(GM_ADDR rankSyncBase, uint32_t aivIndex)
{
    if constexpr (g_coreType == AscendC::AIV) {
        __gm__ int32_t *sequenceAddr = GetMteSyncCountAddress(rankSyncBase, aivIndex);
        uint32_t nextSequence = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(sequenceAddr)) + 1U;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(SYNC_EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(SYNC_EVENT_ID5);
        AscendC::WriteGmByPassDCache(sequenceAddr, static_cast<int32_t>(nextSequence));
    }
}

// 只在 route 发送者发布 count 前等待。快核可先生成/发送 index，与慢核量化重叠。
// 必须比较完整计数，不能用 count 的 8-bit epoch 轮询本地生产者。
// 此后保留的 SyncAll<false> 保证所有发送者结束读取，任一 AIV 才可能推进出口计数；
// 因此同址复用不会让慢读者错过本轮值。轮询使用标量 bypass 和退避，不用 MTE 搬状态。
__aicore__ inline void WaitForMteQuantReady(GM_ADDR rankSyncBase, uint32_t aivIndex, uint32_t aivCount,
                                            int32_t inputSequence)
{
    if constexpr (g_coreType == AscendC::AIV) {
        uint32_t peerIndex = aivIndex;
        for (uint32_t peer = 1U; peer < aivCount; ++peer) {
            peerIndex = peerIndex + 1U == aivCount ? 0U : peerIndex + 1U;
            __gm__ int32_t *sequenceAddr = GetMteSyncCountAddress(rankSyncBase, peerIndex);
            while (AscendC::ReadGmByPassDCache(sequenceAddr) != inputSequence) {
                int64_t startCycle = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
                }
            }
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_INPUT_SYNC_H
