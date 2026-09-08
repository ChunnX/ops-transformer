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
// mode 0 的独立事件；避开 SyncAll 使用的 11--14。GMM 的 mode 4 事件属于另一组计数器。
constexpr uint16_t MTE_QUANT_READY_FLAG = 8;

__aicore__ inline __gm__ int32_t *GetMteSyncCountAddress(GM_ADDR rankSyncBase, uint32_t aivIndex)
{
    return reinterpret_cast<__gm__ int32_t *>(rankSyncBase + MTE_SYNC_COUNT_OFFSET +
                                              static_cast<uint64_t>(aivIndex) * MTE_SYNC_COUNT_STRIDE);
}

__aicore__ inline uint32_t GetMteCountEpoch(int32_t inputSequence)
{
    return (static_cast<uint32_t>(inputSequence) & 0x7FU) | 0x80U;
}

// 所有 AIV（含空任务）在量化写回下发后调用。GM 计数只记账 epoch，不再表示数据就绪；
// 硬件在前序 MTE3 完成后上报 ready，发布端不插入 MTE3_S。出口仍推进一次计数。
__aicore__ inline int32_t PublishMteQuantReady(GM_ADDR rankSyncBase, uint32_t aivIndex)
{
    if constexpr (g_coreType == AscendC::AIV) {
        __gm__ int32_t *sequenceAddr = GetMteSyncCountAddress(rankSyncBase, aivIndex);
        uint32_t nextSequence = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(sequenceAddr)) + 1U;
        AscendC::WriteGmByPassDCache(sequenceAddr, static_cast<int32_t>(nextSequence));
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(MTE_QUANT_READY_FLAG);
        return static_cast<int32_t>(nextSequence);
    }
    return 0;
}

// 所有 AIV 都消费一次事件，不能在无专家任务时跳过。950 仅阻塞 MTE3，标量仍可下发。
// 有 route 的核把等待放在 index 之后、count 之前；无 route 的核在返回前配对等待。
__aicore__ inline void WaitForMteQuantReady()
{
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreWaitFlag<0, PIPE_MTE3>(MTE_QUANT_READY_FLAG);
    }
}

// 每项 count 在 UB 中独占一个 data block，允许跨 rank 段直接使用 Normal 二维搬运。
// 源 UB 在入口 PipeBarrier<PIPE_ALL> 之前不再改写；不需要 MTE3_S 或临时重排。
__aicore__ inline void PublishMteExpertCounts(GM_ADDR *winRankAddr, uint64_t countWinOffset, uint32_t sourceRank,
                                              uint32_t worldSize, uint32_t expertsPerRank, uint32_t ownedExpertBegin,
                                              uint32_t ownedExpertNum, int32_t inputSequence,
                                              const AscendC::LocalTensor<int32_t> &counts)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    if (ownedExpertNum == 0U) {
        return;
    }
    uint32_t epoch = GetMteCountEpoch(inputSequence);
    for (uint32_t idx = 0; idx < ownedExpertNum; ++idx) {
        uint32_t offset = idx * MTE_COUNT_UB_STRIDE;
        uint32_t word = (epoch << 24U) | static_cast<uint32_t>(counts.GetValue(offset));
        counts.SetValue(offset, static_cast<int32_t>(word));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SYNC_EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SYNC_EVENT_ID3);
    for (uint32_t ownedOffset = 0; ownedOffset < ownedExpertNum;) {
        uint32_t globalExpert = ownedExpertBegin + ownedOffset;
        uint32_t dstRank = globalExpert / expertsPerRank;
        uint32_t localExpert = globalExpert % expertsPerRank;
        uint32_t count = ownedExpertNum - ownedOffset;
        if (count > expertsPerRank - localExpert) {
            count = expertsPerRank - localExpert;
        }
        uint64_t offset =
            countWinOffset + (static_cast<uint64_t>(localExpert) * worldSize + sourceRank) * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(winRankAddr[dstRank] + offset));
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(count), sizeof(int32_t), 0,
                                        static_cast<int64_t>(worldSize - 1U) * static_cast<int64_t>(sizeof(int32_t)),
                                        0U};
        AscendC::DataCopyPad<int32_t, AscendC::PaddingMode::Normal>(dst, counts[ownedOffset * MTE_COUNT_UB_STRIDE],
                                                                    copy);
        ownedOffset += count;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_INPUT_SYNC_H
