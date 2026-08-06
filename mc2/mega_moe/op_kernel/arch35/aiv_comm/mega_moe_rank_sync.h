/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_RANK_SYNC_H
#define MEGA_MOE_RANK_SYNC_H

#include "../mega_moe_impl_base.h"
#include "../mega_moe_job_context.h"

namespace MegaMoeImpl {

struct RankSyncContext {
    AivJobContext job;
    uint32_t rankId;
    uint32_t worldSize;
};

struct RankSyncArgs {
    GM_ADDR *winRankAddr;
    GM_ADDR rankSyncInWorldPtr;
};

// Prototype: MegaMoe::CrossRankSyncInWorldSize. Synchronizes participating ranks across the world.
__aicore__ inline void SynchronizeRanksInWorld(const RankSyncContext &context, const RankSyncArgs &args)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (context.job.totalJobs == 0U || context.job.jobIndex >= context.job.totalJobs) {
        return;
    }
    __gm__ int32_t *syncRank = reinterpret_cast<__gm__ int32_t *>(args.rankSyncInWorldPtr);
    __gm__ int32_t *syncCount = reinterpret_cast<__gm__ int32_t *>(
        args.rankSyncInWorldPtr + 48U * 1024U + static_cast<uint64_t>(context.job.jobIndex) * 64U);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIdx = context.job.jobIndex; rankIdx < context.worldSize;
         rankIdx += context.job.totalJobs) {
        __gm__ int32_t *remoteSyncAddr =
            reinterpret_cast<__gm__ int32_t *>(args.winRankAddr[rankIdx]) + context.rankId * INT_CACHELINE;
        WriteGmByPassDCache(remoteSyncAddr, count);
        GmSignalWaitBarrier(syncRank + rankIdx * INT_CACHELINE, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_RANK_SYNC_H
