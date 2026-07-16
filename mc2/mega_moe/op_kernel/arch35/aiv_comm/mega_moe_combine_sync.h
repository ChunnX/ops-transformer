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
 * \file mega_moe_combine_sync.h
 * \brief Pure AIV cross-rank synchronization for the combine pipeline.
 */

#ifndef MEGA_MOE_COMBINE_SYNC_H
#define MEGA_MOE_COMBINE_SYNC_H

#include "kernel_operator.h"
#include "../single_core_context.h"
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#endif

using namespace AscendC;

namespace MegaMoeCombineCommImpl {

struct CrossRankSyncContext {
    GM_ADDR rankSyncAddr;
    GM_ADDR* winRankAddr;
    uint32_t rankId;
    uint32_t worldSize;
};

// Prototype: MegaMoe::CrossRankSyncInWorldSize. Synchronizes one logical AIV thread's assigned ranks.
// The caller owns any barrier required among participating logical threads.
__aicore__ inline void CrossRankSyncOneCore(const MegaMoeImpl::AivThreadContext& thread,
    const CrossRankSyncContext& syncContext)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (thread.threadNum == 0 || thread.threadIdx >= thread.threadNum) {
        return;
    }
    __gm__ int32_t* syncRank = reinterpret_cast<__gm__ int32_t*>(syncContext.rankSyncAddr);
    __gm__ int32_t* syncCount = reinterpret_cast<__gm__ int32_t*>(
        syncContext.rankSyncAddr + 48 * 1024 + thread.threadIdx * 64);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIndex = thread.threadIdx; rankIndex < syncContext.worldSize;
        rankIndex += thread.threadNum) {
        __gm__ int32_t* syncRemoteAddr = reinterpret_cast<__gm__ int32_t*>(syncContext.winRankAddr[rankIndex])
            + syncContext.rankId * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        GmSignalWaitBarrier(syncRank + rankIndex * 16, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
}

} // namespace MegaMoeCombineCommImpl

#endif // MEGA_MOE_COMBINE_SYNC_H
