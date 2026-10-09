/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_RING_SYNC_H
#define MEGA_MOE_RING_SYNC_H

#include "mega_moe_utils.h"

namespace MegaMoeImpl {

// Wave iteration w produces Dispatch(w+1), GMM1/Activation(w), and GMM2/Combine(w-1) when lagged.
// Three independent wave budgets permit one iteration of role skew. Before iteration w >= 2, completion
// of iteration w-2 by every role protects every bank being overwritten. No role waits for its current
// iteration, so AIV1's next dispatch cannot prevent the Activation needed to release an older bank.
__aicore__ inline void WaitMteRingIteration(const Params &params, uint64_t iteration)
{
    if (!IsMteRingBufferEnabled(params.tilingData) || iteration < MTE_RING_WAVE_COUNT - 1U) {
        return;
    }
    const int32_t requiredProgress = static_cast<int32_t>(iteration - MTE_RING_WAVE_COUNT + 2U);
    __gm__ int32_t *progress = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ringProgressPtr);
    const uint32_t roleCount = params.tilingData->aicNum * 3U;
    for (uint32_t role = 0U; role < roleCount; ++role) {
        WaitUntilGmFlagAtLeast(progress + static_cast<uint64_t>(role) * INT_CACHELINE, requiredProgress);
    }
}

// Called outside scheduler/zero-work branches by every AIC, AIV0 and AIV1. The watermarks are part of
// the normal per-launch reset region. Drain asynchronous GM transfers before publishing scalar progress;
// function return alone is not a consumer-completion guarantee. Event 6 is not retained across this call.
__aicore__ inline void PublishMteRingIteration(const Params &params, const BlockJobContext &job,
                                               uint64_t completedIterations)
{
    if (!IsMteRingBufferEnabled(params.tilingData)) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID6>();
    uint32_t role = job.jobIndex;
    if constexpr (g_coreType == AscendC::AIC) {
        SyncFuncStatic<AscendC::HardEvent::FIX_S, SYNC_EVENT_ID6>();
    } else {
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID6>();
        role = params.tilingData->aicNum + job.jobIndex * 2U + GetSubBlockIdx();
    }
    __gm__ int32_t *progress = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ringProgressPtr);
    // The number of 256-row groups is bounded by uint32 maxOutputSize plus expert-tail padding.
    AscendC::WriteGmByPassDCache(progress + static_cast<uint64_t>(role) * INT_CACHELINE,
                                static_cast<int32_t>(completedIterations));
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_RING_SYNC_H
