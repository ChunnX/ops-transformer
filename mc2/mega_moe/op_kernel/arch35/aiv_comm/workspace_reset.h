/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_WORKSPACE_RESET_H
#define MEGA_MOE_WORKSPACE_RESET_H

#include "../single_core_context.h"

namespace MegaMoeImpl {

struct ResetWorkspaceArgs {
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    int32_t flagNum;
    int32_t gmm2CombineSyncCounterNum;
    int32_t resetBatchElementCount;
};

// Prototype: MegaMoe::ResetGmm2CombineSyncCounters. Clears one logical AIV job's combine counters in batches.
// ResetFlagListOneCore validates the job and prepares the zero-filled resetTensor before calling this helper.
__aicore__ inline void ResetGmm2CombineSyncCountersOneCore(const AivThreadContext &thread,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
{
    int32_t threadLen, threadOffset;
    TilingByThreadContext(args.gmm2CombineSyncCounterNum, threadLen, threadOffset,
        thread.jobIndex, thread.totalJobs);
    GlobalTensor<int32_t> counterGm;
    counterGm.SetGlobalBuffer((__gm__ int32_t *)args.gmm2CombineSyncCounterPtr);
    for (int32_t offset = 0; offset < threadLen; offset += args.resetBatchElementCount) {
        int32_t batchCount = threadLen - offset < args.resetBatchElementCount ?
            threadLen - offset : args.resetBatchElementCount;
        DataCopy(counterGm[threadOffset + offset], resetTensor, batchCount);
    }
}

// Prototype: MegaMoe::ResetFlagList. Clears one logical AIV job's dispatch flags in bounded UB batches.
template <bool ResetCombineCounters>
__aicore__ inline void ResetFlagListOneCore(const DispatchPrepareContext &context,
    const ResetWorkspaceArgs &args, LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivThreadContext &thread = context.thread;
    if (thread.totalJobs == 0U || thread.jobIndex >= thread.totalJobs || args.resetBatchElementCount <= 0) {
        return;
    }
    GlobalTensor<int32_t> flagGm;
    flagGm.SetGlobalBuffer((__gm__ int32_t *)args.flagSwiGluToGmm2Ptr);
    int32_t threadLen, threadOffset;
    TilingByThreadContext(args.flagNum, threadLen, threadOffset, thread.jobIndex, thread.totalJobs, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    for (int32_t offset = 0; offset < threadLen; offset += args.resetBatchElementCount) {
        int32_t batchCount = threadLen - offset < args.resetBatchElementCount ?
            threadLen - offset : args.resetBatchElementCount;
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(flagGm[threadOffset + offset], resetTensor, copyParams);
    }
    if constexpr (ResetCombineCounters) {
        ResetGmm2CombineSyncCountersOneCore(thread, args, resetTensor);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_RESET_H
