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

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
struct ResetWorkspaceArgs {
    GM_ADDR flagWorkspacePtr;
    GM_ADDR gmm1TileStatusPtr;
    int32_t flagElementCount;
    int32_t gmm1StatusExpertCapacity;
    int32_t maxTilesPerExpert;
    int32_t resetBatchElementCount;
};

// 清理连续 flag workspace；prefetch 路径额外清理独立分配的 GMM1 tile 状态区。
template <bool TopkWeightsPrefetch>
__aicore__ inline void ResetDispatchWorkspace(const DispatchPrepareContext &context,
                                              const ResetWorkspaceArgs &args,
                                              LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || args.resetBatchElementCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, args.flagWorkspacePtr, args.flagElementCount,
                            args.resetBatchElementCount, resetTensor);
    if constexpr (TopkWeightsPrefetch) {
        int32_t statusElementCount =
            (args.gmm1StatusExpertCapacity * args.maxTilesPerExpert + 1) * INT_CACHELINE;
        ResetWorkspaceRegion<1>(job, args.gmm1TileStatusPtr, statusElementCount,
                                args.resetBatchElementCount, resetTensor);
    }
}
#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_RESET_H
