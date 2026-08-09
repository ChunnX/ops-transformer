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

using namespace AscendC;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
struct ResetWorkspaceConfig {
    int32_t flagElementCount;
    int32_t resetBatchElementCount;
};

// 清理连续 flag workspace；prefetch 路径额外清理独立分配的 GMM1 tile 状态区。
template <bool TopkWeightsPrefetch>
__aicore__ inline void ResetDispatchWorkspace(const DispatchPrepareContext &context,
                                              const Params &params,
                                              const ResetWorkspaceConfig &config,
                                              LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || config.resetBatchElementCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, params.workspaceInfo.flagSwiGluToGmm2Ptr, config.flagElementCount,
                            config.resetBatchElementCount, resetTensor);
    if constexpr (TopkWeightsPrefetch) {
        int32_t statusElementCount =
            (static_cast<int32_t>(context.common.moeExpertPerRank) *
                 static_cast<int32_t>(params.tilingData->maxTilesPerExpert) +
             1) *
            INT_CACHELINE;
        ResetWorkspaceRegion<1>(job, params.workspaceInfo.gmm1TileStatusPtr, statusElementCount,
                                config.resetBatchElementCount, resetTensor);
    }
}
#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_RESET_H
