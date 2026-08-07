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
 * \file mega_moe_utils.h
 * \brief MegaMoe 调度使用的通用 host/device 辅助函数。
 */

#ifndef MEGA_MOE_UTILS_H
#define MEGA_MOE_UTILS_H

#include <cstdint>

#include "mega_moe_base.h"

namespace MegaMoeImpl {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define MEGA_MOE_HOST_DEVICE_INLINE __aicore__ inline
#else
#define MEGA_MOE_HOST_DEVICE_INLINE inline
#endif

struct TokenRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

// 返回当前 wave 的专家数；不足正常 wave 一半的尾块合并到前一个 wave。
MEGA_MOE_HOST_DEVICE_INLINE uint32_t GetWaveExpertCount(uint32_t waveBegin, uint32_t expertCount,
                                                        uint32_t expertsPerWave)
{
    if (waveBegin >= expertCount) {
        return 0U;
    }
    uint32_t remainingExperts = expertCount - waveBegin;
    if (expertsPerWave == 0U || remainingExperts <= expertsPerWave) {
        return remainingExperts;
    }
    uint32_t tailExperts = remainingExperts - expertsPerWave;
    bool mergeSmallTail = tailExperts < expertsPerWave && 2U * tailExperts < expertsPerWave;
    return mergeSmallTail ? remainingExperts : expertsPerWave;
}

// 连续均衡分配 token，前 totalTokens % workerCount 个任务各多处理一个 token。
MEGA_MOE_HOST_DEVICE_INLINE TokenRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx,
                                                             uint32_t workerCount)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t base = totalTokens / workerCount;
    uint32_t remainder = totalTokens % workerCount;
    uint32_t extraBefore = workerIdx < remainder ? workerIdx : remainder;
    return {workerIdx * base + extraBefore, base + static_cast<uint32_t>(workerIdx < remainder)};
}

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
struct ResetWorkspaceArgs {
    GM_ADDR flagWorkspacePtr;
    GM_ADDR gmm1TileStatusPtr;
    int32_t flagElementCount;
    int32_t gmm1StatusExpertCapacity;
    int32_t maxTilesPerExpert;
    int32_t resetBatchElementCount;
};

// 使用调用方准备的全零 UB Tensor，清理当前 AIV 任务负责的连续 GM 区域。
template <int32_t JobAlignment>
__aicore__ inline void ResetWorkspaceRegion(const AivJobContext &job, GM_ADDR regionPtr, int32_t elementCount,
                                            int32_t batchElementCount, LocalTensor<int32_t> &resetTensor)
{
    int32_t jobLen;
    int32_t jobOffset;
    TilingByJobContext(elementCount, jobLen, jobOffset, job.jobIndex, job.totalJobs, JobAlignment);
    GlobalTensor<int32_t> regionGm;
    regionGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(regionPtr));
    for (int32_t offset = 0; offset < jobLen; offset += batchElementCount) {
        int32_t batchCount = jobLen - offset < batchElementCount ? jobLen - offset : batchElementCount;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(regionGm[jobOffset + offset], resetTensor, copyParams);
    }
}

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

#undef MEGA_MOE_HOST_DEVICE_INLINE
} // namespace MegaMoeImpl

#endif // MEGA_MOE_UTILS_H
