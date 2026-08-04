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
 * \brief Common host/device helpers for MegaMoe scheduling.
 */

#ifndef MEGA_MOE_UTILS_H
#define MEGA_MOE_UTILS_H

#include <cstdint>

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

// 计算单个专家贡献的独立 M 分组数。不同专家使用不同权重，不能合并成一次矩阵乘，
// 因此不同专家各自不足 rowsPerMGroup 的尾块也必须分别占用一个分组。
MEGA_MOE_HOST_DEVICE_INLINE uint32_t GetMGroupCountForRows(uint64_t rowCount, uint32_t rowsPerMGroup)
{
    if (rowCount == 0U || rowsPerMGroup == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>((rowCount - 1U) / rowsPerMGroup + 1U);
}

// 反推铺满目标流水深度所需的最少 M 分组数。一个 M 分组产生 tilesPerMGroup 个任务，
// 目标是让每个物理 AIC 至少获得 minTileTasksPerCore 个调度任务。
MEGA_MOE_HOST_DEVICE_INLINE uint32_t GetRequiredMGroupCount(uint32_t coreCount, uint32_t tilesPerMGroup,
                                                            uint32_t minTileTasksPerCore)
{
    if (coreCount == 0U || tilesPerMGroup == 0U || minTileTasksPerCore == 0U) {
        return 1U;
    }
    uint64_t requiredTileTaskCount = static_cast<uint64_t>(coreCount) * minTileTasksPerCore;
    return static_cast<uint32_t>((requiredTileTaskCount - 1U) / tilesPerMGroup + 1U);
}

// 计算一个路由专家 Wave 固定包含多少个专家内 256-row 分组。GMM1 项用于维持生产者深度，
// 让 GMM1 能领先 SwiGLU/GMM2；GMM2 项保证正常 Wave 至少能给每个物理 AIC 一个任务。
MEGA_MOE_HOST_DEVICE_INLINE uint32_t GetMGroupsPerWave(uint32_t coreCount, uint32_t gmm1TilesPerMGroup,
                                                       uint32_t gmm2TilesPerMGroup,
                                                       uint32_t gmm1MinTileTasksPerCore)
{
    uint32_t gmm1RequiredMGroupCount =
        GetRequiredMGroupCount(coreCount, gmm1TilesPerMGroup, gmm1MinTileTasksPerCore);
    uint32_t gmm2RequiredMGroupCount = GetRequiredMGroupCount(coreCount, gmm2TilesPerMGroup, 1U);
    return gmm1RequiredMGroupCount > gmm2RequiredMGroupCount ?
               gmm1RequiredMGroupCount :
               gmm2RequiredMGroupCount;
}

// 从当前专家截取能放入剩余分组预算的最大连续行段。除专家尾部外，segmentStartRow 始终按
// rowsPerMGroup 对齐，因此本行段向上取整后的分组数正好等于它消耗的调度预算。
MEGA_MOE_HOST_DEVICE_INLINE uint32_t GetWaveSegmentRowCount(uint64_t expertRowCount, uint32_t segmentStartRow,
                                                            uint32_t remainingMGroupCount,
                                                            uint32_t rowsPerMGroup)
{
    if (remainingMGroupCount == 0U || rowsPerMGroup == 0U || segmentStartRow >= expertRowCount) {
        return 0U;
    }
    uint64_t remainingExpertRowCount = expertRowCount - segmentStartRow;
    uint64_t maxRowCountAllowedByWave = static_cast<uint64_t>(remainingMGroupCount) * rowsPerMGroup;
    return static_cast<uint32_t>(remainingExpertRowCount < maxRowCountAllowedByWave ?
                                     remainingExpertRowCount :
                                     maxRowCountAllowedByWave);
}

// Return the number of experts in the current wave. A final tail containing
// strictly less than half a normal wave is absorbed by its preceding wave.
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

// Balanced contiguous ownership. Every token belongs to exactly one worker;
// the first (totalTokens % workerCount) workers receive one extra token.
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

#undef MEGA_MOE_HOST_DEVICE_INLINE
} // namespace MegaMoeImpl

#endif // MEGA_MOE_UTILS_H
