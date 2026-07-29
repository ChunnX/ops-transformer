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
 * \file mega_moe_wave_utils.h
 * \brief Host/device helpers used only by the MegaMoe wave scheduler.
 */

#ifndef MEGA_MOE_WAVE_UTILS_H
#define MEGA_MOE_WAVE_UTILS_H

#include <cstdint>

namespace MegaMoeImpl {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define MEGA_MOE_WAVE_HOST_DEVICE_INLINE __aicore__ inline
#else
#define MEGA_MOE_WAVE_HOST_DEVICE_INLINE inline
#endif

struct TokenRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

// Return the begin expert of a penultimate wave that should absorb a small
// final tail. expertCount is returned when the regular partition is retained.
MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t GetSmallTailMergeBegin(uint32_t expertCount, uint32_t expertsPerWave)
{
    if (expertCount == 0U || expertsPerWave == 0U || expertsPerWave >= expertCount) {
        return expertCount;
    }
    uint32_t tailExperts = expertCount % expertsPerWave;
    bool mergeSmallTail = tailExperts != 0U && 2U * tailExperts < expertsPerWave;
    return mergeSmallTail ? expertCount - expertsPerWave - tailExperts : expertCount;
}

// The consumer wave and previous-wave dispatch prefetch share this boundary.
MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t GetWaveEnd(uint32_t waveBegin, uint32_t expertCount,
                                                      uint32_t expertsPerWave, uint32_t smallTailMergeBegin)
{
    if (waveBegin >= expertCount || expertsPerWave == 0U || waveBegin == smallTailMergeBegin) {
        return expertCount;
    }
    uint32_t regularEnd = waveBegin + expertsPerWave;
    return regularEnd < expertCount ? regularEnd : expertCount;
}

// Balanced contiguous ownership. Every token belongs to exactly one worker;
// the first (totalTokens % workerCount) workers receive one extra token.
MEGA_MOE_WAVE_HOST_DEVICE_INLINE TokenRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx,
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

#undef MEGA_MOE_WAVE_HOST_DEVICE_INLINE
} // namespace MegaMoeImpl

#endif // MEGA_MOE_WAVE_UTILS_H
