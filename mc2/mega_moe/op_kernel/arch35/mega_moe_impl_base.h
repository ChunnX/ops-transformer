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
 * \file mega_moe_imp_base.h
 * \brief
 */

#ifndef MEGA_MOE_IMP_BASE_H
#define MEGA_MOE_IMP_BASE_H

namespace MegaMoeImpl {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define MEGA_MOE_HOST_DEVICE_INLINE __aicore__ inline
#else
#define MEGA_MOE_HOST_DEVICE_INLINE inline
#endif

// Diagnostic build used to locate the rolling GMM2/Combine critical path.
// Event-time samples stay in named scalar state and are copied to y only after
// all useful work has completed.  Keeping the fields named (rather than using
// an indexed array) also avoids device-side dynamic indexing and the spills it
// can introduce on the hot path.
#ifndef MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE
#define MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE 1
#endif

static constexpr bool CRITICAL_PATH_TRACE_ENABLED =
    static_cast<bool>(MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE);
static constexpr uint32_t CRITICAL_PATH_TRACE_MAGIC = 0x43505452U; // "CPTR"
static constexpr uint32_t CRITICAL_PATH_TRACE_VERSION = 1U;
static constexpr uint32_t CRITICAL_PATH_TRACE_HEADER_WORDS = 20U;
static constexpr uint32_t CRITICAL_PATH_TRACE_RECORD_WORDS = 26U;

struct CriticalPathTrace {
    uint32_t processBegin = 0U;
    uint32_t rollingBegin = 0U;
    uint32_t dispatchNextWaveEnd = 0U;
    uint32_t dispatchSecondNextWaveEnd = 0U;
    uint32_t firstPublishBegin = 0U;
    uint32_t firstPublishEnd = 0U;
    uint32_t lastPublishBegin = 0U;
    uint32_t lastPublishEnd = 0U;
    uint32_t firstReadyBegin = 0U;
    uint32_t firstReadyEnd = 0U;
    uint32_t lastReadyBegin = 0U;
    uint32_t lastReadyEnd = 0U;
    uint32_t firstCombineBegin = 0U;
    uint32_t firstCombineScalarEnd = 0U;
    uint32_t lastCombineBegin = 0U;
    uint32_t lastCombineScalarEnd = 0U;
    uint32_t lastCombineActualEnd = 0U;
    uint32_t rollingBodyEnd = 0U;
    uint32_t rollingEnd = 0U;
    uint32_t localSyncEnd = 0U;
    uint32_t crossRankEnd = 0U;
    uint32_t workEnd = 0U;
    uint32_t sendCntWaitCycles = 0U;
    uint32_t gmm1ReadyWaitCycles = 0U;
    uint32_t gmm2ReadyWaitCycles = 0U;
    uint32_t combineReadyWaitCycles = 0U;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
    __aicore__ inline static uint32_t Now()
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            return static_cast<uint32_t>(AscendC::GetSystemCycle());
        }
        return 0U;
    }

    __aicore__ inline void BeginPublish(uint32_t now)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            if (firstPublishBegin == 0U) {
                firstPublishBegin = now;
            }
            lastPublishBegin = now;
        }
    }

    __aicore__ inline void EndPublish(uint32_t now)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            if (firstPublishEnd == 0U) {
                firstPublishEnd = now;
            }
            lastPublishEnd = now;
        }
    }

    __aicore__ inline void BeginReady(uint32_t now)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            if (firstReadyBegin == 0U) {
                firstReadyBegin = now;
            }
            lastReadyBegin = now;
        }
    }

    __aicore__ inline void EndReady(uint32_t now, uint32_t begin)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            if (firstReadyEnd == 0U) {
                firstReadyEnd = now;
            }
            lastReadyEnd = now;
            combineReadyWaitCycles += now - begin;
        }
    }

    __aicore__ inline void BeginCombine(uint32_t now)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            if (firstCombineBegin == 0U) {
                firstCombineBegin = now;
            }
            lastCombineBegin = now;
        }
    }

    __aicore__ inline void EndCombineScalar(uint32_t now)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            if (firstCombineScalarEnd == 0U) {
                firstCombineScalarEnd = now;
            }
            lastCombineScalarEnd = now;
        }
    }

    __aicore__ inline void AddSendCntWait(uint32_t begin)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            sendCntWaitCycles += Now() - begin;
        }
    }

    __aicore__ inline void AddGmm1ReadyWait(uint32_t begin)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            gmm1ReadyWaitCycles += Now() - begin;
        }
    }

    __aicore__ inline void AddGmm2ReadyWait(uint32_t begin)
    {
        if constexpr (CRITICAL_PATH_TRACE_ENABLED) {
            gmm2ReadyWaitCycles += Now() - begin;
        }
    }
#endif
};

constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M_256 = 256;
constexpr uint32_t L1_TILE_M_128 = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t L0_TILE_K = 128;
constexpr uint32_t SCALE_K_L1_RATE = 2;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN_256_256 = 256 * 256;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = (MAX_SINGLE_MN_256_256 + 31U) / ALIGN32 * ALIGN32;
constexpr uint32_t MAX_SINGLE_MN_128_256 = 128 * 256;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = (MAX_SINGLE_MN_128_256 + 31U) / ALIGN32 * ALIGN32;
constexpr uint32_t TRIPLE_TENSOR_ADDR = 200U * 1024U; // triple tensor 在 UB 中的起始地址

struct TokenRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

// Balanced contiguous ownership.  Every token belongs to exactly one worker;
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
#endif
