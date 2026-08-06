/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_WAVE_GMM2_COMBINE_H
#define MEGA_MOE_WAVE_GMM2_COMBINE_H

#include "adv_api/reduce/reduce.h"
#include "aiv_comm/mega_moe_token_combine.h"

namespace MegaMoeImpl {

constexpr uint32_t WAVE_GMM2_READY_SCAN_UB_ADDR = 184U * 1024U;
constexpr uint32_t WAVE_GMM2_READY_MAX_SCAN_BYTES =
    MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512 ?
        MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) :
        ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR =
    WAVE_GMM2_READY_SCAN_UB_ADDR +
    ((WAVE_GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_SUM_UB_ADDR = WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;

struct WaveGmm2ReadyArgs {
    GM_ADDR gmm2ReadyPtr;
};

__aicore__ inline uint64_t GetWaveGmm2ReadySlotStride(const WaveCombineContext &context)
{
    return static_cast<uint64_t>(context.job.totalJobs) * INT_CACHELINE;
}

// Every AIC publishes one cache-line-isolated marker after its GMM2 writes are visible.
__aicore__ inline void NotifyWaveGmm2Ready(const WaveCombineContext &context,
                                           const WaveGmm2ReadyArgs &args, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(args.gmm2ReadyPtr);
        uint64_t slotOffset =
            static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(context);
        AscendC::WriteGmByPassDCache(
            readyBase + slotOffset + static_cast<uint64_t>(context.job.jobIndex) * INT_CACHELINE,
            int32_t(1));
    }
}

// One AIV Combine job waits until all AIC jobs have completed the selected expert.
__aicore__ inline void WaitWaveGmm2Ready(const WaveCombineContext &context,
                                         const WaveGmm2ReadyArgs &args, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!context.participates || context.job.totalJobs == 0U) {
        return;
    }
    uint32_t readyElements = context.job.totalJobs * static_cast<uint32_t>(INT_CACHELINE);
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(args.gmm2ReadyPtr);
    uint64_t readySlotOffset =
        static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(context);
    GlobalTensor<int32_t> readyGm;
    readyGm.SetGlobalBuffer(readyBase);
    LocalTensor<int32_t> scanUb(TPosition::VECCALC, WAVE_GMM2_READY_SCAN_UB_ADDR, readyElements);
    LocalTensor<uint8_t> reduceTmpUb(
        TPosition::VECCALC, WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR, ALIGN_512);
    LocalTensor<int32_t> sumUb(
        TPosition::VECCALC, WAVE_GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);
    const uint32_t readyShape[] = {1U, readyElements};
    while (true) {
        DataCopy(scanUb, readyGm[readySlotOffset], readyElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ReduceSum<int32_t, AscendC::Pattern::Reduce::AR, true>(
            sumUb, scanUb, reduceTmpUb, readyShape, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (sumUb.GetValue(0) >= static_cast<int32_t>(context.job.totalJobs)) {
            return;
        }
        int64_t waitStart = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStart < 100) {
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WAVE_GMM2_COMBINE_H
