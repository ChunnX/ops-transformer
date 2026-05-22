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
 * \file mc2_kernel_timing.h
 * \brief Slot-based kernel timing utility.
 *        StartTiming records the begin time, StopTiming computes the elapsed time
 *        and accumulates into the slot. Repeated start/stop on the same slot
 *        accumulates total time (e.g. across loop iterations).
 *        Uses GlobalTensor::SetValue for output — no UB buffer allocation,
 *        immune to TPipe::Reset() and UB capacity issues.
 *        Always outputs exactly 32 uint32 values per core.
 */

#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

class KernelTiming {
public:
    static constexpr int32_t MAX_SLOTS = 32;

    __aicore__ inline KernelTiming() : gmOutputAddr_(0) {}

    __aicore__ inline void Init(__gm__ uint32_t *outputAddr)
    {
        gmOutputAddr_ = reinterpret_cast<uint64_t>(outputAddr);
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            accumulated_[i] = 0;
            pendingStart_[i] = 0;
        }
    }

    __aicore__ inline void StartTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        pendingStart_[slotId] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void StopTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS || pendingStart_[slotId] == 0) {
            return;
        }
        uint64_t now = AscendC::GetSystemCycle();
        accumulated_[slotId] += static_cast<uint32_t>(now - pendingStart_[slotId]);
        pendingStart_[slotId] = 0;
    }

    __aicore__ inline void WriteTimingToOutput(int32_t coreId)
    {
        // Each slot occupies one 64B cache line to avoid write-back conflicts
        // between consecutive SetValue calls.  uint32 stride = 64 / 4 = 16.
        constexpr int32_t STRIDE = 16;
        GlobalTensor<uint32_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(gmOutputAddr_));
        uint32_t base = coreId * MAX_SLOTS * STRIDE;
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            output.SetValue(base + i * STRIDE, accumulated_[i]);
        }
    }

private:
    uint64_t pendingStart_[MAX_SLOTS];
    uint32_t accumulated_[MAX_SLOTS];
    uint64_t gmOutputAddr_;
};

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
