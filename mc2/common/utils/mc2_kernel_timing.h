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
 *        StartTiming records the raw GetSystemCycle() start timestamp,
 *        StopTiming records the raw GetSystemCycle() stop timestamp.
 *        Both timestamps are written to GM for post-processing on the host side.
 *        Supports both AIV-only and AIC+AIV mixed execution via isAicActivated flag.
 *        Each slot uses 2 uint32 values (start, stop), 16 slots per core,
 *        totaling 32 uint32 values per core.
 */

#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

class KernelTiming {
public:
    static constexpr int32_t MAX_SLOTS = 16;

    __aicore__ inline KernelTiming() : gmOutputAddr_(nullptr), currentId_(0) {}

    __aicore__ inline void Init(GM_ADDR outputAddr)
    {
        gmOutputAddr_ = outputAddr;
        currentId_ = 0;
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            startCycles_[i] = 0;
            stopCycles_[i] = 0;
        }
    }

    __aicore__ inline void StartTiming(int32_t slotId = -1)
    {
        if (slotId >= 0) {
            currentId_ = slotId;
        } else {
            currentId_++;
        }
        if (currentId_ < 0 || currentId_ >= MAX_SLOTS) {
            return;
        }
        startCycles_[currentId_] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void StopTiming(int32_t slotId = -1)
    {
        if (slotId >= 0) {
            currentId_ = slotId;
        }
        if (currentId_ < 0 || currentId_ >= MAX_SLOTS) {
            return;
        }
        stopCycles_[currentId_] = AscendC::GetSystemCycle();
    }

    __aicore__ inline int32_t GetCurrentSlotId() const
    {
        return currentId_;
    }

    __aicore__ inline void WriteTimingToOutput(int32_t coreId, bool isAicActivated = false, int32_t aicNum = 0)
    {
        // Cross-core barrier: ensure all cores have finished their MTE3 writes
        // (dispatch output) before any core overwrites expand_x with timing data.
        PipeBarrier<PIPE_ALL>();
        if (isAicActivated) {
            SyncAll<false>();
        } else {
            SyncAll<true>();
        }

        GlobalTensor<uint64_t> output;
        output.SetGlobalBuffer((__gm__ uint64_t *)gmOutputAddr_);

        if (isAicActivated && g_coreType == AscendC::AIV) {
            coreId = coreId + aicNum;
        }
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            output.SetValue(coreId * MAX_SLOTS * 2 + i * 2, startCycles_[i]);
            output.SetValue(coreId * MAX_SLOTS * 2 + i * 2 + 1, stopCycles_[i]);
        }
    }

private:
    uint64_t startCycles_[MAX_SLOTS];
    uint64_t stopCycles_[MAX_SLOTS];
    int32_t  currentId_;
    GM_ADDR  gmOutputAddr_;
};

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
