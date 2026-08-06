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
            accumStartCycles_[i] = 0;
            accumDurationCycles_[i] = 0;
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

    // [kernel-timing] ???????????????????????????
    __aicore__ inline void AccumStartTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        accumStartCycles_[slotId] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void AccumStopTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        accumDurationCycles_[slotId] += AscendC::GetSystemCycle() - accumStartCycles_[slotId];
    }

    __aicore__ inline int32_t GetCurrentSlotId() const
    {
        return currentId_;
    }

    __aicore__ inline void WriteTimingToOutput(int32_t coreId, bool isAicActivated = false, int32_t aicNum = 0)
    {
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
            if (accumDurationCycles_[i] != 0) {
                // ?? slot ?? (0, ????)?Python ? duration = stop - start ??????
                output.SetValue(coreId * MAX_SLOTS * 2 + i * 2, 0);
                output.SetValue(coreId * MAX_SLOTS * 2 + i * 2 + 1, accumDurationCycles_[i]);
            } else {
                output.SetValue(coreId * MAX_SLOTS * 2 + i * 2, startCycles_[i]);
                output.SetValue(coreId * MAX_SLOTS * 2 + i * 2 + 1, stopCycles_[i]);
            }
        }
    }

private:
    uint64_t startCycles_[MAX_SLOTS];
    uint64_t stopCycles_[MAX_SLOTS];
    uint64_t accumStartCycles_[MAX_SLOTS];
    uint64_t accumDurationCycles_[MAX_SLOTS];
    int32_t  currentId_;
    GM_ADDR  gmOutputAddr_;
};

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
