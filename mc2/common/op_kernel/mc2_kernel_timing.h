#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

class KernelTiming {
public:
    static constexpr int32_t MAX_SLOTS = 16;

    KernelTiming() : gmOutputAddr_(nullptr), currentId_(0), numSlotsToWrite_(MAX_SLOTS) {}

    __aicore__ inline void Init(GM_ADDR outputAddr, int32_t numSlotsToWrite = MAX_SLOTS)
    {
        gmOutputAddr_ = outputAddr;
        currentId_ = 0;
        numSlotsToWrite_ = numSlotsToWrite;
        for (int32_t i = 0; i < MAX_SLOTS; i++) {
            startCycles_[i] = 0;
            accumCycles_[i] = 0;
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
        accumCycles_[currentId_] += AscendC::GetSystemCycle() - startCycles_[currentId_];
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
        int32_t slotsToWrite = (numSlotsToWrite_ > 0 && numSlotsToWrite_ <= MAX_SLOTS) ? numSlotsToWrite_ : MAX_SLOTS;
        for (int32_t i = 0; i < slotsToWrite; i++) {
            output.SetValue(coreId * slotsToWrite * 2 + i * 2, (uint64_t)0);
            output.SetValue(coreId * slotsToWrite * 2 + i * 2 + 1, accumCycles_[i]);
        }
    }

private:
    uint64_t startCycles_[MAX_SLOTS];
    uint64_t accumCycles_[MAX_SLOTS];
    int32_t  currentId_;
    int32_t  numSlotsToWrite_;
    GM_ADDR  gmOutputAddr_;
};

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
