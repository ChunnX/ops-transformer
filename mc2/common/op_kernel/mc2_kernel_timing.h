#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

class KernelTiming {
public:
    static constexpr int32_t MAX_SLOTS = 16;
    static constexpr uint32_t BYTES_PER_CORE = MAX_SLOTS * 2 * sizeof(uint64_t);

    __aicore__ inline KernelTiming() : gmOutputAddr_(nullptr) {}

    __aicore__ inline void Init(GM_ADDR outputAddr)
    {
        gmOutputAddr_ = outputAddr;
        accumulatedSlotMask_ = 0;
        activeAccumulatedSlotMask_ = 0;
        for (int32_t i = 0; i < MAX_SLOTS; ++i) {
            startCycles_[i] = 0;
            stopCycles_[i] = 0;
            accumulatedStartCycles_[i] = 0;
        }
    }

    __aicore__ inline void StartTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        startCycles_[slotId] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void StopTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        stopCycles_[slotId] = AscendC::GetSystemCycle();
    }

    __aicore__ inline void StartAccumulatedTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        uint32_t slotMask = 1U << slotId;
        if ((activeAccumulatedSlotMask_ & slotMask) != 0) {
            return;
        }
        uint64_t now = AscendC::GetSystemCycle();
        if ((accumulatedSlotMask_ & slotMask) == 0) {
            startCycles_[slotId] = now;
            stopCycles_[slotId] = 0;
            accumulatedSlotMask_ |= slotMask;
        }
        accumulatedStartCycles_[slotId] = now;
        activeAccumulatedSlotMask_ |= slotMask;
    }

    __aicore__ inline void StopAccumulatedTiming(int32_t slotId)
    {
        if (slotId < 0 || slotId >= MAX_SLOTS) {
            return;
        }
        uint32_t slotMask = 1U << slotId;
        if ((activeAccumulatedSlotMask_ & slotMask) == 0) {
            return;
        }
        stopCycles_[slotId] += AscendC::GetSystemCycle() - accumulatedStartCycles_[slotId];
        activeAccumulatedSlotMask_ &= ~slotMask;
    }

    __aicore__ inline void WriteTimingToOutput(int32_t coreId, bool isAicActivated = false, int32_t aicNum = 0)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (isAicActivated) {
            AscendC::SyncAll<false>();
        } else {
            AscendC::SyncAll<true>();
        }

        AscendC::GlobalTensor<uint64_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(gmOutputAddr_));
        if (isAicActivated && g_coreType == AscendC::AIV) {
            coreId += aicNum;
        }

        for (int32_t i = 0; i < MAX_SLOTS; ++i) {
            int32_t outputOffset = coreId * MAX_SLOTS * 2 + i * 2;
            uint64_t outputStop = stopCycles_[i];
            if ((accumulatedSlotMask_ & (1U << i)) != 0) {
                outputStop = startCycles_[i] + stopCycles_[i];
            }
            output.SetValue(outputOffset, startCycles_[i]);
            output.SetValue(outputOffset + 1, outputStop);
        }
    }

private:
    uint64_t startCycles_[MAX_SLOTS];
    uint64_t stopCycles_[MAX_SLOTS];
    uint64_t accumulatedStartCycles_[MAX_SLOTS];
    uint32_t accumulatedSlotMask_;
    uint32_t activeAccumulatedSlotMask_;
    GM_ADDR gmOutputAddr_;
};

} // namespace Mc2Kernel

#endif // MC2_KERNEL_TIMING_H
