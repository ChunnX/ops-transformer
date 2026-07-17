#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

static constexpr int32_t TIMING_MAX_SLOTS = 16;

inline uint64_t g_timingStart[TIMING_MAX_SLOTS];
inline uint64_t g_timingAccum[TIMING_MAX_SLOTS];
inline int32_t  g_timingCurrentId = 0;
inline int32_t  g_timingNumSlots = TIMING_MAX_SLOTS;
inline GM_ADDR  g_timingOutputAddr = nullptr;

__aicore__ inline void KernelTimingInit(GM_ADDR outputAddr, int32_t numSlotsToWrite = TIMING_MAX_SLOTS)
{
    g_timingOutputAddr = outputAddr;
    g_timingCurrentId = 0;
    g_timingNumSlots = numSlotsToWrite;
    for (int32_t i = 0; i < TIMING_MAX_SLOTS; i++) {
        g_timingStart[i] = 0;
        g_timingAccum[i] = 0;
    }
}

__aicore__ inline void KernelTimingStart(int32_t slotId = -1)
{
    if (slotId >= 0) {
        g_timingCurrentId = slotId;
    } else {
        g_timingCurrentId++;
    }
    if (g_timingCurrentId < 0 || g_timingCurrentId >= TIMING_MAX_SLOTS) {
        return;
    }
    g_timingStart[g_timingCurrentId] = AscendC::GetSystemCycle();
}

__aicore__ inline void KernelTimingStop(int32_t slotId = -1)
{
    if (slotId >= 0) {
        g_timingCurrentId = slotId;
    }
    if (g_timingCurrentId < 0 || g_timingCurrentId >= TIMING_MAX_SLOTS) {
        return;
    }
    g_timingAccum[g_timingCurrentId] += AscendC::GetSystemCycle() - g_timingStart[g_timingCurrentId];
}

__aicore__ inline void KernelTimingWrite(int32_t coreId, bool isAicActivated = false, int32_t aicNum = 0)
{
    PipeBarrier<PIPE_ALL>();
    if (isAicActivated) {
        SyncAll<false>();
    } else {
        SyncAll<true>();
    }

    AscendC::GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer((__gm__ uint64_t *)g_timingOutputAddr);

    if (isAicActivated && g_coreType == AscendC::AIV) {
        coreId = coreId + aicNum;
    }
    int32_t slotsToWrite = (g_timingNumSlots > 0 && g_timingNumSlots <= TIMING_MAX_SLOTS) ? g_timingNumSlots : TIMING_MAX_SLOTS;
    for (int32_t i = 0; i < slotsToWrite; i++) {
        output.SetValue(coreId * slotsToWrite * 2 + i * 2, (uint64_t)0);
        output.SetValue(coreId * slotsToWrite * 2 + i * 2 + 1, g_timingAccum[i]);
    }
}

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
