#ifndef MC2_KERNEL_TIMING_H
#define MC2_KERNEL_TIMING_H

#include "kernel_operator.h"

namespace Mc2Kernel {

__aicore__ inline void WriteTimingToOutput(GM_ADDR outputAddr, int32_t coreId,
    uint64_t accum0, uint64_t accum1, bool isAicActivated = false, int32_t aicNum = 0)
{
    PipeBarrier<PIPE_ALL>();
    if (isAicActivated) {
        SyncAll<false>();
    } else {
        SyncAll<true>();
    }

    AscendC::GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer((__gm__ uint64_t *)outputAddr);

    if (isAicActivated && g_coreType == AscendC::AIV) {
        coreId = coreId + aicNum;
    }
    output.SetValue(coreId * 4 + 0, (uint64_t)0);
    output.SetValue(coreId * 4 + 1, accum0);
    output.SetValue(coreId * 4 + 2, (uint64_t)0);
    output.SetValue(coreId * 4 + 3, accum1);
}

}  // namespace Mc2Kernel

#endif  // MC2_KERNEL_TIMING_H
