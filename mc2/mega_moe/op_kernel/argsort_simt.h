#ifndef ARGSORT_SIMT
#define ARGSORT_SIMT
 
#include "kernel_operator.h"
#include "gmmsq_base.h"
__simt_vf__ __aicore__ LAUNCH_BOUND(MoeInitRoutingV3::SIMT_THREAD_NUM) inline void ArgsortSimt(int64_t elements, int64_t indexBase,
                                            __gm__ int32_t *sortedExpertIndicesGmAddr,
                                            __gm__ int32_t *expandedRowIdxGmAddr) {
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(elements);
        index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int64_t outIndices = sortedExpertIndicesGmAddr[indexBase + index];
        expandedRowIdxGmAddr[outIndices] = indexBase + index;
    }
}
 
#endif