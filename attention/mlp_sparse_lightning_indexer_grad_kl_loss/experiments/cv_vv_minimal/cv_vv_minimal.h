/* Independent diagnostic protocol. Not included by the production/default kernel. */
#ifndef SLIG_CV_VV_MINIMAL_H
#define SLIG_CV_VV_MINIMAL_H

#include "kernel_operator.h"
#include "slig_cv_vv_build_config.h" // Generated only inside the isolated OPP copy.

#if SLIG_CV_VV_ROUNDS < 1 || SLIG_CV_VV_ROUNDS > 65536
#error "SLIG_CV_VV_ROUNDS must be in [1, 65536]"
#endif
#if SLIG_CV_VV_VC_FLAG != 0 || (SLIG_CV_VV_VV_FLAG != 0 && SLIG_CV_VV_VV_FLAG != 15)
#error "The controlled comparison is VC=0, with VV=0 (same) or VV=15 (split)"
#endif

namespace SlIgCvVvMinimal {
__aicore__ inline void Process()
{
    if ASCEND_IS_AIV {
#pragma nounroll
        for (uint32_t iteration = 0; iteration < SLIG_CV_VV_ROUNDS; ++iteration) {
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SLIG_CV_VV_VC_FLAG);
            AscendC::CrossCoreSetFlag<0, PIPE_MTE2>(SLIG_CV_VV_VV_FLAG);
            AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(SLIG_CV_VV_VV_FLAG);
        }
    } else if ASCEND_IS_AIC {
#pragma nounroll
        for (uint32_t iteration = 0; iteration < SLIG_CV_VV_ROUNDS; ++iteration) {
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(SLIG_CV_VV_VC_FLAG);
        }
    }
    // Deliberately no SyncAll, ACK, data movement, local barriers or GM tracing.
}
} // namespace SlIgCvVvMinimal
#endif
