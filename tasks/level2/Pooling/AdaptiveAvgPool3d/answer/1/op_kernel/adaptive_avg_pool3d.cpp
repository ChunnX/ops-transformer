/*!
 * \file adaptive_avg_pool3d.cpp
 * \brief
 */

#include "kernel_operator.h"

#include "adaptive_avg_pool3d_split_c.h"

using namespace AdaptiveAvgPool3d;

#define DISPATCH_OP_IMPL(KernelImpl, ...)              \
    do {                                               \
        KernelImpl<__VA_ARGS__> op;                    \
        TPipe tPipe;                                   \
        op.Init(x, y, workspace, &tilingData, &tPipe); \
        op.Process();                                  \
    } while (0)

extern "C" __global__ __aicore__ void adaptive_avg_pool3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(11)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, half, 1);
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(10)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, bfloat16_t, 1);
#endif
    } else if (TILING_KEY_IS(12)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dSplitC, float, 1);
    }
}