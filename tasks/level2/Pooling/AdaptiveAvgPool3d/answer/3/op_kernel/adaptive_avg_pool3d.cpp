/*!
 * \file adaptive_avg_pool3d.cpp
 * \brief
 */

#include "kernel_operator.h"

#include "adaptive_avg_pool3d_multi_w.h"

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
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    if (TILING_KEY_IS(30)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, bfloat16_t, 1);
    } else 
#endif
    if (TILING_KEY_IS(31)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, half, 1);
    } else if (TILING_KEY_IS(32)) {
        DISPATCH_OP_IMPL(KernelAdaptiveAvgPool3dMultiW, float, 1);
    }
}