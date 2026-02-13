/*!
 * \file avg_pool3_d.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
#include "avg_pool3d_ndhwc_split_c.h"

using namespace AvgPool3d;

#define DISPATCH_OP_IMPL(KernelImpl, ...)                                                                              \
    do {                                                                                                               \
        KernelImpl<__VA_ARGS__> op;                                                                                    \
        TPipe tPipe;                                                                                                   \
        op.Init(x, y, workspace, &tilingData, &tPipe);                                                                 \
        op.Process();                                                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void avg_pool3_d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
#if defined(ORIG_DTYPE_X) && ORIG_DTYPE_X == DT_FLOAT
    if (TILING_KEY_IS(10)) {
        DISPATCH_OP_IMPL(KernelAvgPool3dSplitC, float, 1);
    } 
#endif   
    
#if defined(ORIG_DTYPE_X) && ORIG_DTYPE_X == DT_FLOAT16
    if (TILING_KEY_IS(11)) {
        DISPATCH_OP_IMPL(KernelAvgPool3dSplitC, half, 1);
    }
#endif

#if defined(ORIG_DTYPE_X) && ORIG_DTYPE_X == DT_BF16 && (__CCE_AICORE__ >= 220) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    if (TILING_KEY_IS(12)) {
        DISPATCH_OP_IMPL(KernelAvgPool3dSplitC, bfloat16_t, 1);
    }
#endif
}