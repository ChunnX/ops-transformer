/*!
 * \file adaptive_max_pool3d_grad.cpp
 * \brief
 */

#include "adaptive_max_pool3d_grad_normal.h"

using namespace AdaptiveMaxPool3DGrad;
#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        GET_TILING_DATA(tilingData, tiling);                 \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.Process();                                        \
    } while (0)
extern "C" __global__ __aicore__ void adaptive_max_pool3d_grad(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    if (TILING_KEY_IS(0)) { // Normal New Kernel
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, true);
    }
}