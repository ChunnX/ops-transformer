#include "max_pool3d_grad_with_argmax_scatter_overlap.h"

using namespace MaxPool3DGradWithArgmax;

#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        templateClass<__VA_ARGS__> op(&pipe);                \
        op.Init(x, grad, argmax, y, workspace, &tilingData); \
        op.Process();                                        \
    } while (0)

extern "C" __global__ __aicore__ void max_pool3d_grad_with_argmax(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(102)) {
        GENERAL_OP_IMPL(MaxPoolGradWithArgScatterOverlap, DTYPE_X, DTYPE_GRAD, int32_t, DTYPE_Y);
    }
    return;
}