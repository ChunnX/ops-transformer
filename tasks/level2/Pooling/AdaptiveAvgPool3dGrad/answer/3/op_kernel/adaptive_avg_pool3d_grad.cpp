/*!
 * \file adaptive_avg_pool3d_grad.cpp
 * \brief
 */

#include "adaptive_avg_pool3d_grad_nc_large_float.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void adaptive_avg_pool3d_grad(
    GM_ADDR y_grad, GM_ADDR x, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelAdaptiveAvgPool3DGradNcLargeFloat<float> op;
        op.Init(y_grad, x_grad, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    }
}