/*!
 * \file avg_pool3_d_grad.cpp
 * \brief
 */
#include "avg_pool3_d_grad_no_cast.h"

using namespace AvgPool3DGrad;

extern "C" __global__ __aicore__ void avg_pool3_d_grad(
    GM_ADDR orig_input_shape, GM_ADDR grads, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(2000)) {
        AvgPool3DGrad::KernelAvgPool3DGradNoCast<float> op;
        op.Init(grads, output, tiling_data, userWorkspace);
        op.Process();
    }
}