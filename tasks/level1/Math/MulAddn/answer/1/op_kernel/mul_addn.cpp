/*!
 * \file mul_addn.cpp
 * \brief mul_addn kernel
 */

#include "kernel_operator.h"
#include "mul_addn_align.h"

extern "C" __global__ __aicore__ void mul_addn(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        KernelMulAddnAlign<float> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
    }
}