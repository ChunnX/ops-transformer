/*!
 * \file mul_addn.cpp
 * \brief mul_addn kernel
 */

#include "kernel_operator.h"
#include "mul_addn_align_bf16.h"

extern "C" __global__ __aicore__ void mul_addn(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelMulAddnAlignF16<half, half> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(2)) {
        TPipe pipe;
        KernelMulAddnAlignF16<bfloat16_t, float> op;
        op.Init(x1, x2, y, workspace, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
#endif
    }
}