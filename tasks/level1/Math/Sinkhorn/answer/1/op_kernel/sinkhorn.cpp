/*!
 * \file sinkhorn.cpp
 * \brief
 */
#include "sinkhorn.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sinkhorn(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // 这两句都不要删除，有些调用框架还不能适配，需要保留
    SetSysWorkspace(workspace);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针。

    if (TILING_KEY_IS(0)) {
        // ge::DT_FLOAT
        AscendC::KernelSinkhorn<float, float> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // ge::DT_FLOAT16
        AscendC::KernelSinkhorn<half, half> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(27)) {
        // ge::DT_BFLOAT16
        AscendC::KernelSinkhorn<float, bfloat16_t> op;
        op.Init(cost, p, usrWorkspace, &tiling_data);
        op.Process();
    }
}