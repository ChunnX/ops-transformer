#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void upsample_nearest2d_grad(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
}
