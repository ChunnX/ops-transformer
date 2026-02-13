#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void adaptive_avg_pool3d_grad(
    GM_ADDR y_grad, GM_ADDR x, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
}