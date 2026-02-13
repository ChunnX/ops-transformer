#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void max_pool3d_grad_with_argmax(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}