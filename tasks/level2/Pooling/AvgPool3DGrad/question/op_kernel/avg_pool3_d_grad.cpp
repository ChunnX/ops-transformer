#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void avg_pool3_d_grad(
    GM_ADDR orig_input_shape, GM_ADDR grads, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
}