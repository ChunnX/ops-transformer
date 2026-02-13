#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void repeat_interleave_grad(
    GM_ADDR input_grad, GM_ADDR repeats, GM_ADDR output_grad, GM_ADDR workspace, GM_ADDR tiling)
{
}