#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void linear_index(
    GM_ADDR indices, GM_ADDR var, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
}