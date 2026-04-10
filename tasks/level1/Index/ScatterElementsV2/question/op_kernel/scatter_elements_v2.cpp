#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void scatter_elements_v2(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
}
