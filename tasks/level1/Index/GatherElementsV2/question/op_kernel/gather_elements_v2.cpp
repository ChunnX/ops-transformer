#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void gather_elements_v2(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}
