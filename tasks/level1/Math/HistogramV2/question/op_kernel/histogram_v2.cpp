#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void histogram_v2(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}