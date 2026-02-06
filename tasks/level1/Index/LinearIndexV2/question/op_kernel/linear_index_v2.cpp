#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void linear_index_v2(
    GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling)
{
}