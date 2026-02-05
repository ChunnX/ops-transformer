#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void quantize_add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales, GM_ADDR zeroPoints,
    GM_ADDR y, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
}
