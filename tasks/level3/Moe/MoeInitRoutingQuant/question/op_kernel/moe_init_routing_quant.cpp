#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_init_routing_quant(
    GM_ADDR x,
    GM_ADDR rowIdx,
    GM_ADDR expertIdx,
    GM_ADDR expandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR expandedExpertIdx,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
