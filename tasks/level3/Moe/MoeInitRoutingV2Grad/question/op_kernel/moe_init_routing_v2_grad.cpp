#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_init_routing_v2_grad(
    GM_ADDR gradExpandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR gradX,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
