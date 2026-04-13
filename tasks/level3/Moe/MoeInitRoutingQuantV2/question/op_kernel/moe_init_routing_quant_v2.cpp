#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_init_routing_quant_v2(
    GM_ADDR x,
    GM_ADDR expertIdx,
    GM_ADDR scale,
    GM_ADDR offset,
    GM_ADDR expandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum,
    GM_ADDR expertTokensBeforeCapacity,
    GM_ADDR dynamicQuantScale,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
