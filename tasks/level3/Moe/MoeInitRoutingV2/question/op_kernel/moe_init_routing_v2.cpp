#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_init_routing_v2(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
}
