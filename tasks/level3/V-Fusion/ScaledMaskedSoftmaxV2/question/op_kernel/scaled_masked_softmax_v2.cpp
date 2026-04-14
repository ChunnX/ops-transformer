#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void scaled_masked_softmax_v2(
    GM_ADDR x,
    GM_ADDR mask,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
