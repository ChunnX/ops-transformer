#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void scaled_masked_softmax_grad_v2(
    const GM_ADDR yGrad,
    const GM_ADDR y,
    const GM_ADDR mask,
    const GM_ADDR xGrad,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
