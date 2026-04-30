#include "kernel_operator.h"

extern "C" __global__ __aicore__ void quant_batch_matmul_v3(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
    GM_ADDR workSpace, GM_ADDR tiling)
{
    // TODO: implement quantized batch matmul kernel
}
