#include "kernel_operator.h"

extern "C" __global__ __aicore__ void quantized_batch_norm(
    GM_ADDR x, GM_ADDR mean, GM_ADDR var, GM_ADDR input_scale, GM_ADDR input_zero_point, GM_ADDR output_scale,
    GM_ADDR output_zero_point, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // TODO: Implement quantized batch norm kernel
}
