#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void grouped_bias_add_grad(
    GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias, GM_ADDR workspace, GM_ADDR tiling_data)
{
    
}