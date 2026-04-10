#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_token_permute_grad(
    GM_ADDR permuted_output_grad,
    GM_ADDR sorted_indices,
    GM_ADDR input_grad,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
