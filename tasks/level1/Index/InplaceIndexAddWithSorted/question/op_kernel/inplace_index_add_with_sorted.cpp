#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void inplace_index_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos, GM_ADDR alpha, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
}