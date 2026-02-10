#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void apply_top_k_top_p_with_sorted(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
}