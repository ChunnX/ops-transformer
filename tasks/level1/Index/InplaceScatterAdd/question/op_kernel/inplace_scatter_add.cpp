#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void inplace_scatter_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
}