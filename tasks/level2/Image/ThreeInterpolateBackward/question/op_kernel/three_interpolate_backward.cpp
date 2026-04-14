#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void three_interpolate_backward(GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y, GM_ADDR workspace, GM_ADDR tiling) {
}
