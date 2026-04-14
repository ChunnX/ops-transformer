#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void upsample_bilinear2d_aa_backward(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
}
