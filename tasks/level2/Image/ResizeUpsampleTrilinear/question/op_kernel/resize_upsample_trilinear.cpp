#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void resize_upsample_trilinear(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
}
