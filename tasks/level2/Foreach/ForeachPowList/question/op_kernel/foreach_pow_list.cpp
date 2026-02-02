#include "kernel_operator.h"

using namespace AscendC;
// 核函数入口
extern "C" __global__ __aicore__ void foreach_pow_list(
    GM_ADDR self, GM_ADDR exponent, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {

}
