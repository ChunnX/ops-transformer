#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void batch_mat_mul_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
}
