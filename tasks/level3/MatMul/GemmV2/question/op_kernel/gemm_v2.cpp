#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void gemm_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR alpha, GM_ADDR beta, GM_ADDR ref_c,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
}
