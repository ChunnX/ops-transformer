#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void embedding_dense_grad_v2(GM_ADDR grad, GM_ADDR sortIndices,
    GM_ADDR posIdx, GM_ADDR backProps, GM_ADDR workSpace, GM_ADDR tiling) {
}