#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t K_MAX_SHAPE_DIM = 0;

extern "C" __global__ __aicore__ void weight_quant_batch_matmul_v2(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquant_scale, GM_ADDR antiquant_offset,
    GM_ADDR quant_scale, GM_ADDR quant_offset, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
}
