#include "kernel_operator.h"

extern "C" __global__ __aicore__ void grouped_matmul(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
    GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
    GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // TODO: 实现完整的 GroupedMatmul kernel 计算逻辑
    // 支持：BF16/FP16 非量化、INT8 量化、INT4 量化、反量化等多种模式
    // 通过 tiling key 选择不同的计算路径
}
