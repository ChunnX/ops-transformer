#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void embedding_bag(
    GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets, GM_ADDR per_sample_weights, GM_ADDR y, GM_ADDR offset2bag,
    GM_ADDR bag_size, GM_ADDR max_indices, GM_ADDR workspace, GM_ADDR tiling)
{
}