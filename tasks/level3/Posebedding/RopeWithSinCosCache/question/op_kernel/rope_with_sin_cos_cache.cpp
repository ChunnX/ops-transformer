#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void rope_with_sin_cos_cache(
    GM_ADDR position_id, GM_ADDR query_in, GM_ADDR key_in, GM_ADDR cos_sin_cache,
    GM_ADDR query_out, GM_ADDR key_out, GM_ADDR workspace, GM_ADDR tiling) {
}
