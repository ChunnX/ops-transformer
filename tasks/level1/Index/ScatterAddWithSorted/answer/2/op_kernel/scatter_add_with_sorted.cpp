#include "scatter_add_float_with_sorted.h"

#define CALL_OP_IMPL_FLOAT(...)                                              \
    do {                                                                     \
        KernelScatterAddFloatWithSorted<__VA_ARGS__> op;                     \
        op.Init(tilingDevice, &pipe, var, value, sorted_index, pos, output); \
        op.Process();                                                        \
    } while (0)

extern "C" __global__ __aicore__ void scatter_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterAddWithSortedTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(11)) {
        CALL_OP_IMPL_FLOAT(float);
    } else if (TILING_KEY_IS(12)) {
        CALL_OP_IMPL_FLOAT(half);
    } else if (TILING_KEY_IS(16)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        CALL_OP_IMPL_FLOAT(bfloat16_t);
#endif
    }
}