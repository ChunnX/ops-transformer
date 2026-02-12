#include "scatter_add_int_with_sorted.h"

#define CALL_OP_IMPL_INT(...)                                           \
    do {                                                                \
        KernelScatterAddIntWithSorted<__VA_ARGS__> op;                  \
        op.Init(tilingDevice, &pipe, var, value, sorted_index, output); \
        op.Process();                                                   \
    } while (0)

extern "C" __global__ __aicore__ void scatter_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterAddWithSortedTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(13)) {
        CALL_OP_IMPL_INT(int);
    } else if (TILING_KEY_IS(14)) {
        CALL_OP_IMPL_INT(uint8_t);
    } else if (TILING_KEY_IS(15)) {
        CALL_OP_IMPL_INT(int8_t);
    }
}