#include "scatter_with_sorted.h"

#define CALL_OP_IMPL_SCATTER(...)                                            \
    do {                                                                     \
        KernelScatterWithSorted<__VA_ARGS__> op;                             \
        op.Init(tilingDevice, &pipe, var, value, sorted_index, pos, output); \
        op.Process();                                                        \
    } while (0)

extern "C" __global__ __aicore__ void scatter_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterAddWithSortedTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CALL_OP_IMPL_SCATTER(float);
    } else if (TILING_KEY_IS(2)) {
        CALL_OP_IMPL_SCATTER(half);
    } else if (TILING_KEY_IS(3)) {
        CALL_OP_IMPL_SCATTER(int);
    } else if (TILING_KEY_IS(4)) {
        CALL_OP_IMPL_SCATTER(uint8_t);
    } else if (TILING_KEY_IS(5)) {
        CALL_OP_IMPL_SCATTER(int8_t);
    } else if (TILING_KEY_IS(6)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        CALL_OP_IMPL_SCATTER(bfloat16_t);
#endif
    }
}