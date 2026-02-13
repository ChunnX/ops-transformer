#include "linear_index.h"

#define CALL_OP_IMPL(...)                              \
    do {                                               \
        KernelLinearIndex<__VA_ARGS__> op;             \
        op.Init(tilingDevice, &pipe, indices, output); \
        op.Process();                                  \
    } while (0)

extern "C" __global__ __aicore__ void linear_index(
    GM_ADDR indices, GM_ADDR var, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const LinearIndexTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CALL_OP_IMPL(int, 0);
    } else if (TILING_KEY_IS(2)) {
        CALL_OP_IMPL(int64_t, 0);
    } else if (TILING_KEY_IS(11)) {
        CALL_OP_IMPL(int, 1);
    } else if (TILING_KEY_IS(12)) {
        CALL_OP_IMPL(int64_t, 1);
    } else if (TILING_KEY_IS(21)) {
        CALL_OP_IMPL(int, 2);
    } else if (TILING_KEY_IS(22)) {
        CALL_OP_IMPL(int64_t, 2);
    }
}