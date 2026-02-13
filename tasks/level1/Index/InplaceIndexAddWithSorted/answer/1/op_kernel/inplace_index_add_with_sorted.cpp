/*!
 * \file inplace_index_add_with_sorted.cpp
 * \brief
 */

#include "inplace_index_add_with_sorted_avg.h"

extern "C" __global__ __aicore__ void inplace_index_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos, GM_ADDR alpha, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 使能VectorCore

#define INIT_AND_PROCESS                             \
    op.Init(var, value, sorted_indices, pos, alpha); \
    op.Process()
    if (TILING_KEY_IS(1)) {
        // InplaceIndexAddWithSorted FLOAT axis = 0, AVG index on each core
        InplaceIndexAddWithSortedAvg<float> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(4)) {
        // InplaceIndexAddWithSorted INT16 axis = 0, AVG index on each core
        InplaceIndexAddWithSortedAvg<int16_t> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(5)) {
        // InplaceIndexAddWithSorted INT32 axis = 0, AVG index on each core
        InplaceIndexAddWithSortedAvg<int32_t> op(&pipe, &tilingData);
        INIT_AND_PROCESS;
    }
}