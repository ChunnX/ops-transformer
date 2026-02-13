/*!
 * \file apply_top_k_top_p_with_sorted.cpp
 * \brief
 */

#include "apply_top_k_top_p_with_sorted.h"
using namespace AscendC;
using namespace ApplyTopKTopPWithSortedOp;

extern "C" __global__ __aicore__ void apply_top_k_top_p_with_sorted(GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        ApplyTopKTopPWithSortedOp::ApplyTopKTopPWithSorted<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out, workSpace);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ApplyTopKTopPWithSortedOp::ApplyTopKTopPWithSorted<DTYPE_OUT, float, DTYPE_OUT> op;
        op.InitTilingData(tilingData, sorted_value, sorted_indices, p, k, out, workSpace);
        op.InitBuffer(&pipe);
        op.ProcessTopK();
    }
}