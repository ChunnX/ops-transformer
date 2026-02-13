/*!
 * \file dua_quantize_add_layer_norm.cpp
 * \brief
 */
#include "dua_quantize_add_layer_norm_single_row_kernel.h"

extern "C" __global__ __aicore__ void dua_quantize_add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2,
    GM_ADDR zeroPoints1, GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1000)) {
        KernelDuaQuantizeAddLayerNormSingleRow<bfloat16_t, 1000> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        KernelDuaQuantizeAddLayerNormSingleRow<bfloat16_t, 1001> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales1, scales2, zeroPoints1, zeroPoints2, y1, y2, x, workspace, &tilingData);
        op.Process();
    }
}