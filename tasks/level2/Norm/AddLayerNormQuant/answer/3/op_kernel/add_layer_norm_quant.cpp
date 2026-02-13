#include "add_layer_norm_static_quant_single_row_kernel.h"

extern "C" __global__ __aicore__ void add_layer_norm_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2,
    GM_ADDR zeroOffset1, GM_ADDR zeroOffset2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                                                        \
    op.Init(                                                                                                    \
        x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x, outScale1, outScale2, \
        usrWorkspace, &tilingData);                                                                             \
    op.Process()

    if (TILING_KEY_IS(1020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<half, 1020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<half, 1020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(1021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<half, 1021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<half, 1021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(1022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<half, 1022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<half, 1022> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<float, 2020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<float, 2020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<float, 2021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<float, 2021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<float, 2022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<float, 2022> op(&pipe);
            INIT_AND_PROCESS;
        }
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(3020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<bfloat16_t, 3020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<bfloat16_t, 3020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<bfloat16_t, 3021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<bfloat16_t, 3021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRow<bfloat16_t, 3022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRow<bfloat16_t, 3022> op(&pipe);
            INIT_AND_PROCESS;
        }
#endif
    }
}