#include "add_layer_norm_dynamic_quant_single_row_kernel.h"

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

    if (TILING_KEY_IS(1120)) {
        KernelAddLayerNormDynamicQuantSingleRow<half, 1120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1121)) {
        KernelAddLayerNormDynamicQuantSingleRow<half, 1121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1122)) {
        KernelAddLayerNormDynamicQuantSingleRow<half, 1122> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2120)) {
        KernelAddLayerNormDynamicQuantSingleRow<float, 2120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2121)) {
        KernelAddLayerNormDynamicQuantSingleRow<float, 2121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2122)) {
        KernelAddLayerNormDynamicQuantSingleRow<float, 2122> op(&pipe);
        INIT_AND_PROCESS;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(3120)) {
        KernelAddLayerNormDynamicQuantSingleRow<bfloat16_t, 3120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3121)) {
        KernelAddLayerNormDynamicQuantSingleRow<bfloat16_t, 3121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3122)) {
        KernelAddLayerNormDynamicQuantSingleRow<bfloat16_t, 3122> op(&pipe);
        INIT_AND_PROCESS;
#endif
    }
}