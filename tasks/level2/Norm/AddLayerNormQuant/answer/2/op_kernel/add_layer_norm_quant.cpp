#include "add_layer_norm_dynamic_quant_normal_kernel.h"

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

    if (TILING_KEY_IS(1100)) {
        KernelAddLayerNormDynamicQuantNormal<half, 1100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1101)) {
        KernelAddLayerNormDynamicQuantNormal<half, 1101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1102)) {
        KernelAddLayerNormDynamicQuantNormal<half, 1102> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2100)) {
        KernelAddLayerNormDynamicQuantNormal<float, 2100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2101)) {
        KernelAddLayerNormDynamicQuantNormal<float, 2101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2102)) {
        KernelAddLayerNormDynamicQuantNormal<float, 2102> op(&pipe);
        INIT_AND_PROCESS;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(3100)) {
        KernelAddLayerNormDynamicQuantNormal<bfloat16_t, 3100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3101)) {
        KernelAddLayerNormDynamicQuantNormal<bfloat16_t, 3101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3102)) {
        KernelAddLayerNormDynamicQuantNormal<bfloat16_t, 3102> op(&pipe);
        INIT_AND_PROCESS;
#endif
    }
}