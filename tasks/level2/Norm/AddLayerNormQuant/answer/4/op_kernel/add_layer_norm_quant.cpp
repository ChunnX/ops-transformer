#include "add_layer_norm_static_quant_normal_kernel.h"

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

    if (TILING_KEY_IS(1000)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1001)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1002)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1002> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2001)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2002)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2002> op(&pipe);
        INIT_AND_PROCESS;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(3000)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3001)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3002)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3002> op(&pipe);
        INIT_AND_PROCESS;
#endif
    }
}