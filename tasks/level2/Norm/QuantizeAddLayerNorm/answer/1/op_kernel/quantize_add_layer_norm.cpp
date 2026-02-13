/*!
 * \file quantize_add_layer_norm.cpp
 * \brief
 */

#include "quantize_add_layer_norm_single_row_kernel.h"

extern "C" __global__ __aicore__ void quantize_add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales, GM_ADDR zeroPoints, GM_ADDR y,
    GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    // per channel
    if (TILING_KEY_IS(3000)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3000> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2000)) {
        KernelQuantizeAddLayerNormSingleRow<float, 2000> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    } else if (TILING_KEY_IS(1000)) {
        KernelQuantizeAddLayerNormSingleRow<half, 1000> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    } else if (TILING_KEY_IS(3100)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3100> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2100)) {
        KernelQuantizeAddLayerNormSingleRow<float, 2100> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    } else if (TILING_KEY_IS(1100)) {
        KernelQuantizeAddLayerNormSingleRow<half, 1100> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    }

    // per channel
    if (TILING_KEY_IS(3002)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3002> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2002)) {
        KernelQuantizeAddLayerNormSingleRow<float, 2002> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        KernelQuantizeAddLayerNormSingleRow<half, 1002> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    } else if (TILING_KEY_IS(3102)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormSingleRow<bfloat16_t, 3102> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2102)) {
        KernelQuantizeAddLayerNormSingleRow<float, 2102> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    } else if (TILING_KEY_IS(1102)) {
        KernelQuantizeAddLayerNormSingleRow<half, 1102> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tiling_data.numCore,
            tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.firstDimPerCore,
            tiling_data.firstDimPerCoreTail, tiling_data.eps, tiling_data.aveFactor);
        op.Process();
    }
}