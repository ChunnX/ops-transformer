/*!
 * \file quantize_add_layer_norm.cpp
 * \brief
 */

#include "quantize_add_layer_norm_normal_kernel.h"

extern "C" __global__ __aicore__ void quantize_add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales, GM_ADDR zeroPoints, GM_ADDR y,
    GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    // per_tensor: *100 = MUL(axis=-65535), *101 = DIV(axis=-1) + additional_output
    if (TILING_KEY_IS(3100)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormNormalPerTensorKernel<bfloat16_t, float, 3101> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2100)) {
        KernelQuantizeAddLayerNormNormalPerTensorKernel<float, float, 2101> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
    } else if (TILING_KEY_IS(1100)) {
        KernelQuantizeAddLayerNormNormalPerTensorKernel<half, float, 1101> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
    } else if (TILING_KEY_IS(3001)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormNormalPerTensorKernel<bfloat16_t, float, 3001> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2001)) {
        KernelQuantizeAddLayerNormNormalPerTensorKernel<float, float, 2001> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        KernelQuantizeAddLayerNormNormalPerTensorKernel<half, float, 1001> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
    } else if (TILING_KEY_IS(3101)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        KernelQuantizeAddLayerNormNormalPerTensorKernel<bfloat16_t, float, 3101> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2101)) {
        KernelQuantizeAddLayerNormNormalPerTensorKernel<float, float, 2101> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
    } else if (TILING_KEY_IS(1101)) {
        KernelQuantizeAddLayerNormNormalPerTensorKernel<half, float, 1101> op(&pipe);
        op.Init(
            x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, tiling_data.numCore, tiling_data.numLastDim,
            tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail, tiling_data.eps,
            tiling_data.aveFactor, tiling_data.isDivMode);
        op.Process();
    }
}