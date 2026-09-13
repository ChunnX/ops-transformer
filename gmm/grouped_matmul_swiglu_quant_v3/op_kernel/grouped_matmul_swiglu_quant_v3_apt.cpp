/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend 950 MXFP8 weight-NZ kernel entry for GroupedMatmulSwigluQuantV3.
 */

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "arch35/grouped_matmul_swiglu_quant_v3_mxquant.h"
#include "arch35/grouped_matmul_swiglu_quant_v3_tiling_key.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_v3(
    GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight, GM_ADDR weightScale,
    GM_ADDR weightAssistanceMatrix, GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    // The CGMCT kernel follows the V2 mixed-core ABI: resolve the user
    // workspace before constructing the AIC/AIV pipeline context.
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    (void)userWorkspace;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // Keep the MXFP8 CGMCT execution state consistent with V2: intermediate
    // BF16/FP32 operations must not raise an overflow exception for values
    // that are intentionally handled by the MX quantization epilogue.
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_MODE_0)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_MODE_0, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 0>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_MODE_1)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_MODE_1, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 1>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_MODE_2)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_MODE_2, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 2>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_MODE_3)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_MODE_3, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz, 3>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_TRANS_MODE_0)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_TRANS_MODE_0, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 0>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_TRANS_MODE_1)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_TRANS_MODE_1, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 1>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_TRANS_MODE_2)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_TRANS_MODE_2, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 2>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    } else if (TILING_KEY_IS(GMM_SWIGLU_QUANT_V3_TRANS_MODE_3)) {
        KERNEL_TASK_TYPE(GMM_SWIGLU_QUANT_V3_TRANS_MODE_3, KERNEL_TYPE_MIX_AIC_1_2);
        GroupedMatmulSwigluQuantV3::GmmSwigluAswtV3<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn, 3>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling);
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
