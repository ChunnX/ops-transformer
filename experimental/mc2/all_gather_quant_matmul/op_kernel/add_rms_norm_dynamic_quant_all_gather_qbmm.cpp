/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/*!
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm.cpp
 * \brief Kernel entry: AllGather + Matmul fusion
 *
 * Flow:
 *   AIV: CopyLocalData → SetRemoteFlag → WaitRemoteFlag → ExecuteAllGather
 *   AIC: MatmulProcess (TensorAPI MX quantized matmul)
 *
 * Inputs: pre-quantized fp8 A per rank, fp8 B, e8m0 scales
 * Output: bf16 C per rank
 */

#include "basic_api/kernel_basic_intf.h"
#include "kernel/all_gather_matmul_impl.h"
#include "tiling/all_gather_matmul_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void
add_rms_norm_dynamic_quant_all_gather_qbmm(GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma,
                                           GM_ADDR scaleInv1, GM_ADDR scaleInv2, GM_ADDR smooth_scale, GM_ADDR bias,
                                           GM_ADDR output, GM_ADDR z,
                                           GM_ADDR dynamic_quant_out, GM_ADDR all_gather_data_out,
                                           GM_ADDR all_gather_scales_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(allGatherMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(allGatherMatmulTilingData, tilingData, tilingGM);

    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

        // AllGather + Matmul: Copy pre-quantized fp8 A + e8m0 scaleA to win zone
        allGatherMatmulImpl::allGatherMatmulImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e8m0_t, 0, true, false> impl;
        impl.Init(x1, x2, residual, y, gamma, scaleInv1, smooth_scale, bias, scaleInv2, output, z, dynamic_quant_out,
                  all_gather_data_out, all_gather_scales_out, tilingGM);
        impl.Process();
        return;
    }
}
